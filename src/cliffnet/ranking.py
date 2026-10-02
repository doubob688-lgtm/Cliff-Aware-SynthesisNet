"""Bradley-Terry route ranking from frozen representations and binary labels."""

import copy
import itertools
import json
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F


class PairwiseRanker(nn.Module):
    def __init__(self, width):
        super().__init__()
        # Bias cancels from score_A - score_B and is unnecessary.
        self.score = nn.Linear(width, 1, bias=False)
        nn.init.zeros_(self.score.weight)

    def forward(self, z_tree):
        return self.score(z_tree).squeeze(-1)

    def preference_logits(self, a, b):
        return self(a) - self(b)


def pair_tensors(pairs, route_index):
    return (torch.tensor([route_index[p["tree_A"]] for p in pairs], dtype=torch.long),
            torch.tensor([route_index[p["tree_B"]] for p in pairs], dtype=torch.long),
            torch.tensor([p["preference_label"] for p in pairs], dtype=torch.float32))


def metrics(pairs, probabilities):
    if not pairs:
        return {"n": 0, "accuracy": None, "roc_auc": None}
    labels = np.array([p["preference_label"] for p in pairs])
    probabilities = np.asarray(probabilities)
    tied = probabilities == 0.5
    return {"n": len(pairs), "accuracy": float(((probabilities >= 0.5) == labels).mean()),
            "accuracy_ties_half": float(np.where(tied, 0.5, (probabilities > 0.5) == labels).mean()),
            "ties": int(tied.sum()), "roc_auc": float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None,
            "positive_labels": int(labels.sum())}


def multi_candidate_groups(pairs):
    """Complete three-halide groups, ranked using binary pair labels only.

    All three unordered comparisons must survive the preference-gap filter;
    this avoids assuming an ordering for experimentally indistinguishable rows.
    Ordinal relevance is the number of pairwise wins (0, 1, 2), never raw yield.
    """
    buckets = defaultdict(list)
    for p in pairs:
        if p["changed_component"] == "Aryl halide":
            buckets[json.dumps(p["target_context"], sort_keys=True)].append(p)
    groups = []
    for edges in buckets.values():
        routes = sorted({p[k] for p in edges for k in ("tree_A", "tree_B")})
        unordered = {tuple(sorted((p["tree_A"], p["tree_B"]))) for p in edges}
        if len(routes) != 3 or len(edges) != 3 or len(unordered) != 3:
            continue
        wins = {r: 0 for r in routes}
        for p in edges:
            wins[p["tree_A"] if p["preference_label"] else p["tree_B"]] += 1
        if sorted(wins.values()) != [0, 1, 2]:
            raise ValueError("Cyclic three-candidate preferences cannot define relevance")
        groups.append({"target_context": edges[0]["target_context"], "routes": routes,
                       "ordinal_relevance": [wins[r] for r in routes], "split": edges[0]["split"]})
    return groups


def group_metrics(groups, route_scores):
    """NDCG@3 with linear ordinal gains, MRR of unique best; average score ties."""
    ndcg, mrr = [], []
    for group in groups:
        scores = [route_scores[r] for r in group["routes"]]
        relevance = group["ordinal_relevance"]
        discounts = np.log2(np.arange(3) + 2)
        ideal = float(np.sum(np.sort(relevance)[::-1] / discounts))
        orders = [order for order in itertools.permutations(range(3))
                  if all(scores[order[i]] >= scores[order[i + 1]] for i in range(2))]
        ndcg.append(np.mean([np.sum(np.array(relevance)[list(order)] / discounts) / ideal for order in orders]))
        best = relevance.index(2)
        mrr.append(np.mean([1 / (order.index(best) + 1) for order in orders]))
    return {"valid_groups": len(groups), "ndcg_at_3": float(np.mean(ndcg)) if groups else None,
            "mrr": float(np.mean(mrr)) if groups else None, "relevance": "ordinal pairwise wins, not numerical yield",
            "tie_rule": "average all equally scored permutations"}


def fit_and_evaluate(embeddings, route_index, pairs, epochs=200, learning_rate=0.01, seed=0):
    """Fit score(z) only. Outcomes/gaps are absent from the training interface."""
    torch.manual_seed(seed)
    z = torch.tensor(embeddings, dtype=torch.float32).detach()
    groups = {s: [p for p in pairs if p["split"] == s] for s in ("train", "validation", "test")}
    if any(not p for p in groups.values()):
        raise ValueError("All three target-grouped splits need preference pairs")
    # Estimate scaling on unique training routes only, never validation/test.
    training_routes = sorted({route_index[p[k]] for p in groups["train"] for k in ("tree_A", "tree_B")})
    mean = z[training_routes].mean(0)
    scale = z[training_routes].std(0, unbiased=False).clamp_min(1e-3)
    z = (z - mean) / scale
    ranker = PairwiseRanker(z.shape[1])
    optimizer = torch.optim.AdamW(ranker.parameters(), lr=learning_rate, weight_decay=0.001)
    tensors = {s: pair_tensors(p, route_index) for s, p in groups.items()}
    best, best_loss, best_epoch = None, float("inf"), None
    history = []
    for epoch in range(1, epochs + 1):
        ranker.train()
        optimizer.zero_grad()
        scores = ranker(z)
        a, b, labels = tensors["train"]
        loss = F.binary_cross_entropy_with_logits(scores[a] - scores[b], labels)
        loss.backward()
        optimizer.step()
        ranker.eval()
        with torch.no_grad():
            scores = ranker(z)
            a, b, labels = tensors["validation"]
            validation_loss = float(F.binary_cross_entropy_with_logits(scores[a] - scores[b], labels))
        if validation_loss < best_loss:
            best, best_loss, best_epoch = copy.deepcopy(ranker.state_dict()), validation_loss, epoch
        if epoch == 1 or epoch % 20 == 0:
            history.append({"epoch": epoch, "train_pairwise_logistic_loss": float(loss), "validation_pairwise_logistic_loss": validation_loss})
    ranker.load_state_dict(best)
    report = {"best_epoch": best_epoch, "validation_logistic_loss": best_loss, "epochs": epochs,
              "learning_rate": learning_rate, "weight_decay": 0.001, "seed": seed, "history": history, "splits": {}}
    with torch.no_grad():
        scores = ranker(z)
        for split, examples in groups.items():
            a, b, _ = tensors[split]
            probabilities = torch.sigmoid(scores[a] - scores[b]).numpy()
            stratified = {"overall": metrics(examples, probabilities)}
            for name, selector in {
                "high_similarity": lambda p: p["high_similarity"],
                "route_cliff": lambda p: p["route_cliff"],
                "halide_alternative": lambda p: p["changed_component"] == "Aryl halide",
                "condition_only": lambda p: p["changed_component"] != "Aryl halide",
            }.items():
                indices = [i for i, p in enumerate(examples) if selector(p)]
                stratified[name] = metrics([examples[i] for i in indices], probabilities[indices])
            report["splits"][split] = stratified
            route_scores = {r: float(scores[i]) for r, i in route_index.items()}
            stratified["multi_candidate"] = group_metrics(multi_candidate_groups(examples), route_scores)
    saved = {"ranker_state_dict": best, "training_mean": mean, "training_scale": scale,
             "feature_width": z.shape[1], "loss": "binary_cross_entropy_with_logits(score_A-score_B, preference_label)",
             "encoder_frozen": True}
    return report, saved
