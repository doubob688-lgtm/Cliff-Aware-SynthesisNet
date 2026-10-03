"""Condition-aware single-step benchmark reusing the validated frozen cache."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch

from cliffnet.baseline import load_predictor
from cliffnet.buchwald import COMPONENTS, RAW_FILE, checksum, construct_pairs, load_and_audit, route_records, split_targets
from cliffnet.conditions import (ConditionEncoder, collision_counts, representations,
                                 training_statistics)
from cliffnet.encoder import FrozenTreeEncoder
from cliffnet.experiment import native_tree, validate_split, verify_encoder
from cliffnet.ranking import PairwiseRanker, fit_and_evaluate, multi_candidate_groups


SUBSETS = ("overall", "high_similarity", "route_cliff", "halide_alternative",
           "ligand", "base", "additive", "condition_only")
# Provenance: completed codex/pathway-ranking baseline, never generated here.
CACHE_SHA256 = "365f5e5035f4c0ed05cfdbc471d5ae9258330d5075f334da1e2e5b7772fd258f"
PAIRS_SHA256 = "5c5c22973d4b3e3bbbb6843961ca0a2cc2ee881d0020b3429f67408d57911dd5"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def audit_pair_labels(frame, records, pairs, assignment):
    """Audit outcomes in isolation; never pass this frame to a feature encoder."""
    lookup = frame.set_index("reaction_id")
    seen = set()
    for p in pairs:
        a, b = lookup.loc[p["tree_A"]], lookup.loc[p["tree_B"]]
        key = tuple(sorted((p["tree_A"], p["tree_B"])))
        if key in seen:
            raise AssertionError("Duplicate/reverse preference pair")
        seen.add(key)
        changed = [c for c in COMPONENTS if a[c] != b[c]]
        if changed != [p["changed_component"]] or a["target"] != b["target"]:
            raise AssertionError("One-component/target pair contract violated")
        if abs(a["Output"] - b["Output"]) < 20 or p["preference_label"] != int(a["Output"] > b["Output"]):
            raise AssertionError("Preference-gap or binary label contract violated")
        if p["split"] != assignment[a["target"]]:
            raise AssertionError("Frozen split assignment changed")
    if records != route_records(frame):
        raise AssertionError("Frozen experimental observations changed")
    forbidden = {"Output", "yield", "yield_gap", "yield_percent"}
    if any(forbidden.intersection(p) for p in pairs):
        raise AssertionError("Numerical outcomes in preference records")
    return {"pairs": len(pairs), "unique_unordered_pairs": len(seen),
            "minimum_preference_gap_pp": 20, "labels_verified_from_raw_only": True,
            "reverse_pair_augmentation": False, "records_unchanged": True}


def load_frozen_artifacts(directory, raw_file=RAW_FILE):
    directory = Path(directory)
    if checksum(directory / "z_tree.npy") != CACHE_SHA256 or checksum(directory / "pairs.jsonl") != PAIRS_SHA256:
        raise AssertionError("Cache or frozen preference file hash mismatch")
    baseline = json.loads((directory / "summary.json").read_text())
    cache = np.load(directory / "z_tree.npy", allow_pickle=False)
    if cache.shape != (15, 1200) or not np.isfinite(cache).all():
        raise AssertionError("Expected finite validated [15,1200] cache")
    records, pairs = read_jsonl(directory / "reactions.jsonl"), read_jsonl(directory / "pairs.jsonl")
    frame, audit = load_and_audit(raw_file)
    recomputed, pair_summary = construct_pairs(frame)
    if pairs != recomputed or pair_summary != baseline["pair_construction"] or audit != baseline["audit"]:
        raise AssertionError("Frozen source, pair labels, flags, orientation or splits changed")
    checks = audit_pair_labels(frame, records, pairs, baseline["pair_construction"]["target_split"])
    if multi_candidate_groups(pairs) != read_jsonl(directory / "multi_candidate_groups.jsonl"):
        raise AssertionError("Frozen multi-candidate groups changed")
    route_index = baseline["route_index"]
    aryl_order = baseline["aryl_order"]
    if set(route_index) != {r["reaction_id"] for r in records}:
        raise AssertionError("Cache mapping does not cover exactly the frozen observations")
    for r in records:
        if aryl_order[route_index[r["reaction_id"]]] != r["tree"]["reactants"][0]:
            raise AssertionError("Native cache mapping changed")
    validate_split(pairs)
    manifest = {name: checksum(directory / name) for name in
                ("z_tree.npy", "pairs.jsonl", "reactions.jsonl", "multi_candidate_groups.jsonl", "summary.json")}
    return frame, records, pairs, baseline, cache, checks, manifest


def state_digest(model):
    digest = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def verify_backbone(checkpoint, baseline, cache):
    """Verify released predictor/cache once; no encoder or old baseline training."""
    checkpoint_sha = checksum(checkpoint)
    predictor, _ = load_predictor(checkpoint)
    encoder = FrozenTreeEncoder(predictor)
    state_before = state_digest(predictor)
    graphs = [native_tree(a, baseline["audit"]["products_by_aryl"][a]) for a in baseline["aryl_order"]]
    observed, report = verify_encoder(encoder, graphs)
    if not np.array_equal(observed, cache):
        raise AssertionError("Released encoder does not exactly reproduce the frozen cache")
    if report["prediction_sha256"] != baseline["encoder"]["prediction_sha256"]:
        raise AssertionError("Original released-checkpoint forward outputs changed")
    # An actual pairwise backward pass through encoded representations must
    # leave the backbone gradient-free, in eval, and bitwise unchanged.
    encoded = encoder.encode_tree(Batch.from_data_list(graphs[:2]))
    condition = torch.eye(2)
    joint = torch.cat((encoded.z_tree, condition), dim=1)
    head = PairwiseRanker(joint.shape[1])
    loss = F.binary_cross_entropy_with_logits(head(joint)[0] - head(joint)[1], torch.tensor(1.0))
    loss.backward()
    assert head.score.weight.grad is not None and head.score.weight.grad.count_nonzero() > 0
    assert all(not p.requires_grad and p.grad is None for p in predictor.parameters())
    assert not predictor.training and state_digest(predictor) == state_before
    assert checksum(checkpoint) == checkpoint_sha
    report.update({"cache_reused": True, "cache_reproduced_bitwise": True,
                   "checkpoint_sha256": checkpoint_sha, "predictor_state_sha256": state_before,
                   "ranking_backward_checked": True, "backbone_gradients": 0,
                   "backbone_state_unchanged": True, "checkpoint_file_unchanged": True})
    return encoder, report


def macro_metrics(folds):
    macro = {s: {m: float(np.mean([f["splits"]["test"][s][m] for f in folds]))
                 for m in ("accuracy", "accuracy_ties_half", "roc_auc")} for s in SUBSETS}
    macro["multi_candidate"] = {m: float(np.mean([f["splits"]["test"]["multi_candidate"][m] for f in folds]))
                                for m in ("ndcg_at_3", "mrr")}
    return macro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=Path("results/pathway"))
    parser.add_argument("--raw-file", type=Path, default=RAW_FILE)
    parser.add_argument("--checkpoint", help="Defaults to the validated baseline's 3-RXN checkpoint")
    parser.add_argument("--output", type=Path, default=Path("results/pathway/condition_aware"))
    parser.add_argument("--report", type=Path, default=Path("docs/pathway/condition_aware_summary.json"))
    args = parser.parse_args()
    if args.output.resolve() == args.baseline_dir.resolve():
        raise ValueError("Condition output must not overwrite baseline artifacts")
    torch.set_num_threads(1)
    frame, records, pairs, baseline, cache, pair_checks, manifest = load_frozen_artifacts(args.baseline_dir, args.raw_file)
    checkpoint = args.checkpoint or baseline["encoder"]["checkpoint"]
    encoder, backbone_checks = verify_backbone(checkpoint, baseline, cache)
    print("Frozen source/pairs/splits/cache and released encoder verified", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    observations = {r["reaction_id"]: i for i, r in enumerate(records)}
    native_index = baseline["route_index"]
    per_observation_tree = cache[[native_index[r["reaction_id"]] for r in records]]
    results = {name: [] for name in ("tree_only", "condition_only", "tree_condition")}
    encoding_checks = []
    # Reproduce the small existing linear Tree-only probe on the reused cache;
    # never refit or rebuild the encoder, dataset, checkpoints or random controls.
    for fold in range(3):
        assignment = split_targets(frame, fold=fold)
        frozen_assignment = baseline["results"]["pretrained_tree"]["leave_one_scaffold_out"][fold]["target_split"]
        if assignment != frozen_assignment:
            raise AssertionError("Scaffold fold assignment changed")
        examples = [{**p, "split": assignment[p["target_context"]["target"]]} for p in pairs]
        validate_split(examples)
        train_ids = sorted({p[k] for p in examples if p["split"] == "train" for k in ("tree_A", "tree_B")})
        train_rows = [observations[r] for r in train_ids]
        conditions = [r["conditions"] for r in records]
        condition_encoder = ConditionEncoder.fit(conditions[i] for i in train_rows)
        condition_features = condition_encoder.transform(conditions)
        repeat = ConditionEncoder.fit(conditions[i] for i in reversed(train_rows)).transform(conditions)
        if not np.array_equal(repeat, condition_features):
            raise AssertionError("Nondeterministic condition encoding")
        features = representations(per_observation_tree, condition_features)
        tree_stats = training_statistics(cache, [native_index[r] for r in train_ids])
        condition_stats = training_statistics(condition_features, train_rows)
        normalization = {"condition_only": condition_stats,
                         "tree_condition": tuple(torch.cat((t, c)) for t, c in zip(tree_stats, condition_stats))}
        checks = {"fold": fold, "target_split": assignment, "condition_encoder": condition_encoder.metadata(),
                  "training_observation_ids": train_ids, "deterministic_exact": True,
                  "normalization": "tree block: unique training native trees; condition block: unique training observations; same blocks in joint model",
                  "unknown_categories": {}, "collisions": {}}
        for split in ("train", "validation", "test"):
            selected = [r["conditions"] for r in records if assignment[r["target"]] == split]
            checks["unknown_categories"][split] = condition_encoder.unknown_counts(selected)
        for name in results:
            f = features[name]
            checks["collisions"][name] = collision_counts(f, observations, examples)
            if name != "tree_only" and checks["collisions"][name]["condition_only_pair_collisions"] != 0:
                raise AssertionError("Condition-only comparisons still collide")
            if name == "tree_only":
                report, saved = fit_and_evaluate(cache, native_index, examples)
                # The unchanged protocol must replay every recorded metric and
                # the selected epoch exactly, before accepting the comparison.
                old = baseline["results"]["pretrained_tree"]["leave_one_scaffold_out"][fold]
                assert report["best_epoch"] == old["best_epoch"]
                for split in ("train", "validation", "test"):
                    for subset, values in old["splits"][split].items():
                        if report["splits"][split][subset] != values:
                            raise AssertionError("Tree-only baseline metrics did not replay exactly")
                report["baseline_reproduced_exactly"] = True
            else:
                report, saved = fit_and_evaluate(f, observations, examples,
                                                training_normalization=normalization[name])
            report.update({"fold": fold, "target_split": assignment, "feature_width": int(f.shape[1])})
            saved.update({"condition_encoder": None if name == "tree_only" else condition_encoder.metadata(),
                          "training_observation_ids": train_ids, "tree_cache_sha256": CACHE_SHA256,
                          "representation": name})
            torch.save(saved, args.output / f"{name}_fold{fold}_ranker.pt")
            results[name].append(report)
            print(f"{name} fold {fold}: accuracy={report['splits']['test']['overall']['accuracy']:.6f} "
                  f"AUC={report['splits']['test']['overall']['roc_auc']:.6f}", flush=True)
        np.save(args.output / f"z_condition_fold{fold}.npy", condition_features)
        encoding_checks.append(checks)
    if any(p.grad is not None or p.requires_grad for p in encoder.parameters()):
        raise AssertionError("Backbone received ranking gradients")
    if state_digest(encoder.predictor) != backbone_checks["predictor_state_sha256"]:
        raise AssertionError("Frozen checkpoint parameters changed during ranking")
    backbone_checks["all_nine_training_runs_backbone_gradients"] = 0
    del encoder
    gc.collect()
    results = {name: {"primary_split": folds[0], "leave_one_scaffold_out": folds,
                      "macro_scaffold_test": macro_metrics(folds)} for name, folds in results.items()}
    comparisons = {}
    for a, b in (("condition_only", "tree_only"), ("tree_condition", "tree_only"), ("tree_condition", "condition_only")):
        key = f"{a}_vs_{b}"
        comparisons[key] = {"macro_test_accuracy_delta": results[a]["macro_scaffold_test"]["overall"]["accuracy"] - results[b]["macro_scaffold_test"]["overall"]["accuracy"],
                            "macro_test_auc_delta": results[a]["macro_scaffold_test"]["overall"]["roc_auc"] - results[b]["macro_scaffold_test"]["overall"]["roc_auc"],
                            "fold_test_accuracy_deltas": [x["splits"]["test"]["overall"]["accuracy"] - y["splits"]["test"]["overall"]["accuracy"]
                                                          for x, y in zip(results[a]["leave_one_scaffold_out"], results[b]["leave_one_scaffold_out"])]}
    report = {"scope": "condition-aware single-step reaction/pathway preference benchmark",
              "baseline_provenance": {"commit": "e26a7378c0e4ab09e90dd29863158195b1f5a495", "files_sha256": manifest},
              "raw_sha256": baseline["audit"]["raw_sha256"], "pair_construction": baseline["pair_construction"],
              "pair_label_checks": pair_checks, "backbone_checks": backbone_checks,
              "encoding_checks": encoding_checks, "results": results, "comparisons": comparisons,
              "protocol": {"head": "bias-free linear Bradley-Terry score", "probability": "sigmoid(score(A)-score(B))",
                           "epochs": 200, "optimizer": "AdamW", "learning_rate": 0.01, "weight_decay": 0.001,
                           "seed": 0, "torch_threads": 1, "selection": "validation binary logistic loss",
                           "normalization_fit": "training only; block statistics identical in separate and joint heads",
                           "category_fit": "training only; validation/test may transform but never extend vocabulary",
                           "model_input_roles": ["frozen native tree", "Ligand", "Base", "Additive"],
                           "yield_or_gap_features": False, "trainable_encoder": False,
                           "ndcg_mrr": baseline["ndcg_mrr"]},
              "limitations": ["Single-step data; no multi-step pathway optimization claim",
                              "Only three independent scaffold groups, with shared observations across pairs",
                              "Linear concatenation adds no tree-condition interaction terms",
                              "NDCG/MRR retain the 71 existing halide groups, not a new condition-ranking protocol",
                              "Route-cliff candidates are operational flags, not validated mechanistic cliffs"],
              "remaining_aca_blocker": "Validate transferable tree-condition interactions and calibrated cliff/uncertainty definitions on broader independent chemistry before ACA; this simple additive probe cannot resolve context-dependent condition preferences."}
    for name, original_sha in manifest.items():
        if checksum(args.baseline_dir / name) != original_sha:
            raise AssertionError("Baseline artifact was modified")
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (args.output / "summary.json").write_text(payload, encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(payload, encoding="utf-8")
    print(json.dumps(comparisons, indent=2), flush=True)
    print(f"Completed single-step benchmark: {args.report}", flush=True)


if __name__ == "__main__":
    main()
