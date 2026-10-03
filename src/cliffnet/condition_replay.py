"""Independently replay persisted condition benchmark heads without training."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from cliffnet.buchwald import checksum
from cliffnet.condition_experiment import read_jsonl
from cliffnet.conditions import ConditionEncoder, training_statistics
from cliffnet.ranking import group_metrics, multi_candidate_groups


def replay(baseline_dir, output_dir, summary):
    torch.set_num_threads(1)
    baseline_dir, output_dir = Path(baseline_dir), Path(output_dir)
    for name, expected in summary["baseline_provenance"]["files_sha256"].items():
        if checksum(baseline_dir / name) != expected:
            raise AssertionError("Baseline artifact hash changed")
    original = json.loads((baseline_dir / "summary.json").read_text())
    records = read_jsonl(baseline_dir / "reactions.jsonl")
    pairs = read_jsonl(baseline_dir / "pairs.jsonl")
    tree = np.load(baseline_dir / "z_tree.npy", allow_pickle=False)
    tree_index = original["route_index"]
    observation_index = {r["reaction_id"]: i for i, r in enumerate(records)}
    condition_rows = [r["conditions"] for r in records]
    checks = []
    for name, result in summary["results"].items():
        for fold, report in enumerate(result["leave_one_scaffold_out"]):
            saved = torch.load(output_dir / f"{name}_fold{fold}_ranker.pt", map_location="cpu", weights_only=False)
            assignment = report["target_split"]
            examples = [{**p, "split": assignment[p["target_context"]["target"]]} for p in pairs]
            training_ids = sorted({p[k] for p in examples if p["split"] == "train" for k in ("tree_A", "tree_B")})
            assert saved["training_observation_ids"] == training_ids
            tree_stats = training_statistics(tree, [tree_index[r] for r in training_ids])
            if name == "tree_only":
                features, index, expected_stats = tree, tree_index, tree_stats
            else:
                from_saved = ConditionEncoder(saved["condition_encoder"]["vocabularies"])
                from_train = ConditionEncoder.fit(records[observation_index[r]]["conditions"] for r in training_ids)
                assert from_saved.metadata() == from_train.metadata()
                condition = from_saved.transform(condition_rows)
                np.testing.assert_array_equal(condition, np.load(output_dir / f"z_condition_fold{fold}.npy", allow_pickle=False))
                c_stats = training_statistics(condition, [observation_index[r] for r in training_ids])
                index = observation_index
                if name == "condition_only":
                    features, expected_stats = condition, c_stats
                else:
                    features = np.concatenate((tree[[tree_index[r["reaction_id"]] for r in records]], condition), axis=1)
                    expected_stats = tuple(torch.cat((t, c)) for t, c in zip(tree_stats, c_stats))
            for expected, actual in zip(expected_stats, (saved["training_mean"], saved["training_scale"])):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            # Direct saved linear algebra: no training/evaluation function call.
            normalized = (torch.from_numpy(features) - saved["training_mean"]) / saved["training_scale"]
            scores = torch.nn.functional.linear(normalized, saved["ranker_state_dict"]["score.weight"]).squeeze(-1)
            for split in ("train", "validation", "test"):
                split_pairs = [p for p in examples if p["split"] == split]
                for subset, select in {
                    "overall": lambda p: True,
                    "high_similarity": lambda p: p["high_similarity"],
                    "route_cliff": lambda p: p["route_cliff"],
                    "halide_alternative": lambda p: p["changed_component"] == "Aryl halide",
                    "condition_only": lambda p: p["changed_component"] != "Aryl halide",
                    "ligand": lambda p: p["changed_component"] == "Ligand",
                    "base": lambda p: p["changed_component"] == "Base",
                    "additive": lambda p: p["changed_component"] == "Additive",
                }.items():
                    chosen = [p for p in split_pairs if select(p)]
                    labels = np.array([p["preference_label"] for p in chosen])
                    probabilities = torch.sigmoid(scores[[index[p["tree_A"]] for p in chosen]] -
                                                  scores[[index[p["tree_B"]] for p in chosen]]).numpy()
                    tied = probabilities == .5
                    actual = {"n": len(chosen), "accuracy": float(((probabilities >= .5) == labels).mean()),
                              "accuracy_ties_half": float(np.where(tied, .5, (probabilities > .5) == labels).mean()),
                              "roc_auc": float(roc_auc_score(labels, probabilities)),
                              "ties": int(tied.sum()), "positive_labels": int(labels.sum())}
                    expected = report["splits"][split][subset]
                    for metric, value in actual.items():
                        np.testing.assert_allclose(value, expected[metric], rtol=0, atol=1e-12,
                                                   err_msg=f"{name}/{fold}/{split}/{subset}/{metric}")
                multi = group_metrics(multi_candidate_groups(split_pairs), {r: float(scores[i]) for r, i in index.items()})
                assert multi == report["splits"][split]["multi_candidate"]
            checks.append({"representation": name, "fold": fold, "all_split_metrics_reproduced": True,
                           "metric_absolute_tolerance": 1e-12,
                           "training_only_vocab_and_normalization_reproduced_exactly": True})
    return {"training_performed": False, "verified_heads": len(checks), "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=Path("results/pathway"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/pathway/condition_aware"))
    parser.add_argument("--summary", type=Path, default=Path("docs/pathway/condition_aware_summary.json"))
    args = parser.parse_args()
    checks = replay(args.baseline_dir, args.output_dir, json.loads(args.summary.read_text()))
    (args.output_dir / "replay_checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
