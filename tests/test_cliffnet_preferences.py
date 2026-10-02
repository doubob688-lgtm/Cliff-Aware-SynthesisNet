import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cliffnet.buchwald import COMPONENTS, RAW_SHA256, construct_pairs, download_raw, load_and_audit, product_for, route_records
from cliffnet.experiment import native_tree, validate_split
from cliffnet.ranking import PairwiseRanker, fit_and_evaluate, group_metrics, multi_candidate_groups


def fixture():
    rows = []
    for core in ["c1ccc(CC)cc1", "c1ccc(OC)cc1", "c1ccc(C(F)(F)F)cc1", "c1ccccn1", "c1cccnc1"]:
        for aryl, outcome in zip([f"{h}{core}" for h in ["Cl", "Br", "I"]], [5, 35, 65]):
            for ligand, offset in [("P(C)(C)C", 0), ("P(CC)(CC)CC", 25)]:
                rows.append({"Aryl halide": aryl, "Ligand": ligand, "Base": "N", "Additive": "CCO",
                             "Output": outcome + offset, "target": product_for(aryl),
                             "reaction_id": f"unit:{len(rows)}"})
    return pd.DataFrame(rows)


class TestPreferenceConstruction(unittest.TestCase):
    def test_existing_raw_file_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "original.xlsx"
            path.write_bytes(b"different-original-bytes")
            with self.assertRaisesRegex(ValueError, "left unchanged"):
                download_raw(path)
            self.assertEqual(path.read_bytes(), b"different-original-bytes")

    def test_pinned_benchmark_and_upstream_native_tree_features(self):
        frame, audit = load_and_audit()
        self.assertEqual(audit["raw_sha256"], RAW_SHA256)
        self.assertEqual(len(frame), 3955)
        self.assertEqual(audit["duplicates_canonical_conditions"], 0)
        self.assertEqual(len(audit["missing_factorial_combinations"]), 5)
        row = frame.iloc[0]
        graph = native_tree(row["Aryl halide"], row["target"])
        self.assertEqual(tuple(graph.x.shape), (4, 4219))
        self.assertEqual(tuple(graph.edge_index.shape), (2, 6))
        self.assertEqual(graph.x[2, 4096:4187].argmax(), 41)
        self.assertEqual(graph.x[2, 4096:4187].sum(), 1)
        for node in range(4):
            torch.testing.assert_close(graph.x[node, 2048:4096], graph.x[3, 2048:4096])
        # Another experimental condition record has exactly the same native tree.
        other = frame[(frame["Aryl halide"] == row["Aryl halide"]) & (frame["Ligand"] != row["Ligand"])].iloc[0]
        torch.testing.assert_close(graph.x, native_tree(other["Aryl halide"], other["target"]).x, rtol=0, atol=0)

    def test_one_change_same_context_and_gap_labels(self):
        frame = fixture()
        pairs, summary = construct_pairs(frame)
        self.assertEqual(len(pairs), 45)
        self.assertEqual(summary["threshold_pair_counts"], {"10": 45, "20": 45, "30": 30, "40": 10})
        lookup = frame.set_index("reaction_id")
        seen = set()
        for p in pairs:
            a, b = lookup.loc[p["tree_A"]], lookup.loc[p["tree_B"]]
            self.assertEqual(a["target"], b["target"])
            self.assertEqual(sum(a[c] != b[c] for c in COMPONENTS), 1)
            self.assertGreaterEqual(abs(a["Output"] - b["Output"]), 20)
            self.assertEqual(p["preference_label"], int(a["Output"] > b["Output"]))
            key = tuple(sorted((p["tree_A"], p["tree_B"])))
            self.assertNotIn(key, seen)
            seen.add(key)
            self.assertNotIn("Output", p)
        self.assertTrue(all("Output" not in r for r in route_records(frame)))

    def test_determinism_orientation_and_outcome_independent_split(self):
        frame = fixture()
        pairs, summary = construct_pairs(frame)
        self.assertEqual(pairs, construct_pairs(frame)[0])
        reversed_outcomes = frame.copy()
        reversed_outcomes["Output"] = 100 - frame["Output"]
        flipped, after = construct_pairs(reversed_outcomes)
        self.assertEqual(summary["target_split"], after["target_split"])
        for a, b in zip(pairs, flipped):
            self.assertEqual(a["tree_A"], b["tree_A"])
            self.assertEqual(a["tree_B"], b["tree_B"])
            self.assertEqual(a["reaction_similarity"], b["reaction_similarity"])
            self.assertEqual(a["preference_label"], 1 - b["preference_label"])

    def test_group_split_disjoint_underlying_rows_and_targets(self):
        pairs, _ = construct_pairs(fixture())
        validate_split(pairs)
        contaminated = [dict(p) for p in pairs]
        contaminated[0]["split"] = "test" if pairs[0]["split"] != "test" else "train"
        with self.assertRaisesRegex(AssertionError, "leakage"):
            validate_split(contaminated)

    def test_cliff_definition_and_threshold_monotonicity(self):
        pairs, _ = construct_pairs(fixture())
        _, wider = construct_pairs(fixture(), gap=30)
        self.assertLess(wider["pairs"], len(pairs))
        for p in pairs:
            if p["route_cliff"]:
                self.assertTrue(p["high_similarity"])
                self.assertGreaterEqual(p["reaction_similarity"], 0.9)
                self.assertGreaterEqual(p["tree_similarity"], 0.8)


class TestPairwiseRanking(unittest.TestCase):
    def test_complete_groups_and_tie_aware_metrics_use_ordinal_labels(self):
        pairs, _ = construct_pairs(fixture())
        groups = multi_candidate_groups(pairs)
        self.assertEqual(len(groups), 10)
        group = groups[0]
        self.assertEqual(sorted(group["ordinal_relevance"]), [0, 1, 2])
        scores = dict(zip(group["routes"], group["ordinal_relevance"]))
        perfect = group_metrics([group], scores)
        self.assertEqual(perfect["ndcg_at_3"], 1.0)
        self.assertEqual(perfect["mrr"], 1.0)
        tied = group_metrics([group], {r: 0 for r in group["routes"]})
        self.assertAlmostEqual(tied["mrr"], (1 + 0.5 + 1 / 3) / 3)
        self.assertLess(tied["ndcg_at_3"], 1)
        missing_edge = pairs[1:]
        self.assertLess(len(multi_candidate_groups(missing_edge)), len(groups))

    def test_antisymmetric_probabilities(self):
        model = PairwiseRanker(3)
        with torch.no_grad():
            model.score.weight.copy_(torch.tensor([[1.0, 2.0, -1.0]]))
        a, b = torch.tensor([[1.0, 2.0, 3.0]]), torch.zeros(1, 3)
        torch.testing.assert_close(torch.sigmoid(model.preference_logits(a, b)), 1 - torch.sigmoid(model.preference_logits(b, a)))

    def test_identical_embeddings_are_chance_and_cannot_learn_conditions(self):
        pairs, _ = construct_pairs(fixture())
        index = {f"unit:{i}": 0 for i in range(30)}
        report, saved = fit_and_evaluate(np.ones((1, 1200)), index, pairs, epochs=3)
        for split in report["splits"].values():
            self.assertEqual(split["overall"]["roc_auc"], 0.5)
            self.assertEqual(split["overall"]["accuracy_ties_half"], 0.5)
            self.assertEqual(split["overall"]["ties"], split["overall"]["n"])
        self.assertEqual(saved["ranker_state_dict"]["score.weight"].count_nonzero(), 0)

    def test_logistic_ranker_learns_binary_preferences(self):
        pairs = [{"tree_A": f"{s}:a", "tree_B": f"{s}:b", "preference_label": 1, "split": s,
                  "target_context": {"target": s, "fixed_components": {}},
                  "high_similarity": True, "route_cliff": True, "changed_component": "Aryl halide"}
                 for s in ["train", "validation", "test"]]
        index = {p[k]: 0 if k == "tree_A" else 1 for p in pairs for k in ["tree_A", "tree_B"]}
        report, _ = fit_and_evaluate(np.array([[1.0], [-1.0]]), index, pairs, epochs=30)
        self.assertEqual(report["splits"]["test"]["overall"]["accuracy"], 1.0)
        self.assertLess(report["history"][-1]["train_pairwise_logistic_loss"], report["history"][0]["train_pairwise_logistic_loss"])


if __name__ == "__main__":
    unittest.main()
