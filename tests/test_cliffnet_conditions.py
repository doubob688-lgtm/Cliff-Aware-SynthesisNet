import copy
import json
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import Data

from cliffnet.buchwald import construct_pairs, route_records, split_targets
from cliffnet.conditions import ConditionEncoder, collision_counts, representations, training_statistics
from cliffnet.condition_experiment import audit_pair_labels
from cliffnet.encoder import FrozenTreeEncoder
from cliffnet.experiment import validate_split
from cliffnet.ranking import PairwiseRanker, fit_and_evaluate
from synnet.models.mlp import GNNModel
from tests.test_cliffnet_preferences import fixture


class TestConditionEncoding(unittest.TestCase):
    def setUp(self):
        self.rows = [{"Ligand": l, "Base": b, "Additive": a}
                     for l, b, a in (("L2", "B1", "A2"), ("L1", "B1", "A1"), ("L2", "B2", "A1"))]
        self.encoder = ConditionEncoder.fit(self.rows)

    def test_explicit_one_hot_blocks(self):
        z = self.encoder.transform(self.rows)
        self.assertEqual(z.shape, (3, 9))
        np.testing.assert_array_equal(z.sum(1), [3, 3, 3])
        np.testing.assert_array_equal(z[0], [0, 1, 0, 1, 0, 0, 0, 1, 0])
        for role, offset in self.encoder.offsets.items():
            np.testing.assert_array_equal(z[:, offset:offset + 3].sum(1), np.ones(3))

    def test_deterministic_order_serialization_and_repeat(self):
        a = self.encoder.transform(self.rows)
        b = ConditionEncoder.fit(reversed(self.rows)).transform(self.rows)
        restored = ConditionEncoder(json.loads(json.dumps(self.encoder.metadata()))["vocabularies"])
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, restored.transform(self.rows))
        np.testing.assert_array_equal(a, self.encoder.transform(self.rows))
        self.assertEqual(self.encoder.transform([]).shape, (0, self.encoder.width))

    def test_held_out_categories_do_not_fit_vocabulary_or_normalization(self):
        held_out = {"Ligand": "NEW_L", "Base": "NEW_B", "Additive": "NEW_A"}
        before = copy.deepcopy(self.encoder.metadata())
        z = self.encoder.transform(self.rows + [held_out])
        np.testing.assert_array_equal(z[-1], [0, 0, 1, 0, 0, 1, 0, 0, 1])
        self.assertEqual(before, self.encoder.metadata())
        self.assertEqual(self.encoder.unknown_counts([held_out]), {"Ligand": 1, "Base": 1, "Additive": 1})
        first = training_statistics(z, [0, 1, 2])
        mutated = z.copy()
        mutated[-1] = 1e6
        for a, b in zip(first, training_statistics(mutated, [0, 1, 2])):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(first[0][2], 0)  # Unknown ligand slot absent from training.

    def test_yields_gaps_and_flags_are_ignored(self):
        clean = self.encoder.transform(self.rows)
        contaminated = [{**r, "Output": i * 40, "yield_gap": 99, "high_similarity": True}
                        for i, r in enumerate(self.rows)]
        np.testing.assert_array_equal(clean, self.encoder.transform(contaminated))

    def test_representation_widths_and_invalid_shapes(self):
        z = self.encoder.transform(self.rows)
        combined = representations(np.zeros((3, 1200), np.float32), z)
        self.assertEqual(combined["tree_only"].shape, (3, 1200))
        self.assertEqual(combined["condition_only"].shape, (3, 9))
        self.assertEqual(combined["tree_condition"].shape, (3, 1209))
        with self.assertRaises(ValueError):
            representations(np.zeros((2, 1200)), z)
        with self.assertRaises(ValueError):
            representations(np.full((3, 1200), np.nan), z)

    def test_exact_collision_counts_include_observation_and_component_pairs(self):
        pairs = [{"tree_A": "a", "tree_B": "b", "changed_component": "Ligand"},
                 {"tree_A": "a", "tree_B": "c", "changed_component": "Aryl halide"}]
        index = {"a": 0, "b": 1, "c": 2}
        tree = np.array([[0, 1], [0, 1], [1, 0]], dtype=np.float32)
        condition = self.encoder.transform([self.rows[0], self.rows[1], self.rows[0]])
        stats = collision_counts(tree, index, pairs)
        self.assertEqual(stats["unique_representations"], 2)
        self.assertEqual(stats["unordered_observation_collisions"], 1)
        self.assertEqual(stats["duplicate_observations_beyond_first"], 1)
        self.assertEqual(stats["condition_only_pair_collisions"], 1)
        condition_stats = collision_counts(condition, index, pairs)
        self.assertEqual(condition_stats["by_changed_component"]["Aryl halide"]["collisions"], 1)
        self.assertEqual(condition_stats["condition_only_pair_collisions"], 0)
        combined = representations(tree, condition)["tree_condition"]
        self.assertEqual(collision_counts(combined, index, pairs)["unordered_observation_collisions"], 0)


class TestConditionRankingIntegrity(unittest.TestCase):
    def test_actual_backward_leaves_frozen_backbone_and_forward_unchanged(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        predictor = GNNModel(12, 1200, 347, num_layers=2, layer_name="Transformer", dp_rate=0.5).eval()
        graph = Data(x=torch.randn(4, 12), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]))
        with torch.no_grad():
            original = predictor(graph)
        state = {k: v.clone() for k, v in predictor.state_dict().items()}
        encoder = FrozenTreeEncoder(predictor)
        encoder.train(True)
        encoded = encoder.encode_tree(graph)
        condition = torch.tensor([[1., 0.], [0., 1.]])
        joint = torch.cat((encoded.z_tree.expand(2, -1), condition), dim=1)
        ranker = PairwiseRanker(1202)
        optimizer = torch.optim.AdamW(ranker.parameters(), lr=.01)
        loss = F.binary_cross_entropy_with_logits(ranker.preference_logits(joint[:1], joint[1:]), torch.ones(1))
        loss.backward()
        optimizer.step()
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in predictor.parameters()))
        self.assertTrue(ranker.score.weight.grad.count_nonzero() > 0)
        self.assertFalse(predictor.training)
        torch.testing.assert_close(original, encoder(graph), rtol=0, atol=0)
        for key, value in state.items():
            torch.testing.assert_close(value, predictor.state_dict()[key], rtol=0, atol=0)

    def test_scaffold_splits_and_pair_labels_cannot_change(self):
        frame = fixture()
        pairs, summary = construct_pairs(frame)
        records = route_records(frame)
        audit_pair_labels(frame, records, pairs, summary["target_split"])
        for fold in range(3):
            assignment = split_targets(frame, fold=fold)
            validate_split([{**p, "split": assignment[p["target_context"]["target"]]} for p in pairs])
        for corruption in ("label", "reverse", "gap", "split"):
            frame = fixture()
            changed = copy.deepcopy(pairs)
            if corruption == "label":
                changed[0]["preference_label"] = 1 - changed[0]["preference_label"]
            elif corruption == "reverse":
                changed.append({**changed[0], "tree_A": changed[0]["tree_B"], "tree_B": changed[0]["tree_A"]})
            elif corruption == "gap":
                frame = frame.copy()
                ids = frame.reaction_id.isin([changed[0]["tree_A"], changed[0]["tree_B"]])
                frame.loc[ids, "Output"] = 50
            else:
                changed[0]["split"] = "invalid"
            with self.assertRaises(AssertionError):
                audit_pair_labels(frame, records, changed, summary["target_split"])

    def test_original_fold_assignments_match_tracked_baseline(self):
        from cliffnet.buchwald import load_and_audit
        frame, _ = load_and_audit()
        baseline = json.loads(Path("docs/pathway/summary.json").read_text())
        for fold, report in enumerate(baseline["results"]["pretrained_tree"]["leave_one_scaffold_out"]):
            self.assertEqual(split_targets(frame, fold=fold), report["target_split"])

    def test_joint_normalization_blocks_and_condition_learning(self):
        frame = fixture()
        pairs, _ = construct_pairs(frame)
        records = route_records(frame)
        index = {r["reaction_id"]: i for i, r in enumerate(records)}
        train = sorted({index[p[k]] for p in pairs if p["split"] == "train" for k in ("tree_A", "tree_B")})
        rows = [r["conditions"] for r in records]
        encoder = ConditionEncoder.fit(rows[i] for i in train)
        condition = encoder.transform(rows)
        tree = np.zeros((len(records), 2), dtype=np.float32)
        stats = training_statistics(condition, train)
        joint_stats = tuple(torch.cat((a, b)) for a, b in zip(training_statistics(tree, train), stats))
        c, c_saved = fit_and_evaluate(condition, index, pairs, epochs=30, training_normalization=stats)
        j, j_saved = fit_and_evaluate(representations(tree, condition)["tree_condition"], index, pairs,
                                     epochs=30, training_normalization=joint_stats)
        self.assertEqual(c["splits"]["test"]["condition_only"]["accuracy"], 1)
        self.assertEqual(j["splits"]["test"]["condition_only"]["accuracy"], 1)
        torch.testing.assert_close(c_saved["training_mean"], j_saved["training_mean"][2:], rtol=0, atol=0)
        torch.testing.assert_close(c_saved["training_scale"], j_saved["training_scale"][2:], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
