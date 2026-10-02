"""Regression checks for Phase 0 tooling; no released checkpoint is needed."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from cliffnet.baseline import load_predictor, prepare_input
from synnet.encoding.fingerprints import fp_256, fp_2048
from synnet.models.common import load_gnn_from_ckpt
from synnet.models.mlp import GNN


class TestPhase0Baseline(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        torch.manual_seed(0)
        torch.set_num_threads(1)
        embeddings = self.directory / "validation_embeddings.npy"
        # A real molecular FP for the small test checkpoint's validation data.
        # This is not a substitute for the published model's Enamine catalog.
        np.save(embeddings, np.array([fp_256("CCO")], dtype=np.float32))
        self.original = GNN(c_in=4219, c_hidden=8, c_out=347, num_layers=2,
                            layer_name="Transformer", heads=2, concat=False,
                            dp_rate=0.0, X=str(embeddings))
        self.saved = {"state_dict": self.original.state_dict(),
                      "hyper_parameters": dict(self.original.hparams),
                      "pytorch-lightning_version": pl.__version__}
        self.checkpoint = self.directory / "small.ckpt"
        torch.save(self.saved, self.checkpoint)

    def test_loader_matches_upstream_lightning_loader_exactly(self):
        reference = load_gnn_from_ckpt(self.checkpoint).eval()
        restored, _ = load_predictor(self.checkpoint)
        data, _, _ = prepare_input("tests/data/ref/st_data.json.gz")
        self.assertEqual(set(reference.state_dict()), set(restored.state_dict()))
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
        with torch.inference_mode():
            torch.testing.assert_close(reference(data), restored(data), rtol=0, atol=0)

    def test_partial_tree_contains_only_target_information(self):
        data, skeleton, target = prepare_input("tests/data/ref/st_data.json.gz")
        self.assertEqual(tuple(data.x.shape), (4, 4219))
        self.assertEqual(tuple(data.edge_index.shape), (2, 6))
        self.assertEqual(skeleton.mask.sum(), 1)
        self.assertEqual(skeleton.mask[skeleton.tree_root], 1)
        expected_target = torch.tensor(fp_2048(target), dtype=torch.float32)
        for node in range(4):
            torch.testing.assert_close(data.x[node, 2048:4096], expected_target)
            self.assertEqual(data.x[node, 4096:4187].count_nonzero(), 0)
            if node != skeleton.tree_root:
                self.assertEqual(data.x[node, :2048].count_nonzero(), 0)
        edges = set(map(tuple, data.edge_index.T.tolist()))
        self.assertTrue(all((destination, source) in edges for source, destination in edges))

    def test_loading_rejects_missing_weights(self):
        self.saved["state_dict"].pop(next(iter(self.saved["state_dict"])))
        torch.save(self.saved, self.checkpoint)
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            load_predictor(self.checkpoint)

    def test_loading_rejects_unexpected_checkpoint_weights(self):
        self.saved["state_dict"]["extra.weight"] = torch.ones(1)
        torch.save(self.saved, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "model"):
            load_predictor(self.checkpoint)


if __name__ == "__main__":
    unittest.main()
