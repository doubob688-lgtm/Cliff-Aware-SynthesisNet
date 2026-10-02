import unittest

import torch
from torch_geometric.data import Batch, Data

from cliffnet.encoder import FrozenTreeEncoder
from synnet.models.mlp import GNNModel


class TestFrozenTreeEncoder(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        self.model = GNNModel(12, 1200, 347, num_layers=2,
                              layer_name="Transformer", dp_rate=0.5).eval()
        self.encoder = FrozenTreeEncoder(self.model)
        self.a = Data(x=torch.randn(3, 12), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]))
        self.b = Data(x=torch.randn(2, 12), edge_index=torch.tensor([[0, 1], [1, 0]]))

    def test_deterministic_finite_and_frozen(self):
        self.encoder.train(True)
        a, b = self.encoder.encode_tree(self.a), self.encoder.encode_tree(self.a)
        self.assertFalse(self.model.training)
        self.assertTrue(all(not p.requires_grad for p in self.model.parameters()))
        self.assertEqual(tuple(a.node_hidden.shape), (3, 1200))
        self.assertEqual(tuple(a.z_tree.shape), (1, 1200))
        self.assertTrue(torch.isfinite(a.z_tree).all())
        torch.testing.assert_close(a.z_tree, b.z_tree, rtol=0, atol=0)

    def test_batching_and_pooling(self):
        batch = Batch.from_data_list([self.a, self.b])
        encoded = self.encoder.encode_tree(batch)
        self.assertEqual(tuple(encoded.z_tree.shape), (2, 1200))
        for i, graph in enumerate([self.a, self.b]):
            torch.testing.assert_close(encoded.z_tree[i], self.encoder.encode_tree(graph).z_tree[0], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(encoded.z_tree[i], encoded.node_hidden[batch.batch == i].mean(0))

    def test_predictions_and_checkpoint_keys_unchanged(self):
        batch = Batch.from_data_list([self.a, self.b])
        state = {k: v.clone() for k, v in self.model.state_dict().items()}
        with torch.no_grad():
            original = self.model(batch)
        encoded = self.encoder.encode_tree(batch)
        torch.testing.assert_close(original, encoded.predictions, rtol=0, atol=0)
        torch.testing.assert_close(original, self.encoder(batch), rtol=0, atol=0)
        self.assertEqual(set(state), set(self.model.state_dict()))
        for k, v in state.items():
            torch.testing.assert_close(v, self.model.state_dict()[k], rtol=0, atol=0)
        self.assertFalse(self.model.layers[-1]._forward_pre_hooks)

    def test_rejects_nonfinite_inputs(self):
        self.a.x[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.encoder.encode_tree(self.a)


if __name__ == "__main__":
    unittest.main()
