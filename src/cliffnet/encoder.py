"""Frozen, checkpoint-compatible access to the upstream pre-output tree state."""

from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.nn import MessagePassing, global_mean_pool


@dataclass
class TreeEncoding:
    node_hidden: torch.Tensor
    z_tree: torch.Tensor
    predictions: torch.Tensor


class FrozenTreeEncoder(nn.Module):
    """Observe GNNModel without replacing its layers or changing its forward.

    Load released weights with cliffnet.baseline.load_predictor BEFORE wrapping.
    Checkpoint keys remain those of ``encoder.predictor.state_dict()``; save a
    ranker separately. Encoding installs a temporary observation-only pre-hook.
    Calls on the same predictor must be serial (hooks are module-wide).
    """

    def __init__(self, predictor, hidden_width=1200, output_width=347):
        super().__init__()
        if not isinstance(predictor.layers[-1], MessagePassing):
            raise ValueError("Expected a final upstream graph convolution")
        self.predictor = predictor
        self.hidden_width = hidden_width
        self.output_width = output_width
        self.predictor.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        # A parent ranker's train() call must never enable pretrained dropout.
        super().train(False)
        return self

    def forward(self, data, **kwargs):
        return self.predictor(data, **kwargs)

    @torch.no_grad()
    def encode_tree(self, data):
        if data.num_nodes == 0 or not torch.isfinite(data.x).all():
            raise ValueError("Expected a nonempty graph with finite features")
        captured = []

        def observe(_module, args):
            captured.append(args[0].detach().clone())
            # Return None: no replacement of convolution inputs.

        handle = self.predictor.layers[-1].register_forward_pre_hook(observe)
        try:
            predictions = self.predictor(data)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("Expected exactly one final convolution call")
        hidden = captured[0]
        if hidden.shape != (data.num_nodes, self.hidden_width):
            raise ValueError(f"Unexpected pre-output hidden shape: {tuple(hidden.shape)}")
        if predictions.shape != (data.num_nodes, self.output_width):
            raise ValueError(f"Unexpected prediction shape: {tuple(predictions.shape)}")
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=hidden.device)
        if batch.shape != (data.num_nodes,) or batch.dtype != torch.long:
            raise ValueError("Expected one integer graph index per node")
        if not torch.equal(torch.unique(batch), torch.arange(batch.max() + 1, device=batch.device)):
            raise ValueError("Graph indices must be contiguous, nonnegative, and nonempty")
        pooled = global_mean_pool(hidden, batch)
        if not torch.isfinite(hidden).all() or not torch.isfinite(pooled).all() or not torch.isfinite(predictions).all():
            raise ValueError("Nonfinite encoder output")
        return TreeEncoding(hidden, pooled, predictions)
