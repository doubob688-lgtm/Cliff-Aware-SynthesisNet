"""Run a pretrained SynthesisNet GNN forward pass on one upstream tree fixture.

This is a numerical smoke test, not complete synthetic-tree generation. It loads
the unchanged GNNModel and its exact weights directly, avoiding the Lightning
training wrapper's mandatory, absent Enamine validation-embedding file.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from synnet.models.mlp import GNNModel
from synnet.utils.data_utils import Skeleton, SyntheticTreeSet


def load_predictor(checkpoint):
    """Restore the upstream inference module, strictly matching all its weights."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hparams = saved["hyper_parameters"]
    keys = ("c_in", "c_hidden", "c_out", "num_layers", "layer_name", "dp_rate",
            "heads", "concat")
    config = {key: hparams[key] for key in keys if key in hparams}
    if config.get("layer_name") != "Transformer":
        raise ValueError("Phase 0 smoke test supports the released Transformer checkpoints only")
    state = saved["state_dict"]
    if not state or any(not key.startswith("model.") for key in state):
        raise ValueError("Expected only model.* weights in the upstream GNN checkpoint")
    model = GNNModel(**config)
    model.load_state_dict({key.removeprefix("model."): value
                           for key, value in state.items()}, strict=True)
    return model.eval(), config


def prepare_input(tree_file, target=None):
    """Use the upstream state builder and positional encoding without changes."""
    # Import here: upstream gnn.py loads $HOME/SynthesisNet/results/viz/
    # skeletons-train.pkl even when only its positional encoding is needed.
    from synnet.models.gnn import PtrDataset

    tree = SyntheticTreeSet().load(str(tree_file))[0]
    skeleton = Skeleton(tree, -1)
    skeleton.clear_tree()
    target = target or tree.root.smiles
    skeleton.modify_tree(skeleton.tree_root, smiles=target)
    _, features, _ = skeleton.get_state(rxn_target_down_bb=True, rxn_target_down=True)
    for node in range(len(features)):
        if node != skeleton.tree_root and not skeleton.rxns[node] and not skeleton.leaves[node]:
            features[node, :2048] = 0
    pe = PtrDataset.positionalencoding1d(32, len(features)).numpy()
    data = Data(x=torch.tensor(np.concatenate((features, pe), axis=-1), dtype=torch.float32),
                edge_index=torch.tensor(skeleton.bidir_edges, dtype=torch.int64))
    return data, skeleton, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="Trusted released GNN checkpoint; repeat for BB and RXN")
    parser.add_argument("--tree-file", default="tests/data/ref/st_data.json.gz")
    parser.add_argument("--target", help="Optional target SMILES; defaults to first fixture root")
    parser.add_argument("--output", default="results/phase0/forward.json")
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(1)
    data, skeleton, target = prepare_input(args.tree_file, args.target)
    report = {"scope": "pretrained_gnn_forward_only", "complete_tree_inference": False,
              "target": target, "tree_file": args.tree_file,
              "input_shape": list(data.x.shape), "edge_index": data.edge_index.tolist(),
              "root": skeleton.tree_root, "filled_mask": skeleton.mask.tolist(),
              "torch": torch.__version__, "models": []}
    for checkpoint in args.checkpoint:
        print(f"Loading {checkpoint}", flush=True)
        model, config = load_predictor(checkpoint)
        if data.x.shape[1] != config["c_in"]:
            raise ValueError("Fixture feature width does not match checkpoint")
        with torch.inference_mode():
            prediction = model(data)
            repeated = model(data)
        if prediction.shape != (data.num_nodes, 347) or not torch.isfinite(prediction).all():
            raise AssertionError("Expected finite [num_nodes, 256 + 91] predictions")
        torch.testing.assert_close(prediction, repeated, rtol=0, atol=0)
        values = prediction.numpy()
        report["models"].append({"checkpoint": str(checkpoint), "config": config,
                                 "output_shape": list(values.shape), "finite": True,
                                 "repeat_exact": True,
                                 "prediction_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                                 "unmasked_rxn_argmax": values[:, 256:].argmax(axis=-1).tolist(),
                                 "prediction": values.tolist()})
        del model, prediction, repeated, values
        gc.collect()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(report['models'])} pretrained GNN forward passes; report: {output}")
    print("Complete-tree baseline remains unverified; Phase 1 is blocked.")


if __name__ == "__main__":
    main()
