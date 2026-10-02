"""Replay the first frozen native-tree route preference experiment."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Batch, Data

from cliffnet.baseline import load_predictor
from cliffnet.buchwald import (AMINE, RAW_FILE, construct_pairs, load_and_audit,
                              route_records, split_targets, write_jsonl)
from cliffnet.encoder import FrozenTreeEncoder
from cliffnet.ranking import fit_and_evaluate, multi_candidate_groups
from synnet.utils.data_utils import Skeleton, SyntheticTree


def native_tree(aryl, target):
    """A complete 4-node upstream tree; no fabricated condition building blocks."""
    from synnet.models.gnn import PtrDataset
    templates = Path("data/assets/reaction-templates/hb.txt").read_text().splitlines()
    rxn = rdChemReactions.ReactionFromSmarts(templates[41])
    products = set()
    for result in rxn.RunReactants((Chem.MolFromSmiles(aryl), Chem.MolFromSmiles(AMINE))):
        Chem.SanitizeMol(result[0])
        products.add(Chem.MolToSmiles(result[0]))
    if products != {target}:
        raise ValueError(f"Upstream template 41 disagrees with benchmark product: {products}, {target}")
    tree = SyntheticTree()
    tree.update(0, 41, aryl, AMINE, target)
    tree.update(3, None, None, None, None)
    skeleton = Skeleton(tree, -1)
    skeleton.reset(mask=np.arange(4))
    _, features, _ = skeleton.get_state()
    pe = PtrDataset.positionalencoding1d(32, len(features)).numpy()
    return Data(x=torch.tensor(np.concatenate((features, pe), axis=1), dtype=torch.float32),
                edge_index=torch.tensor(skeleton.bidir_edges, dtype=torch.long))


def verify_encoder(encoder, graphs):
    """Actual-checkpoint bitwise output preservation and tolerant graph batching."""
    batch = Batch.from_data_list(graphs)
    with torch.inference_mode():
        original = encoder.predictor(batch)
        first = encoder.encode_tree(batch)
        repeated = encoder.encode_tree(batch)
        torch.testing.assert_close(original, first.predictions, rtol=0, atol=0)
        torch.testing.assert_close(first.z_tree, repeated.z_tree, rtol=0, atol=0)
        max_error = 0.0
        for i, graph in enumerate(graphs):
            single = encoder.encode_tree(graph)
            max_error = max(max_error, float((first.z_tree[i] - single.z_tree[0]).abs().max()))
            torch.testing.assert_close(first.z_tree[i], single.z_tree[0], rtol=1e-4, atol=2e-5)
    return first.z_tree.numpy().copy(), {
        "node_hidden_shape": list(first.node_hidden.shape), "z_tree_shape": list(first.z_tree.shape),
        "prediction_shape": list(first.predictions.shape), "predictions_bitwise_unchanged": True,
        "embeddings_repeat_exact": True, "finite": True, "single_vs_batch_max_abs_error": max_error,
        "prediction_sha256": hashlib.sha256(original.numpy().tobytes()).hexdigest()}


def validate_split(pairs):
    targets, rows, scaffolds, scaffold_cache = {}, {}, {}, {}
    for p in pairs:
        target = p["target_context"]["target"]
        if target in targets and targets[target] != p["split"]:
            raise AssertionError("Target-family leakage")
        targets[target] = p["split"]
        if target not in scaffold_cache:
            scaffold_cache[target] = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(target))
        scaffold = scaffold_cache[target]
        if scaffold in scaffolds and scaffolds[scaffold] != p["split"]:
            raise AssertionError("Product-scaffold leakage")
        scaffolds[scaffold] = p["split"]
        for key in ("tree_A", "tree_B"):
            if p[key] in rows and rows[p[key]] != p["split"]:
                raise AssertionError("Underlying observation leakage")
            rows[p[key]] = p["split"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--verify-checkpoint", action="append", default=[])
    parser.add_argument("--raw-file", type=Path, default=RAW_FILE)
    parser.add_argument("--output", type=Path, default=Path("results/pathway"))
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    args.output.mkdir(parents=True, exist_ok=True)
    frame, audit = load_and_audit(args.raw_file)
    pairs, pair_summary = construct_pairs(frame)
    validate_split(pairs)
    print(f"Audit: {len(frame)} observations; {len(pairs)} preference pairs", flush=True)
    write_jsonl(args.output / "reactions.jsonl", route_records(frame))
    write_jsonl(args.output / "pairs.jsonl", pairs)
    multi_groups = multi_candidate_groups(pairs)
    write_jsonl(args.output / "multi_candidate_groups.jsonl", multi_groups)
    aryls = sorted(frame["Aryl halide"].unique())
    graphs = [native_tree(aryl, audit["products_by_aryl"][aryl]) for aryl in aryls]
    route_index = {r["reaction_id"]: aryls.index(r["Aryl halide"]) for _, r in frame.iterrows()}
    print("Loading frozen pretrained encoder", flush=True)
    predictor, config = load_predictor(args.checkpoint)
    encoder = FrozenTreeEncoder(predictor)
    pretrained, verification = verify_encoder(encoder, graphs)
    np.save(args.output / "z_tree.npy", pretrained)
    # Random encoder control uses the same architecture/topology, seed 0.
    torch.manual_seed(0)
    for layer in predictor.layers:
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    random_embeddings, _ = verify_encoder(encoder, graphs)
    np.save(args.output / "random_z_tree.npy", random_embeddings)
    del encoder, predictor
    gc.collect()
    additional_verification = []
    for checkpoint in args.verify_checkpoint:
        print(f"Verifying additional BB/RXN checkpoint: {checkpoint}", flush=True)
        predictor, _ = load_predictor(checkpoint)
        _, checks = verify_encoder(FrozenTreeEncoder(predictor), graphs[:2])
        additional_verification.append({"checkpoint": checkpoint, **checks})
        del predictor
        gc.collect()
    # Morgan control observes exactly the native chemical features; no outcomes
    # or experimental-condition vectors are added to any model input.
    morgan = np.stack([g.x[:, :4096].mean(0).numpy() for g in graphs])
    baselines = {"pretrained_tree": pretrained, "random_tree": random_embeddings, "morgan_native_tree": morgan}
    results = {}
    for name, features in baselines.items():
        folds = []
        for fold in range(3):
            assignment = split_targets(frame, fold=fold)
            examples = [{**p, "split": assignment[p["target_context"]["target"]]} for p in pairs]
            validate_split(examples)
            report, saved = fit_and_evaluate(features, route_index, examples, epochs=args.epochs)
            report["fold"] = fold
            report["target_split"] = assignment
            folds.append(report)
            if fold == 0:
                torch.save(saved, args.output / f"{name}_ranker.pt")
            print(f"{name} fold {fold}: test accuracy {report['splits']['test']['overall']['accuracy']:.4f}; halide {report['splits']['test']['halide_alternative']['accuracy']:.4f}", flush=True)
        results[name] = {"primary_split": folds[0], "leave_one_scaffold_out": folds,
                         "mean_test_accuracy": float(np.mean([f["splits"]["test"]["overall"]["accuracy"] for f in folds])),
                         "mean_test_halide_accuracy": float(np.mean([f["splits"]["test"]["halide_alternative"]["accuracy"] for f in folds])),
                         "macro_scaffold_test": {subset: {metric: float(np.mean([f["splits"]["test"][subset][metric] for f in folds]))
                                                          for metric in ("accuracy", "accuracy_ties_half", "roc_auc")}
                                                 for subset in ("overall", "high_similarity", "route_cliff", "halide_alternative", "condition_only")}}
    summary = {"audit": audit, "pair_construction": pair_summary,
               "encoder": {"checkpoint": args.checkpoint, "config": config, **verification},
               "additional_checkpoint_verification": additional_verification,
               "native_unique_trees": len(aryls), "experimental_route_records": len(frame),
               "per_route_z_tree_shape": [len(frame), 1200], "route_index": route_index,
               "aryl_order": aryls, "results": results,
               "ndcg_mrr": {"computed": True, "valid_three_candidate_groups": len(multi_groups),
                            "definition": "Same target/ligand/base/additive, all three halide comparisons survive >=20pp filtering; ordinal relevance from binary preference wins"},
               "limitations": ["Single-step HTE routes, not multi-step pathway comparisons",
                               "Native tree features omit ligand/base/additive; condition-only pairs have identical embeddings",
                               "Only five targets and three product scaffolds; train/validation/test use one scaffold each",
                               "Large pair counts share observations; pairs are not independent samples",
                               "No experimental uncertainties/replicates are supplied in FullCV_01",
                               "Similarity and preference thresholds are operational candidate definitions, not validated cliff labels"]}
    # Compact tracked report excludes redundant row-to-tree index mapping.
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Completed frozen pairwise ranking experiment: {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
