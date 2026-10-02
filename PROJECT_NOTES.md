# Cliff-Aware SynthesisNet Generator — Phase 0

2026-10-03 (Asia/Shanghai). Upstream: `shiningsunnyday/SynthesisNet`,
commit `2fefd24dd1b746165fd0b073f340f8ecbb2e15f4`.

**Pretrained BB/RXN GNN forward inference works. Complete synthetic-tree
inference is blocked; Phase 1 has not started.** All original upstream code is
unchanged. New code is limited to `src/cliffnet/baseline.py` and its tests.

## Architecture map

`target SMILES → 2048-bit Morgan FP → recognizer/supplied skeleton → root-only
partial tree → bidirectional graph + features + PE → separate RXN/BB GNNs →
frontier selection + constrained predictions → filled skeleton → RDKit reaction
execution → product tree`.

| Important file | Functions / role |
| --- | --- |
| `src/synnet/models/common.py` | `load_gnn_from_ckpt`, `load_mlp_from_ckpt`, `find_best_model_ckpt` |
| `src/synnet/models/mlp.py` | `GNNModel.forward`: actual TransformerConv architecture; `GNN`: Lightning wrapper; `MLP`: skeleton recognizer |
| `src/synnet/utils/data_utils.py` | `SyntheticTree`, `Skeleton`, `Skeleton.get_state`, `Skeleton.reconstruct`, `SkeletonSet`, `Reaction.run_reaction` |
| `src/synnet/utils/reconstruct_utils.py` | `set_models`, `load_data`, `predict_skeleton`, `decode`, `wrapper_decoder`, `pick_node`, `fill_in`, `reconstruct` |
| `scripts/reconstruct-targets.py`, `scripts/mcmc.py`, `sandbox/optimize.py` | Batch reconstruction, analog search, GA/GP generation entry points |

Trees contain alternating reaction/chemical nodes. Skeletons are postorder-indexed
NetworkX graphs with a filled-node mask and ordered reactants. The released GNNs
use five TransformerConv layers, hidden width 1200 and 347 output values per node:
256 BB fingerprint values and 91 reaction scores. Input width is 4219:
local FP 2048 + target FP 2048 + RXN indicators 91 + sinusoidal PE 32.

Full repository inventory, execution details and future seams:
[ARCHITECTURE.md](docs/phase0/ARCHITECTURE.md).

## Checkpoints, assets and environment

Verified published 3-RXN and 3-NN checkpoints are at
`/home/doubo/projects/SynthesisNet-master/ckpts/{3-RXN,3-NN}/` in WSL. Their exact
filenames appear below. The 3-REC recognizer is present and hash-verified but was
not exercised. [assets.json](docs/phase0/assets.json) records published download
URLs, file IDs, byte sizes and MD5s; no checkpoint binaries are committed.

The released training skeleton pickle was downloaded and verified at
`/home/doubo/SynthesisNet/results/viz/skeletons-train.pkl` to satisfy an upstream
import-time requirement. The HB template file is tracked. Complete inference
still needs the canonical matched Enamine CSV, aligned 256-bit BB embedding
matrix and reaction collection with available reactants. Recognizer evaluation
also needs matching `config.json` and canonical skeleton class ordering. Program
hash caches, 2048-bit BB embeddings and oracle/docking assets are conditional.

Clean runtime: WSL Ubuntu 22.04, Python 3.9.18, torch 2.1.0+cpu, Lightning 2.1.1,
PyG 2.4.0, RDKit 2023.9.5. A fresh isolated environment was installed from
[requirements.lock](docs/phase0/requirements.lock).

## Successful command and expected output

From the repository root inside WSL:

```bash
PYTHONPATH=src /home/doubo/.cache/cliffnet/phase0-clean/bin/python -m cliffnet.baseline \
  --checkpoint /home/doubo/projects/SynthesisNet-master/ckpts/3-RXN/ckpts.epoch=27-val_accuracy_loss=0.36.ckpt \
  --checkpoint /home/doubo/projects/SynthesisNet-master/ckpts/3-NN/ckpts.epoch=7-val_nn_accuracy_loss=0.67.ckpt
```

Input: first upstream `tests/data/ref/st_data.json.gz` tree; all labels cleared
except target root `CCOc1ccc(C2NCCc3c2cc(OC)c(OC)c3OC)o1`. The graph has four nodes,
six directed edges and root-only mask `[0,0,0,1]`. Output: finite `[4,347]` raw
predictions per pretrained network, identical on repeated eval calls; report at
`results/phase0/forward.json`. This does not select BBs or execute reactions.

The harness uses the unchanged state builder/positional encoding and strictly
loads every predictor weight into unchanged `GNNModel`. It bypasses the Lightning
training wrapper's missing validation-embedding file. Four new regression tests
and three discoverable upstream tests pass. Commands and provenance:
[RUNBOOK.md](docs/phase0/RUNBOOK.md), [forward-summary.json](docs/phase0/forward-summary.json).

## Safest future insertion points and blockers

`encode_tree()` belongs semantically immediately before the final graph
convolution in `GNNModel.forward`: observe `[N,1200]` hidden features, then pool
by graph into `z_tree`. Start with an adapter/hook under `src/cliffnet/`; preserve
state-dict keys and forward outputs. BB and RXN encoders are separate.
Candidate actions belong in `wrapper_decoder`/`fill_in` after feasibility/catalog
filtering and before ranking. At most two upstream files need small future seams:
`models/mlp.py` and `utils/reconstruct_utils.py`. Nothing is implemented yet.

Blockers: missing canonical chemical assets; native checkpoint loader requires
the embedding matrix; native decoder imports missing `tdc`. PyTDC installation
was dry-run only because its dependency metadata requests a second RDKit
distribution. Native Windows also lacks `fcntl`; inference runs in WSL.
All observed issues and unexecuted static findings are recorded in
[ISSUES.md](docs/phase0/ISSUES.md).

Do not introduce experimental reaction data until one native fixed-skeleton
decode verifies all filled nodes, reaction execution and a valid root product,
followed by recognizer-to-tree inference. An upstream helper can report the best
intermediate/BB instead of the root, so its Tanimoto score alone is insufficient.
