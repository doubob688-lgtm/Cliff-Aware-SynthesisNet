# Cliff-Aware SynthesisNet Generator — Project notes

2026-10-03 (Asia/Shanghai). Upstream: `shiningsunnyday/SynthesisNet`,
commit `2fefd24dd1b746165fd0b073f340f8ecbb2e15f4`.

**Pretrained BB/RXN forward inference and the frozen pathway-ranking experiment
work. Complete synthetic-tree generation remains blocked.** The user's subsequent
pathway-first instruction authorizes the HTE experiment independently of the
earlier generation gate. All upstream code/checkpoint files remain unchanged;
new model/data/ranking code lives under `src/cliffnet/`. No TDC, numerical yield
regression, ACA loss or guided generation is implemented.

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
| `src/cliffnet/encoder.py` | `FrozenTreeEncoder.encode_tree`: final-convolution pre-hook, frozen eval, mean pooling and unchanged predictions |
| `src/cliffnet/buchwald.py` | Pinned XLSX loader/audit, scaffold grouping, one-component preference pairs and annotated native routes |
| `src/cliffnet/ranking.py`, `src/cliffnet/experiment.py` | Frozen linear Bradley-Terry ranker, controls, scaffold folds and pair/group evaluation |

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

Runtime: WSL Ubuntu 22.04, Python 3.9.18, torch 2.1.0+cpu, Lightning 2.1.1,
PyG 2.4.0, RDKit 2023.9.5. [Pathway requirements](docs/pathway/requirements.lock)
extend the isolated Phase 0 lock with openpyxl/et-xmlfile. Pip check passes.

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
loads predictor weights, bypassing the training wrapper's missing validation
embedding. Phase 0 commands/provenance: [RUNBOOK.md](docs/phase0/RUNBOOK.md).

## Frozen pathway-ranking experiment

Exact source: `rxn4chemistry/rxn_yields` commit
`d9e6b87ce1b881978490d68bfc00021e3b48127a`,
`data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx`, sheet `FullCV_01`.
The original workbook is preserved in `data/raw/buchwald_hartwig/`.
SHA-256: `d7f318abfbb42eaadca482b1e9aa54540eeed60e0089b6a0eb4e75366c31ee2e`.
All 3,955 rows are usable; no missing cells/duplicate conditions. Counts:
15 aryl halides, 4 ligands, 3 bases, 22 additives; five factorial combinations
are absent. There are five target products and three product scaffolds.

Same-product one-component comparisons produce 16,330 unique preference pairs
at a 20pp gap, including 3,197 high-similarity pairs and 1,441 cliff candidates
at a 40pp gap. Labels are binary; yields are absent from exported training data.
Scaffold-disjoint primary split: 7,907 / 4,078 / 4,345 train/validation/test pairs.

The 3-RXN encoder exposes `[N,1200]`, mean-pools to `[B,1200]`, and preserves
`[N,347]` predictions exactly. Both released BB/RXN checkpoints pass numerical
preservation, finite/deterministic embedding and batching checks. Fifteen native
trees map to all HTE records; conditions remain separate annotations.

Primary test accuracy: 50.66% overall, 57.14% high-similarity, 55.69% cliff;
ROC-AUC: 0.519 / 0.615 / 0.541. Three-scaffold macro accuracy: pretrained 48.57%,
random encoder and native Morgan controls 53.78%. No pretrained advantage is
demonstrated. Condition-only embeddings are identical: tie-adjusted accuracy
50%, AUC 0.5. Seventy-one complete three-halide groups support ordinal NDCG/MRR;
primary test has four groups, NDCG@3 0.860, MRR 0.500.

Replay from the repository root in WSL:

```bash
PYTHONPATH=src /home/doubo/.cache/cliffnet/phase0-clean/bin/python -m cliffnet.experiment \
  --checkpoint /home/doubo/projects/SynthesisNet-master/ckpts/3-RXN/ckpts.epoch=27-val_accuracy_loss=0.36.ckpt
```

Exact runnable commands, 21 passing tests, full audit/rules/results and issues:
[RUNBOOK.md](docs/pathway/RUNBOOK.md), [REPORT.md](docs/pathway/REPORT.md),
[summary.json](docs/pathway/summary.json), [ISSUES.md](docs/pathway/ISSUES.md).

## Safest future insertion points and blockers

`encode_tree()` now observes the input to the final graph convolution with a
temporary hook under `src/cliffnet/`, preserving original predictor state keys
and outputs. BB and RXN encoders are separate; ranking uses the 3-RXN encoder.
Candidate actions belong in `wrapper_decoder`/`fill_in` after feasibility/catalog
filtering and before ranking. At most two upstream files need small future seams:
`models/mlp.py` and `utils/reconstruct_utils.py`. Decoder integration is deferred.

Blockers: missing canonical chemical assets; native checkpoint loader requires
the embedding matrix; native decoder imports missing `tdc`. PyTDC installation
was dry-run only because its dependency metadata requests a second RDKit
distribution. Native Windows also lacks `fcntl`; inference runs in WSL.
All observed issues and unexecuted static findings are recorded in
[ISSUES.md](docs/phase0/ISSUES.md).

Before ACA: define a condition-aware representation, obtain broader route data
and experimental uncertainty, and validate independent chemical groups. Before
guided generation: restore canonical assets and verify fixed-skeleton decoding
and root reaction execution, then recognizer-to-tree inference. A helper's best
intermediate/BB Tanimoto alone does not verify the root product.
