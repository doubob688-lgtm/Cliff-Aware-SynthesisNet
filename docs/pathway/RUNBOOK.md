# Frozen pathway ranking replay

The first section records the completed historical Tree-only run. For the new
condition-aware experiment, use the command in the final section; do not rerun
Phase 0 or rebuild the original frozen baseline.

Run from the repository root in WSL Ubuntu 22.04 with Python 3.9.18. See the
[Phase 0 runbook](../phase0/RUNBOOK.md) for environment creation and published
checkpoint/skeleton downloads. No TDC is used or required. Existing upstream
`gnn.py` still requires its verified skeleton pickle at import time.

```bash
/home/doubo/.cache/cliffnet/phase0-clean/bin/python -m pip install -r docs/pathway/requirements.lock
/home/doubo/.cache/cliffnet/phase0-clean/bin/python -m pip check
PYTHONPATH=src /home/doubo/.cache/cliffnet/phase0-clean/bin/python -m unittest \
  tests.test_cliffnet_encoder tests.test_cliffnet_preferences \
  tests.test_phase0_baseline tests.test_Optimization -v
PYTHONPATH=src /home/doubo/.cache/cliffnet/phase0-clean/bin/python -m cliffnet.experiment \
  --checkpoint /home/doubo/projects/SynthesisNet-master/ckpts/3-RXN/ckpts.epoch=27-val_accuracy_loss=0.36.ckpt \
  --verify-checkpoint /home/doubo/projects/SynthesisNet-master/ckpts/3-NN/ckpts.epoch=7-val_nn_accuracy_loss=0.67.ckpt
```

Exact PowerShell launcher used for this worktree:

```powershell
wsl -d Ubuntu-22.04 --cd 'C:\Users\doubo\.codex\worktrees\66f9\Cliff-Aware SynthesisNet Generator' -- env PYTHONPATH=src /home/doubo/.cache/cliffnet/phase0-clean/bin/python -m cliffnet.experiment --checkpoint /home/doubo/projects/SynthesisNet-master/ckpts/3-RXN/ckpts.epoch=27-val_accuracy_loss=0.36.ckpt --verify-checkpoint /home/doubo/projects/SynthesisNet-master/ckpts/3-NN/ckpts.epoch=7-val_nn_accuracy_loss=0.67.ckpt
```

Replace local interpreter/checkpoint paths when replaying on another machine.
The downloader fetches the pinned XLSX only if absent, then verifies SHA-256.
CPU, one Torch thread, encoder seed 0, split seed 42, 200 ranker epochs, learning
rate 0.01 and validation logistic-loss checkpoint selection are fixed in code.
Splits use product Bemis-Murcko scaffolds, not the source's random FullCV split.

Outputs under ignored `results/pathway/`:

| File | Contents |
| --- | --- |
| `summary.json` | Dataset audit, thresholds, splits, checkpoint checks, three scaffold folds and metrics for all controls |
| `reactions.jsonl` | 3,955 experimental route records: native single-step tree plus separate condition annotations; no yields |
| `pairs.jsonl` | 16,330 unique preference pairs; target/context, tree IDs, binary label, split and similarity/candidate flags |
| `multi_candidate_groups.jsonl` | 71 complete three-halide groups; ordinal relevance from binary preference wins |
| `z_tree.npy` | `[15,1200]` frozen 3-RXN native-tree cache; `aryl_order` and `route_index` in summary map 3,955 records to it |
| `random_z_tree.npy` | Equivalent cache for seed-0 random encoder control |
| `*_ranker.pt` | Primary-fold linear ranking weights, training-only mean/scale, feature width and pairwise loss metadata |

Score a cached route by applying `(z - training_mean) / training_scale` and the
saved `score.weight`; compute `sigmoid(score_A - score_B)` for preferences.
Encoder weights are never saved into the ranker checkpoint. To use the adapter,
first load the original predictor with `load_predictor`, then wrap it in
`FrozenTreeEncoder`; call `encode_tree(Data/Batch)` for hidden nodes, pooling and
unchanged predictions. Call `encoder(data)` for the original forward interface.
Save original predictor keys via `encoder.predictor.state_dict()` if needed.

Tests cover frozen eval/determinism, finite outputs, pooling/batching, exact
prediction/state preservation, raw-file immutability, benchmark counts, native
tree semantics, one-change/gap rules, orientation/split independence from
outcomes, scaffold/observation leakage checks, Bradley-Terry learning and valid
tie-aware NDCG/MRR groups. Actual released-checkpoint preservation checks run in
the experiment command, in addition to the small-model unit tests.

## Condition-aware single-step ranking baseline

Successful environment: the **existing** WSL Ubuntu 22.04
`/home/doubo/.cache/cliffnet/phase0-clean/bin/python` (Python 3.9.18,
Torch 2.1.0 CPU, RDKit 2023.09.5). No dependency installation or environment
rebuild was needed. Worktree:
`C:\Users\doubo\.codex\worktrees\e9fc\Cliff-Aware SynthesisNet Generator`,
branch `codex/condition-aware-ranking`, based on the completed
`codex/pathway-ranking` commit `e26a7378c0e4ab09e90dd29863158195b1f5a495`.

The ignored frozen artifacts were copied into **this new worktree**, with the
source worktree left unchanged. Exact successful preparation from the new
worktree root:

```powershell
New-Item -ItemType Directory -Force results/pathway | Out-Null
$baselinePath = 'C:\Users\doubo\.codex\worktrees\66f9\Cliff-Aware SynthesisNet Generator\results\pathway'
Copy-Item -LiteralPath "$baselinePath\summary.json","$baselinePath\reactions.jsonl","$baselinePath\pairs.jsonl","$baselinePath\multi_candidate_groups.jsonl","$baselinePath\z_tree.npy","$baselinePath\pretrained_tree_ranker.pt" -Destination results/pathway
```

The tracked pinned XLSX is already present. The runner refuses altered cache or
pair hashes, checks the source checksum, compares all observations/pairs/flags
and fold assignments to the frozen baseline, then verifies released encoder
outputs and an actual ranking backward pass. It uses the cache for every head;
the one encoder pass only verifies preservation. No random/Morgan encoder
controls, Phase 0 experiment, checkpoint training or guided decoding are rerun.
The 3-RXN checkpoint defaults to the path stored in the frozen summary;
`--checkpoint` may override its location, but forward/cache equality is required.

Exact successful complete benchmark command:

```powershell
wsl -d Ubuntu-22.04 --cd 'C:\Users\doubo\.codex\worktrees\e9fc\Cliff-Aware SynthesisNet Generator' -- env PYTHONPATH=src /home/doubo/.cache/cliffnet/phase0-clean/bin/python -m cliffnet.condition_experiment --baseline-dir results/pathway --output results/pathway/condition_aware --report docs/pathway/condition_aware_summary.json
```

Exact test command (24 passing tests):

```powershell
wsl -d Ubuntu-22.04 --cd 'C:\Users\doubo\.codex\worktrees\e9fc\Cliff-Aware SynthesisNet Generator' -- env PYTHONPATH=src /home/doubo/.cache/cliffnet/phase0-clean/bin/python -m unittest tests.test_cliffnet_conditions tests.test_cliffnet_encoder tests.test_cliffnet_preferences -v
```

Independent replay command, with **no training**:

```powershell
wsl -d Ubuntu-22.04 --cd 'C:\Users\doubo\.codex\worktrees\e9fc\Cliff-Aware SynthesisNet Generator' -- env PYTHONPATH=src /home/doubo/.cache/cliffnet/phase0-clean/bin/python -m cliffnet.condition_replay
```

Replay checks all nine saved heads across all three partitions and component/
similarity strata, including NDCG/MRR. It rebuilds vocabularies and normalization
statistics from training observation IDs only and matches those saved values
exactly. Metric tolerance is absolute 1e-12, allowing last-bit AUC summation
differences; hard accuracy and tie counts match.

| Output | Contract |
| --- | --- |
| `docs/pathway/condition_aware_summary.json` | Tracked full results: all folds/splits/component metrics, losses, comparisons, vocabulary and training IDs, collision counts, source/cache hashes and frozen-backbone verification |
| `results/pathway/condition_aware/summary.json` | Identical runtime metrics JSON |
| `results/pathway/condition_aware/{tree_only,condition_only,tree_condition}_fold{0,1,2}_ranker.pt` | Nine scalar ranking heads, training mean/scale, feature width, representation name and condition vocabulary where applicable; no backbone weights |
| `results/pathway/condition_aware/z_condition_fold{0,1,2}.npy` | `[3955,32]` deterministic condition encodings in original `reactions.jsonl` order |
| `results/pathway/condition_aware/replay_checks.json` | Independent saved-head replay and train-only statistics checks |

Inference: Tree-only uses the original `route_index` into `z_tree.npy`.
Condition-only encodes the three condition strings using the saved vocabulary.
Tree + Condition concatenates that tree vector and condition encoding in this
order. Apply saved `(features - training_mean) / training_scale`, then
`score.weight`; compare two scalar scores with the Bradley-Terry sigmoid.
Numerical yield, yield gap, scaffold ID and similarity/candidate flags are not
features. Stop at reporting this single-step ablation; do not invoke ACA.
