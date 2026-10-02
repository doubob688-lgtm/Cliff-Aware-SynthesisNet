# Frozen pathway ranking replay

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
