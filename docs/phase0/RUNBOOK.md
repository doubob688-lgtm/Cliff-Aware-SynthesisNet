# Phase 0 replay

This replays a **pretrained numerical GNN forward pass**, not complete-tree
generation. Source must remain at the recorded upstream revision plus Phase 0
tooling. All commands below use Python 3.9.18 on Linux x86_64 (WSL Ubuntu 22.04).
Use Windows Git for this Windows-managed worktree; its gitdir is not Linux-native.

## Environment

The final tested venv has no system-site-packages. The existing conda environment
supplies only the Python 3.9.18 interpreter used to create it:

```bash
/home/doubo/miniconda3/envs/synthesisnet_infer/bin/python -m venv /home/doubo/.cache/cliffnet/phase0-clean
/home/doubo/.cache/cliffnet/phase0-clean/bin/python -m pip install -r docs/phase0/requirements.lock
/home/doubo/.cache/cliffnet/phase0-clean/bin/python -m pip check
```

For another host, use a Python 3.9.18 interpreter and an isolated venv of your
choice. The lock pins 53 distributions, including CPU torch, PyG, Lightning,
RDKit, NumPy and Faiss. PyTDC, GA, docking, optional DGL encoders and GPU packages
are excluded. Source is selected through `PYTHONPATH=src`; no editable installation
from an older checkout is used.

## Assets

The released checkpoints already existed and were reused only after MD5
verification against official Figshare metadata. Exact filenames, URLs, sizes and
MD5s are in [assets.json](assets.json). On a new machine, download the two GNNs
to a local model directory and substitute their paths in the command below.
Together they are about 3.41 GB; full training/recognizer datasets are unnecessary.

Upstream `gnn.py` loads the training skeleton dictionary at import time even
though this smoke test only needs positional encoding. In the current WSL account
the required directory did not exist, so these commands created it without
overwriting any existing project:

```bash
mkdir -p /home/doubo/SynthesisNet/results/viz
curl --fail --location --retry 2 \
  --output /home/doubo/SynthesisNet/results/viz/skeletons-train.pkl \
  https://ndownloader.figshare.com/files/53202905
md5sum /home/doubo/SynthesisNet/results/viz/skeletons-train.pkl
# ed074d973a3a2ad0a158c271df296487
```

The portable required path is the current account's normal home directory plus
`SynthesisNet/results/viz/skeletons-train.pkl`. Do not change HOME. Use the
published asset with that hash; an empty/dummy pickle is not a baseline.

## Successful command

Run at the repository root inside WSL:

```bash
PYTHONPATH=src /home/doubo/.cache/cliffnet/phase0-clean/bin/python -m cliffnet.baseline \
  --checkpoint /home/doubo/projects/SynthesisNet-master/ckpts/3-RXN/ckpts.epoch=27-val_accuracy_loss=0.36.ckpt \
  --checkpoint /home/doubo/projects/SynthesisNet-master/ckpts/3-NN/ckpts.epoch=7-val_nn_accuracy_loss=0.67.ckpt
```

Exact PowerShell launcher for this checkout:

```powershell
wsl -d Ubuntu-22.04 --cd 'C:\Users\doubo\.codex\worktrees\66f9\Cliff-Aware SynthesisNet Generator' -- env PYTHONPATH=src /home/doubo/.cache/cliffnet/phase0-clean/bin/python -m cliffnet.baseline --checkpoint /home/doubo/projects/SynthesisNet-master/ckpts/3-RXN/ckpts.epoch=27-val_accuracy_loss=0.36.ckpt --checkpoint /home/doubo/projects/SynthesisNet-master/ckpts/3-NN/ckpts.epoch=7-val_nn_accuracy_loss=0.67.ckpt
```

Expected final messages:

```text
PASS: 2 pretrained GNN forward passes; report: results/phase0/forward.json
Complete-tree baseline remains unverified; Phase 1 is blocked.
```

The harness uses a fixed upstream four-node fixture, target-only partial features,
CPU, one Torch thread, eval mode and seed 0. Each model loads sequentially to
limit memory. `load_state_dict(strict=True)` checks all predictor weights, and two
forward calls must match exactly. The Lightning checkpoint's training validation
embedding path is unused by the inference module and deliberately not fabricated.
No weights, layers, scores or sampling behavior are changed.

Output JSON includes the target, partial mask, bidirectional edges, per-model
architecture, raw `[4,347]` predictions, finiteness/determinism checks and hashes.
Only a compact summary is committed at [forward-summary.json](forward-summary.json).
The first 256 values are BB embeddings, the final 91 are RXN scores. Outputs for
non-frontier nodes and unmasked argmaxes are diagnostic, not candidate actions.
No catalog lookup, recognizer or reaction execution is validated by this command.

## Tests

```bash
PYTHONPATH=src /home/doubo/.cache/cliffnet/phase0-clean/bin/python -m unittest tests.test_phase0_baseline tests.test_Optimization -v
```

Seven tests pass: four Phase 0 tests and three upstream optimization tests.
New tests use a tiny temporary checkpoint and compare to the original upstream
Lightning loader with valid test-only fingerprint data. They also verify that
hidden labels do not leak into input and incomplete/extra checkpoint weights are
rejected. No released catalog is replaced. See [ISSUES.md](ISSUES.md) for native
generation attempts and what must be resolved next.
