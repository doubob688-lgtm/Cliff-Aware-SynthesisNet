# Phase 0 issue log

Observed during the 2026-10-03 audit of upstream commit `2fefd24`. No original
source fixes were applied. Setup changes were restricted to new isolated WSL
environments, a published skeleton asset and new baseline tooling.

## Observed failures and limitations

| Issue | Evidence and disposition |
| --- | --- |
| No configured Git author / detached worktree | Created `codex/phase0-baseline`; user authorized `doubob688-lgtm` with `234836038+doubob688-lgtm@users.noreply.github.com` in local Git config. |
| Windows Python command is an app alias; conda is not on Windows PATH | Located existing WSL Ubuntu 22.04 / Python 3.9.18. Native decoder imports Unix `fcntl`, so WSL is the selected runtime. |
| Existing Python environment lacks Lightning and TDC; torch is 1.12.1 | Preserved it. Initially made a system-site-packages overlay with torch 2.1.0+cpu / Lightning 2.1.1, then installed a fresh venv from the saved 53-package lock. |
| Historical environment export is not a portable Windows specification | Linux build IDs, CUDA/triton packages; conda torch 1.12.1 plus pip torch 2.1.0 and both RDKit distributions. Export retained unchanged; numerical smoke runtime uses an explicit CPU lock. |
| `load_gnn_from_ckpt()` fails despite a present checkpoint | Actual traceback ends at `models/mlp.py:291`, `np.load(X)`: missing `data/assets/building-blocks/enamine_us_emb_fp_256.npy`. That field serves training validation; Phase 0 harness restores only unchanged `GNNModel` strictly. Native loader remains blocked. |
| Decoder cannot import without skeleton asset at an absolute HOME-derived path | `models/gnn.py:37` requests `/home/doubo/SynthesisNet/results/viz/skeletons-train.pkl` during import. Downloaded the real 84,253,844-byte published asset at that exact path and verified MD5. No HOME override, placeholder pickle, or source patch. |
| Decoder import then fails on TDC | `reconstruct_utils.py:41`: `ModuleNotFoundError: No module named 'tdc'`. The native one-target command below fails at this import before any generation. |
| PyTDC dry run requests overlapping RDKit distribution | `pip install --dry-run pytdc==0.4.1` proposes `rdkit-pypi==2022.9.5` alongside installed `rdkit==2023.9.5`, plus optional oracle dependencies. No install performed; chemical runtime preserved. Resolve separately before complete inference. |
| Canonical chemical assets unavailable | Checkout and release metadata do not supply the matched Enamine CSV / aligned 256-bit FP matrix / full available-reactant collection. The vendor catalog requires access. No synthetic replacement was created. |
| Existing old-folder chemical assets are customized | Its `enamine_us_emb_fp_256.npy` is only 2,176 bytes / two rows; companion files are `cas_azide*`. Rejected as baseline catalog. Only official hash-matching checkpoints were reused. |
| Linux git cannot read Windows worktree gitdir paths | Provenance snapshot initially failed because `.git` contains a Windows absolute gitdir. Used native Windows Git to read the upstream SHA and commit; did not rewrite `.git`. Python inference works from the WSL-mounted tree. |
| Windows document-check helper defaulted to GBK | Its first UTF-8 Markdown read raised UnicodeDecodeError. Specified UTF-8 for the helper; no repository file content or model behavior changed. |
| Package-index metadata warnings in clean install | Old pip discarded CPU-index Jinja2/typing-extensions candidates for name normalization, then successfully used PyPI wheels. Installation succeeded; dependency check passes. |
| Upstream skeleton file handle warning | Tests emit `ResourceWarning: unclosed file` at `gnn.py:37`. It is an upstream import-side effect; tests pass and source is unchanged. |
| Automated Figshare web read unavailable | Web reader could not open article API endpoints. Official Figshare REST metadata was read directly over HTTPS; local MD5s were checked against it. |

## Static code findings (not reproduced by full decoding)

- `GNN.forward(data, mask)` calls `self.model(x, edge_index)` although
  `GNNModel.forward()` expects `Data` and an attention flag. Existing inference
  correctly returns/calls `.model`; do not use the Lightning wrapper's forward.
- `GNN.__init__(X=None)` refers to undefined `molembedder`. Do not bypass the
  missing asset by passing None; no patch was made.
- `reconstruct-targets.py` has a hardcoded `index == 1`, four-node selection in
  the no-data path. Its no-recognizer lookup stores a `Skeleton`, while batching
  expects a list. This entry point needs careful case selection before use.
- Topological decoding enumerates every order before applying
  `max_topological_orders`; large skeletons can be costly even with a small cap.
- `reconstruct()` compares all chemical nodes and can return an intermediate or
  BB. A nonzero similarity is not evidence of a valid complete root product.
- `--data` in reconstruct-targets.py accepts only a path containing `chembl`
  with `canonical_smiles`; the bundled sample-target text is not directly accepted.
- `tests/_test_Predict.py` imports obsolete `syn_net.models.chkpt_loader`.
  Legacy `_test*` suites are not discovered by default; upstream README itself
  identifies only the optimization suite as working. Passing those three tests
  does not validate the GNN decoder.
- Optional paths refer to author cluster directories (`/ssd`, `/dccstor`, `/u`).
  Program-cache filters, GA stump lookup and docking need additional assets.
  The two uninitialized submodules are not needed for numerical GNN inference.

## Exact failed commands

All Python commands below ran in WSL from the current repository root, with
`PYTHONPATH=src`. The initial numerical runtime is called `phase0-env`; the
final clean tested runtime is `phase0-clean`.

```bash
PYTHONPATH=src /home/doubo/.cache/cliffnet/phase0-env/bin/python -c 'from synnet.models.common import load_gnn_from_ckpt; load_gnn_from_ckpt("/home/doubo/projects/SynthesisNet-master/ckpts/3-RXN/ckpts.epoch=27-val_accuracy_loss=0.36.ckpt")'
# FileNotFoundError: data/assets/building-blocks/enamine_us_emb_fp_256.npy

PYTHONPATH=src /home/doubo/.cache/cliffnet/phase0-env/bin/python -c 'import synnet.utils.reconstruct_utils'
# Before asset download: FileNotFoundError: /home/doubo/SynthesisNet/results/viz/skeletons-train.pkl
# After asset download: ModuleNotFoundError: No module named 'tdc'

PYTHONPATH=src /home/doubo/.cache/cliffnet/phase0-env/bin/python scripts/reconstruct-targets.py \
  --ckpt-rxn /home/doubo/projects/SynthesisNet-master/ckpts/3-RXN/ckpts.epoch=27-val_accuracy_loss=0.36.ckpt \
  --ckpt-bb /home/doubo/projects/SynthesisNet-master/ckpts/3-NN/ckpts.epoch=7-val_nn_accuracy_loss=0.67.ckpt \
  --skeleton-set-file /home/doubo/SynthesisNet/results/viz/skeletons-train.pkl \
  --out-dir results/phase0/native --num 1 --ncpu 1 \
  --top-k 1 --top-k-rxn 1 --strategy conf --test-correct-method reconstruct
# ModuleNotFoundError: No module named 'tdc'; no predictions produced
```

Phase 1 gate remains closed. Obtain the canonical aligned assets and settle the
native runtime before any experimental data, encoding API change or loss work.
