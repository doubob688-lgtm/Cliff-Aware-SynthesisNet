# Run configurations and launchers

This directory contains the shell launchers previously under `scripts/` and the
Conda environment definitions previously at the repository root.

From the repository root:

```bash
conda env create -f run/environment.yml
conda activate synthesisnet
pip install -e .
bash run/generate.sh
```

Other environment definitions are `run/synnet.yml`, `run/synnet_updated.yml`, and
`run/synnet_ga.yml`. Choose the one appropriate for your workflow; the root
README describes the genetic search environment.

Launchers are upstream experiment configurations. Before running them, review
dataset paths, checkpoint locations, CPU/GPU settings, and cluster-specific paths
or job commands. Use `model/` for new checkpoint artifacts. Existing launchers
may write results, caches, and logs to the paths specified in their arguments.

Shell launchers require Bash (for example, Git Bash or WSL on Windows).
