# Model artifacts

Store downloaded or trained checkpoint directories here. Model implementations
remain in `src/synnet/models/` so Python imports continue to work.

The upstream project links these pretrained assets:

- [Surrogate checkpoints](https://figshare.com/articles/dataset/SynthesisNet_surrogate_checkpoints/28557581)
- [Supporting pickle assets](https://figshare.com/articles/dataset/SynthesisNet_assets/28658243)

For example, place building-block and reaction checkpoints under `model/3-NN/`
and `model/3-RXN/`, then pass those paths to `--ckpt-bb` and `--ckpt-rxn`.
The number must match the maximum reaction count used to train the checkpoints.
Supporting datasets belong in `data/`; some legacy launchers expect skeleton
assets under `results/viz/`, so adjust their path arguments as needed.

Checkpoint formats listed in `.gitattributes` use Git LFS. No pretrained weights
are included in this directory yet. Small test checkpoints remain in
`tests/data/ref/` as test fixtures.
