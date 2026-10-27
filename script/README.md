# Data and analysis scripts

Python utilities and analysis notebooks live here. Shell launchers live in
`run/`. Run Python commands from the repository root after installing the
environment and package described in the root README.

The supplied Enamine CSV is already a matched building-block dataset, so it can
be used as input to the embedding step:

```bash
python script/02-compute-embeddings.py \
    --building-blocks-file data/assets/building-blocks/enamine_us_matched.csv \
    --output-file data/assets/building-blocks/enamine_us_emb_fp_256.npy \
    --featurization-fct fp_256
```

The numbered `00` through `07` utilities cover extraction, filtering, embedding,
synthetic tree generation, and training-data preparation. See `INSTRUCTIONS.md`
and the root README for the complete pipeline and required reaction assets.
