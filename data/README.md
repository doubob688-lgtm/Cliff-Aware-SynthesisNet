# Datasets and reference assets

| Location | Contents |
| --- | --- |
| `assets/building-blocks/enamine_us_matched.csv` | Supplied Enamine matched building blocks, stored in regular Git. |
| `assets/reaction-templates/hb.txt` | Hartenfeller-Button reaction templates. |
| `assets/molecules/sample-targets.txt` | Example target molecules. |
| `assets/figs/` | Figures used by the project documentation. |

The Enamine CSV was copied unchanged from the local file
`C:/Users/doubo/Downloads/enamine_us_matched.csv` on 2026-10-28. Its columns are
an exported row index and `SMILES`. The original Enamine SDF is not included.
The source file's acquisition date and matching parameters were not supplied.

The file contains 147,505 rows with no empty SMILES values and is 4,569,098 bytes.
Its SHA-256 checksum is
`eba1640981f5a1fd2a69adabb0a8cdbbb4e41dc6f75e1fb6fb8b3687e4fa27e4`.

Use the matched CSV directly for downstream embeddings. To start from raw
Enamine SDF instead, follow the extraction and filtering steps in the root
README and `INSTRUCTIONS.md`.

Store derived embeddings, reaction collections, and training datasets under
`data/`. Binary formats listed in `.gitattributes` use Git LFS. After cloning,
run `git lfs install` and `git lfs pull` to retrieve the actual file contents.
The 4.6 MB Enamine CSV uses regular Git because GitHub rejects new LFS uploads
to this public fork.
