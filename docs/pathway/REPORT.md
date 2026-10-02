# First frozen route-ranking baseline

The encoder adapter, exact-source HTE audit, binary preferences and frozen
Bradley-Terry experiment run successfully. The experiment does **not** demonstrate
an advantage from pretrained SynthesisNet embeddings over the controls. Native
tree features cannot distinguish experimental condition changes. No ACA loss,
numerical yield predictor or guided generation is implemented.

## Dataset provenance and audit

Source: [rxn4chemistry/rxn_yields, pinned XLSX](https://github.com/rxn4chemistry/rxn_yields/blob/d9e6b87ce1b881978490d68bfc00021e3b48127a/data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx),
commit `d9e6b87ce1b881978490d68bfc00021e3b48127a`. The unchanged 2,134,762-byte
workbook is tracked at
`data/raw/buchwald_hartwig/Dreher_and_Doyle_input_data.xlsx`.

SHA-256: `d7f318abfbb42eaadca482b1e9aa54540eeed60e0089b6a0eb4e75366c31ee2e`.

Only `FullCV_01!A1:E3956` is read. All 3,955 rows are usable. The source
[training script](https://github.com/rxn4chemistry/rxn_yields/blob/d9e6b87ce1b881978490d68bfc00021e3b48127a/training_scripts/launch_buchwald_hartwig_training.py)
reads this benchmark sheet directly; no additional filtering or fabricated rows
were applied. Its numerical yield models and random row split are not used.

| Audit item | Result |
| --- | --- |
| Columns | Ligand, Additive, Base, Aryl halide, Output |
| Missing cells | 0 in every column |
| Unique aryl halides / ligands / bases / additives | 15 / 4 / 3 / 22, unchanged by canonicalization |
| Exact duplicate rows / duplicate canonical conditions | 0 / 0 |
| Expected factorial combinations | 3,960; five absent combinations are recorded in summary.json |
| Products / product Bemis-Murcko scaffolds | 5 / 3 |
| Yield min / max, percentage points | 0 / 99.99999 |
| Yield mean / SD | 33.0853 / 27.2912 |
| Yield Q1 / median / Q3 | 7.8776 / 28.7617 / 53.2787 |
| Zero-yield rows | 236 |
| Yield histogram [0,20), [20,40), [40,60), [60,80), [80,100] | 1,632 / 833 / 680 / 530 / 280 |

Other workbook sheets are alternative benchmark splits and are preserved without
being combined. Products use the exact forward SMARTS and fixed p-toluidine from
the source's [data.py](https://github.com/rxn4chemistry/rxn_yields/blob/d9e6b87ce1b881978490d68bfc00021e3b48127a/rxn_yields/data.py).
All 15 aryl variants independently reproduce those products with the existing
SynthesisNet HB template at zero-based index 41.

## Trees, adapter and preference construction

Each experimental record references a completed upstream `SyntheticTree`:
aryl halide + p-toluidine → target, using RXN 41. `Skeleton.get_state()` builds
the original four-node graph and features; upstream sinusoidal PE is appended.
Ligand/base/additive are separate condition annotations. They are not converted
into fictitious stoichiometric tree children. This is an HTE single-step route
comparison proxy, not an experimentally measured multi-step pathway benchmark.

`FrozenTreeEncoder.encode_tree()` temporarily observes the input to
`GNNModel.layers[-1]`, the final TransformerConv (ModuleList index 12 for these
five-layer checkpoints). This captures the penultimate activated/dropout state
in frozen eval mode: `[N,1200]`. Mean pooling includes every node per graph and
returns `z_tree: [B,1200]`, plus unchanged `[N,347]` predictions. The adapter
contains the original predictor; loading/saving its original checkpoint keys
uses `encoder.predictor`, not a renamed upstream model.

The ranking run uses the frozen 3-RXN checkpoint. Its 15 native trees produce a
`[15,1200]` cache mapping all 3,955 experimental records to embeddings. Batched
verification captures `[60,1200]` node states and `[60,347]` predictions. The
3-NN checkpoint is independently verified on two trees. Both preserve original
BB/RXN forward values **bitwise on identical inputs**, repeat embeddings exactly
and produce finite outputs. Maximum pooled single-vs-batch error is 4.29e-6
for RXN and 1.79e-7 for BB; batched floating-point equivalence uses a tolerance.

Pair construction rules:

1. Same Buchwald-Hartwig family, exact target and fixed p-toluidine.
2. Exactly one of aryl halide, ligand, base or additive differs; other measured
   components are identical. Aryl variants change Cl/Br/I for the same product.
3. Each unordered observation pair appears once. SHA-256/seed-42 orientation
   chooses A/B independently of outcomes. Label 1 means A is preferred; 0 means B.
4. Use yields only for eligibility and binary preferences. Exported reaction
   and pair records omit numerical yields. Models receive frozen representations
   and binary labels; numerical gaps/similarities are never ranker features.
5. Audit multiple gaps, then use the user's suggested 20pp preference gap.
   The gap is not selected by held-out prediction performance.

| Minimum absolute outcome gap | Unique preference pairs |
| --- | ---: |
| 10pp | 27,008 |
| **20pp** | **16,330** |
| 30pp | 10,248 |
| 40pp | 6,129 |

There are 55,302 eligible one-component comparisons before outcome-gap filtering.
At 20pp, changed-component counts are aryl halide 1,520; ligand 1,466; base 994;
additive 12,350.

Similarity uses RDKit Morgan radius-2, 2048-bit Tanimoto. Native tree similarity
is the mean over the aligned aryl, amine and product chemical nodes, with an
identical reaction template/topology. Reaction/context similarity averages six
aligned roles: aryl, amine, product, ligand, base, additive. Constant catalyst is
excluded. Individual changed-component Tanimoto and reactant similarity are
exported alongside the averages so the operational definition is reviewable.

High-similarity pairs require reaction/context similarity ≥0.90 and native tree
similarity ≥0.80: **3,197** pairs. Route-cliff candidates additionally require a
≥40pp outcome gap: **1,441** candidates. High role-average similarity can coexist
with moderate similarity in the changed component; these are candidate flags,
not validated mechanistic cliff classifications.

## Splits and training

Group products by canonical Bemis-Murcko scaffold before assigning any pair.
All conditions, halide variants and phenyl-substituent analogues in the same
scaffold remain in one partition. Runtime checks reject shared scaffolds,
target families or underlying observations across partitions. No reverse-pair
augmentation or random pair split is used.

Primary fold:

| Partition | Chemical family | Preference pairs |
| --- | --- | ---: |
| Train | Phenyl targets: ethyl, methoxy and trifluoromethyl | 7,907 |
| Validation | 3-pyridyl target | 4,078 |
| Test | 2-pyridyl target | 4,345 |

Three outer folds rotate the three scaffolds, retaining a separate validation
scaffold for each. Only three independent scaffold families are available; pairs
share observations, so no pair-level confidence intervals or significance claims
are justified.

The head is a bias-free linear score on training-standardized frozen embeddings:
`sigmoid(score(z_A) - score(z_B))`. Train-only unique-route mean/SD normalization,
seed 0, AdamW, learning rate 0.01, weight decay 0.001 and 200 full-batch epochs are
fixed. Select the epoch using validation binary logistic loss. No encoder
parameter receives a gradient. Controls use the identical protocol with a seed-0
random Transformer encoder or pooled native Morgan features, excluding conditions.

## Ranking results

Primary pretrained test results; hard classification uses P≥0.5 for label 1.
Tie-adjusted accuracy awards half credit to equal scores. AUC accounts for ties.

| Test subset | Pairs | Accuracy | Tie-adjusted accuracy | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Overall | 4,345 | 50.66% | 50.97% | 0.519 |
| High similarity | 644 | 57.14% | 56.52% | 0.615 |
| Route-cliff candidates | 255 | 55.69% | 52.16% | 0.541 |
| Halide alternatives | 126 | 83.33% | 83.33% | 0.774 |
| Condition-only alternatives | 4,219 | 49.68% | 50.00% | 0.500 |

Three-scaffold macro averages (each held-out scaffold gets equal weight):

| Representation | Overall accuracy | High-similarity accuracy | Cliff accuracy | Halide accuracy | Overall ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen pretrained 3-RXN | 48.57% | 44.79% | 45.92% | 45.80% | 0.478 |
| Seed-0 random encoder | 53.78% | 70.47% | 72.41% | 99.37% | 0.578 |
| Native Morgan control | 53.78% | 70.47% | 72.41% | 99.37% | 0.578 |

Some halide ordering is learnable in the primary fold, but the pretrained model
does not transfer that ordering consistently across scaffolds or improve on
controls. This experiment provides **no evidence of useful additional pretrained
route-selection signal**. It does not establish that every possible pretrained
probe/representation is uninformative. There are only 15 unique native trees,
and a 1200-feature linear probe with few independent chemical families is fragile.

Across the dataset, 14,810 condition-only pairs have identical native embeddings;
all predict P=0.5. Changing the loss cannot recover absent condition information.

Valid multi-candidate evaluation uses same-target/ligand/base/additive groups with
all three halides and all three comparisons surviving the 20pp filter. Binary
preference wins give ordinal relevance 0/1/2, not numerical yields. There are
**71 groups** (primary train/validation/test: 47/20/4). NDCG@3 uses linear ordinal
gains, MRR ranks the unique preferred candidate, and score ties are averaged.

| Primary test representation | Groups | NDCG@3 | MRR |
| --- | ---: | ---: | ---: |
| Pretrained tree | 4 | 0.860 | 0.500 |
| Random encoder | 4 | 1.000 | 1.000 |
| Native Morgan | 4 | 1.000 | 1.000 |

Full fold metrics, loss histories, source audit and checkpoint verification are
in [summary.json](summary.json). [RUNBOOK.md](RUNBOOK.md) contains the exact
successful command and output-file contracts. All 21 tests pass and pip check
reports no broken requirements. Replaying the saved ranker and embedding cache
independently reproduces test accuracy 0.5065592635 and all 4,219 exact condition
ties. Nonblocking upstream skeleton-file and Torch storage warnings are recorded
with all encountered issues in [ISSUES.md](ISSUES.md).

## Before adding ACA training

The immediate blocker is a defined, tested representation of experimental
reaction conditions that preserves the upstream prediction contract. Additional
needs are broader independent chemical families, multi-step experimental routes
for pathway claims, uncertainty/replicate handling and calibrated cliff criteria.
Native complete-tree generation separately needs canonical BB/catalog/reaction
assets and a TDC-free integration boundary. No upstream decoder change or ACA
implementation is part of this baseline.
