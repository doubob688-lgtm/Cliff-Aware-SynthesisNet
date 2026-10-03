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

At completion of the tree-only experiment, the immediate blocker was a defined,
tested representation of experimental reaction conditions that preserves the
upstream prediction contract. The next section addresses this representation
blocker. Additional
needs are broader independent chemical families, multi-step experimental routes
for pathway claims, uncertainty/replicate handling and calibrated cliff criteria.
Native complete-tree generation separately needs canonical BB/catalog/reaction
assets and a TDC-free integration boundary. No upstream decoder change or ACA
implementation is part of this baseline.

## Condition-aware single-step ranking baseline

Completed in the new `e9fc` worktree on branch `codex/condition-aware-ranking`,
starting from completed baseline commit
`e26a7378c0e4ab09e90dd29863158195b1f5a495`. This is a **single-step
reaction/pathway preference benchmark**, with no multi-step optimization claim.
Phase 0, the pretrained encoder and the original baseline artifacts were not
rebuilt. All upstream `src/synnet` code is unchanged.

### Representations and comparable training

| Representation | Feature width | Input |
| --- | ---: | --- |
| Tree-only | 1200 | Validated frozen pretrained 3-RXN `z_tree` cache |
| Condition-only | 32 | Categorical ligand/base/additive one-hot blocks |
| Tree + Condition | 1232 | Concatenation of the preceding two vectors |

Condition vocabularies are fitted **separately in each fold, using only unique
training observations participating in training preference pairs**. The category
strings are the existing canonical SMILES; each role is sorted lexicographically.
Block order is ligand, base, additive. All folds contain the same 4/3/22 training
categories. A final unknown-category slot in each block gives widths 5/4/23,
offsets 0/5/9 and total width 32. Exactly one bit per role is active. There are no
trainable condition-encoder parameters. Exact vocabulary strings, offsets and
training observation IDs are recorded in
[condition_aware_summary.json](condition_aware_summary.json) and saved heads.
Held-out transforms cannot extend the vocabulary. Unknown counts are zero for
all actual partitions; unit tests verify explicit unseen-category behavior.
These are held-out **chemical scaffolds**, with shared experimental condition
categories, not a benchmark of unseen ligand/base/additive categories.

The bias-free scalar linear heads all optimize
`P(A preferred to B) = sigmoid(score(A) - score(B))` with the existing 200
full-batch epochs, seed 0, AdamW, learning rate 0.01, weight decay 0.001 and
validation logistic-loss epoch selection. Tree-block normalization uses only
unique training native trees, exactly reproducing the original baseline.
Condition-block normalization uses only unique training observations. The same
mean/population-SD (floor 1e-3) for each block is used in both its separate head
and the joint head. Validation/test influence neither vocabularies nor these
statistics; validation labels select the epoch as in the original protocol.

The workbook remains byte-identical. All 3,955 records, 16,330 oriented unique
pairs, the 20pp eligibility rule, binary labels, candidate/similarity flags and
all three scaffold fold assignments match the frozen artifacts exactly.
There is no random pair split or reverse-pair augmentation. Numerical yields
are read only by the source/pair audit and label/eligibility construction;
yield, yield gap and similarity flags never enter model feature matrices.
The original tie conventions and 71 complete three-halide groups are preserved.
NDCG@3/MRR remain **halide-group** metrics, not a new multi-candidate
condition-ranking evaluation.

### Three-scaffold held-out ablation

Each scaffold receives equal weight. Accuracies are percentages; AUC is unitless.
“Cliff Acc” refers only to the existing **route-cliff candidates**.

| Representation | Overall Acc | AUC | High-Sim Acc | Cliff Acc | Halide Acc | Ligand Acc | Base Acc | Additive Acc | Condition-only Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tree-only | 48.57 | 0.4782 | 44.79 | 45.92 | 45.80 | 49.63 | 51.01 | 49.55 | 49.65 |
| Condition-only | 86.44 | 0.9413 | 76.86 | 77.64 | 48.64 | 97.32 | 89.76 | 88.79 | 89.71 |
| Tree + Condition | 85.92 | 0.8962 | 74.90 | 73.84 | 50.68 | 97.32 | 89.76 | 88.62 | 89.58 |

Additional scaffold-macro metrics:

| Representation | Overall tie-adjusted Acc (%) | NDCG@3 | MRR |
| --- | ---: | ---: | ---: |
| Tree-only | 48.89 | 0.8247 | 0.6288 |
| Condition-only | 86.52 | 0.8100 | 0.6111 |
| Tree + Condition | 85.92 | 0.8247 | 0.6288 |

| Held-out scaffold (fold) | Test pairs | Tree-only Acc (%) | Condition-only Acc (%) | Tree + Condition Acc (%) |
| --- | ---: | ---: | ---: | ---: |
| 2-pyridyl (0, primary) | 4,345 | 50.66 | 93.16 | 94.13 |
| Phenyl target family (1) | 7,907 | 48.93 | 84.55 | 85.84 |
| 3-pyridyl (2) | 4,078 | 46.13 | 81.61 | 77.81 |

### Primary held-out test

The primary test contains 126 halide, 506 ligand, 373 base and 3,340 additive
pairs, totaling 4,345, with 644 high-similarity pairs and 255 route-cliff
candidates. Its 4,219 condition-only alternatives are ligand/base/additive pairs.

| Representation | Overall Acc | AUC | High-Sim Acc | Cliff Acc | Halide Acc | Ligand Acc | Base Acc | Additive Acc | Condition-only Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tree-only | 50.66 | 0.5190 | 57.14 | 55.69 | 83.33 | 48.81 | 49.60 | 49.82 | 49.68 |
| Condition-only | 93.16 | 0.9857 | 87.42 | 95.29 | 47.62 | 97.04 | 91.96 | 94.43 | 94.52 |
| Tree + Condition | 94.13 | 0.9772 | 94.57 | 98.04 | 83.33 | 97.04 | 91.96 | 94.34 | 94.45 |

| Representation | Overall tie-adjusted Acc (%) | NDCG@3 (4 groups) | MRR |
| --- | ---: | ---: | ---: |
| Tree-only | 50.97 | 0.8597 | 0.5000 |
| Condition-only | 93.23 | 0.8100 | 0.6111 |
| Tree + Condition | 94.13 | 0.8597 | 0.5000 |

All train/validation/test metrics, AUC and tie-adjusted accuracy for every
component and subset, loss histories and selected epochs are in the JSON.
Selected epochs by folds 0/1/2 are Tree-only 1/1/1, Condition-only 77/200/200,
Tree + Condition 43/200/98. Tree-only reproduces the previous fold metrics and
epoch choices exactly, using the existing cache.

### Collision and preservation checks

Counts cover all 3,955 observations and all 16,330 pairs; they are identical
across folds because all condition categories are present in each training fold.
Collisions mean **exact feature-vector equality before normalization**, not score
ties or a floating-point similarity threshold.

| Representation | Unique vectors | Unordered observation collisions | Preference-pair collisions | Condition-only pair collisions |
| --- | ---: | ---: | ---: | ---: |
| Tree-only | 15 | 519,426 | 14,810 | 14,810 |
| Condition-only | 264 | 27,651 | 1,520 | 0 |
| Tree + Condition | 3,955 | 0 | 0 | 0 |

Condition-only's 1,520 colliding preference pairs are all halide alternatives;
it deliberately lacks an aryl-halide input. The 1,466 ligand, 994 base and 12,350
additive preference pairs have zero condition-aware representation collisions.
Thus the missing-condition representation blocker is resolved for this dataset.

The reused `z_tree.npy` SHA-256 is
`365f5e5035f4c0ed05cfdbc471d5ae9258330d5075f334da1e2e5b7772fd258f`.
One verification pass through the released 3-RXN encoder reproduces this cache
bitwise and preserves the original output digest
`e42a2c4d8033a1b330cf8fc364775fb57387ac011b9e6a8ac8eb9c0c531396b1`.
A real Bradley-Terry backward pass leaves every backbone gradient absent,
all backbone parameters frozen and the predictor state bitwise unchanged.
The backbone also receives zero gradients across all nine head-training runs.
Checkpoint bytes and original baseline artifact hashes remain unchanged.

All **24 tests pass**, including condition encoding, deterministic vocabulary
serialization, unseen-category handling, dimensions, training-only normalization,
frozen backward/forward preservation, scaffold splits, outcome-derived pair
labels and exact collision counts. Independent replay of all nine persisted
heads reproduces all split/stratum metrics within absolute tolerance 1e-12
(AUC summation can differ in its last floating-point bit), with bitwise-identical
training vocabulary and normalization statistics. No training occurs in replay.

### Interpretation and remaining ACA blocker

**Condition-only improves over Tree-only:** macro accuracy +37.87pp and AUC
+0.4631, with higher overall accuracy in all three held-out folds.
**Tree + Condition improves over Tree-only:** macro accuracy +37.35pp and AUC
+0.4180, again with higher accuracy in all folds.
**Tree + Condition does not improve over Condition-only across held-out
scaffolds:** macro accuracy -0.51pp and AUC -0.0451. The primary accuracy gain
of +0.97pp and fold-1 gain of +1.29pp are offset by a fold-2 loss of -3.80pp;
joint AUC is lower in every fold. Joint macro high-similarity and route-cliff
candidate accuracies also decrease. These results provide **no evidence that
pretrained SynthesisNet adds transferable value beyond conditions in this probe**.

A linear score on concatenation is additive: the tree term cancels in
condition-only pairs, so this model cannot express substrate-dependent changes
in condition preference. The simple condition representation is now validated,
but the remaining blocker before ACA is scientifically validating transferable
tree-condition interactions and the cliff/uncertainty definition on broader
independent chemistry. Three scaffold families, shared observations, additive-
dominated pairs and absent independent replicates limit interpretation; no
pair-level significance claim is made. Route-cliff candidates remain operational
flags, not mechanistically validated reactivity cliffs. Multi-step experimental
routes would be needed for any multi-step pathway claim. ACA/cliff training,
guided generation and numerical yield prediction have **not** been started.
