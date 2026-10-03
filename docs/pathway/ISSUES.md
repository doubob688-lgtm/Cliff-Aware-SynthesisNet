# Observed issues and scope limits

1. **Condition information is absent from native trees.** There are 3,955 HTE
   records but only 15 distinct native single-step trees. Ligand, base and
   additive changes are metadata; all 14,810 condition-only preference pairs
   have equal tree embeddings and probability 0.5. Do not encode these reagents
   as fabricated stoichiometric BB children or call such a topology upstream.
   **Condition-aware update:** a separate training-only categorical encoder now
   resolves all 14,810 condition-only pair collisions; native trees are unchanged.
2. **Few independent chemical families.** The five products have three
   Bemis-Murcko scaffolds. The final splits keep scaffolds, target families,
   underlying observations and all halide variants disjoint. Each fold has one
   scaffold for training, one for validation and one for testing. Thousands of
   overlapping pairs do not imply thousands of independent chemical examples.
3. **HTE single-step scope.** These records compare reaction conditions and
   halide choices for one coupling; they do not establish multi-step pathway
   ranking performance or experimental multi-step route outcomes.
4. **Incomplete factorial design.** FullCV_01 contains 3,955 of 3,960 possible
   component combinations. Five absent combinations are listed in the audit;
   source sheets were not merged and missing experiments were not imputed.
5. **Uncertainty absent.** This sheet supplies no per-reaction error bars or
   independent replicate IDs. The 20pp preference / 40pp cliff gaps and
   similarity thresholds are explicit heuristics, not validated cliff truth.
6. **Native decoder remains blocked.** Canonical Enamine/catalog assets and
   available-reactant collection are still missing; upstream reconstruction
   imports TDC. This experiment imports neither TDC nor reconstruction, and
   verifies the recorded reactants/products with upstream HB template 41 using
   RDKit directly. Full synthetic-tree generation remains unverified.
7. **Upstream import side effects persist.** `synnet.models.gnn` loads the
   published skeleton pickle through a hardcoded home path and emits an
   unclosed-file ResourceWarning. No upstream fix was applied. Our workbook
   loader's own unclosed ExcelFile warning was corrected with a context manager.
8. **Test fixture repair.** Adding multi-candidate metrics exposed a missing
   `target_context` in the synthetic logistic-loss unit fixture. The fixture was
   corrected to match the route-preference schema; actual experiment records
   already contained that field.
9. **Environment tooling.** WSL has no `rg`; native Windows `rg` was used for
   repository searches. Windows Git is required for the managed worktree. The
   two new XLSX-reading dependencies are pinned in the pathway lock; pip check
   passes. No global/conda environment or original RDKit installation was changed.
   Replaying the saved ranking head emits Torch 2.1's TypedStorage deprecation
   warning; the saved weights and cache reproduce the recorded accuracy exactly.

10. **No demonstrated additional pretrained-tree value.** The completed
    condition-aware scaffold-held-out macro accuracy is 86.44% for Condition-only
    versus 85.92% for Tree + Condition; AUC is 0.9413 versus 0.8962. Gains over
    Tree-only do not establish a benefit from SynthesisNet beyond conditions.
11. **Additive head cannot learn tree-condition interactions.** A scalar linear
    head on concatenation cancels the tree term whenever only a condition
    changes. It cannot express a substrate-dependent condition ordering.
    Validation epoch choices differ between the separate and joint models.
12. **Condition-category and multi-candidate coverage.** All actual held-out
    condition categories appear in the training vocabulary; unknown handling is
    tested but these results do not measure unseen-category generalization.
    The preserved NDCG/MRR groups vary halides only. Condition-only cannot
    distinguish those candidates and receives tie-averaged group metrics.
13. **Replay floating-point tolerance.** Independent saved-head replay matches
    metrics with absolute tolerance 1e-12. One AUC summation differed by about
    1e-16 during the initial strict equality check; the replay comparison was
    corrected to a declared numeric tolerance. Vocabulary/statistics are checked
    exactly and hard accuracy/tie counts are unchanged. This is not a training,
    split or label difference.

The first condition representation is now defined and tested on the original
scaffold splits. Before ACA, validate transferable chemistry-condition
interactions and calibrated route-cliff/uncertainty definitions using broader
independent chemistry. Only three scaffold families and no replicate uncertainty
remain a scientific limitation. Obtain experimental multi-step routes before
making pathway-level claims. Preserve the original checkpoint/output contract;
native generation assets remain a separate blocker for guided decoding. No
ACA/cliff loss, guided generation or numerical yield predictor is implemented.
