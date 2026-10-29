# EXP-106: Symmetry-aware multi-path Tesseract tree and best-fit selection on T-LESS

## Status
**On hold (DEC-011, 2026-10-13):** runs in the cross-dataset block together with EXP-064, after the SPEED+ phases. Protocol below unchanged.

**Pre-registered 2026-10-13, before any data was processed** (DEC-010, user request). Runs after EXP-105. Pulls T-LESS work ahead of EXP-064's Phase-6 slot; EXP-064 itself is unchanged.

## Research question
T-LESS objects have orientations that look identical (true symmetries) or nearly so. Proposed by the user: let **each look-alike orientation have its own path in the tree**, so that the probability chain splits across those paths instead of forcing one arbitrary label. Then (a) does the multi-path tree predict better under the symmetry-aware metric than single-path training, and (b) can a best-fit step (render-and-compare with the CAD model, flow/diffusion scoring) pick the right candidate among the paths and their flips?

## Data (DEC-009 paths; read-only)
- **Objects (declared now, before any evaluation):** 1, 4 (continuous symmetry); 5, 11 (one discrete 180° symmetry); 18, 21 (no symmetry in the BOP metadata, but visually near-symmetric); 27 (three discrete symmetries). Spans the symmetry classes and sizes (58–152 mm).
- **Train:** `train_pbr` scenes 0–44 plus `train_primesense` (real) for these objects; instances with visible fraction ≥ 0.3; at most 12,000 crops per object (seeded).
- **Validation (selection):** `train_pbr` scenes 45–49 (whole scenes held out).
- **Test:** the BOP19 targets of these objects in `test_primesense` (all instances); test-only, never used for selection.
- **Input:** RGB, GT square crop from the projected model bounding box (same crop rule as SPEED+), frozen DINOv3-L `tokens8` and MoGe-2 `normals16~and`; equal-weight PoE of the two trees, one tree per branch shared across objects with a learned object embedding added to the tokens. Normalization statistics from train only.
- **Rotation convention:** the tree predicts the **allocentric** rotation (camera-ray corrected, so the crop's appearance depends only on it). Conversion to egocentric for rendering is verified by reprojecting the model's bounding-box corners onto the image before training.
- **Symmetry sets:** from `models_info.json`; continuous symmetries discretized to 36 steps around their axis; discrete ones used exactly.

## Arms (same encoders, tree architecture, recipe and budget as EXP-036)
- **A0 single path:** GT path of the annotated rotation (raw labels).
- **A1 canonical path:** the equivalent rotation closest to identity (standard canonicalization baseline).
- **A2 multi-path (proposed):** the target is the per-level mass of the whole symmetry set {R·S}. Implemented as the EXP-044 level-marginal loss with the equivalent rotations in place of flow samples, so each look-alike orientation gets its own tree path and the probability chain divides between them. Objects without metadata symmetry reduce to A0.

## Best-fit selection (on top of the selected arm)
Candidates: top-8 beam leaves, plus their 180° flips about the model's x / y / z axes and its declared symmetry axes, plus leaf-cell corners (separate arms, as in EXP-105). Scorers:
1. tree PoE log-probability (control);
2. **render-and-compare:** CAD model rendered (PyTorch point-splat of the dense model surface, no new libraries) with GT translation, compared at 74 × 74 with MoGe-2's predicted normals (mean cosine) and foreground mask (IoU), 2a / 2b / sum;
3. **flow / diffusion:** an EXP-042-recipe flow head trained on T-LESS train, scored by exact log-density and by re-noise / return (as EXP-105). Stage 2: run only if stages 1–2 complete.

Gate before use: on validation, the renderer must rank the GT rotation above its non-symmetric flips in most cases. Otherwise a convention is wrong and the renderer is not used.

## Metrics
- **Primary:** symmetry-aware geodesic error (minimum over the symmetry set) per object and object-balanced, test and validation.
- Raw geodesic error, reported separately.
- **Mode recall:** fraction of distinct look-alike orientations that appear among the top-8 beams.
- acc@10/20, fraction > 90° and > 150°, oracle best-of-candidates, runtime.

## Pre-registered criteria
1. **Multi-path helps:** A2's validation symmetry-aware mean (object-balanced, 3 seeds) is below the better of A0 and A1 by more than the larger seed std.
2. **Best fit helps:** a scorer's re-ranked validation symmetry-aware mean is below the selected arm's greedy mean by more than the larger seed std.
3. Test numbers are reported for every arm whatever the direction.

## Baselines
Internal: A0, A1, greedy, oracle. External T-LESS rotation results (e.g. SpyroPose, LiePose, Implicit-PDF) are taken from the papers and labelled matched only if crop, symmetry metric and test subset match; otherwise they are listed as unmatched context. The local `liepose_data` T-LESS logs use a different test set (6,815 crops) and pipeline, so they are not a baseline.

## Resources
About 84k crops in total. DINO `tokens8` is about 128 KB per crop, about 11 GB in total. Cached on hdd2 (280 GB free), not root. Feature extraction is a GPU job in the guarded queue.

## Amendment 2026-10-13 (before any run): arm K, the original Kurz et al. (2017) corner formulation
Requested by the user (DEC-010 addendum). Added next to the other arms; nothing else changes.
- **Representation (as in the original paper):** all **8** facets of the tesseract (±w, ±x, ±y, ±z), so q and −q are both represented, and rotations are represented by the **corners** of the subdivided cubes (radially projected), not by cube centres. A corner is shared by up to 8 neighbouring cubes, so one rotation has many tree paths.
- **Tree:** the same Transformer, with an 8-way root (one per facet) and 8 children per level, depth 5.
- **Training target (multi-path):** for each training rotation, the nearest corner representative, for both q (on its facet) and −q (on the opposite facet); the target mass is spread uniformly over every leaf cube that has that corner, in both copies. Implemented with the EXP-044 level-marginal loss, using these paths in place of samples.
- **Decoding:** beam search (top 8 leaves); each leaf votes its probability to its 8 corners; votes for the same rotation (same corner, either copy) are summed; the prediction is the corner with the largest summed mass.
- **What is compared:** K vs the single-path centre tree (EXP-036 recipe, same features, budget and seeds), under each experiment's own metrics and criteria.
- **Expectation, stated before results:** q and −q are the same rotation, and corners shared between cubes are only a few degrees apart. So the redundancy is not expected by itself to resolve 180° flips (different rotations, far apart on the hypersphere); any gain is expected in precision and at chart seams. Reported whatever the outcome.

## Runs / Results / Decision (results recorded 2026-10-29; see the 2026-10-27 amendment for the arms actually run)
- **Evaluations:**
  - T-LESS: A0 `RUN-20261027-201558`, A1 `RUN-20261027-232339`, A2 `RUN-20261028-221625`.
  - SYMSOL-T: A0 `RUN-20261028-014408`, A1 `RUN-20261028-041345`, A2 `RUN-20261028-120353`.
- **Criterion 1** (validation, PoE greedy, object-balanced, 3 seeds):
  - **T-LESS: PASS.** A2 10.82 ± 0.11 vs the better of A0/A1, A1 at 11.46 ± 0.04.
  - **SYMSOL-T: PASS.** A2 2.72 ± 0.01 vs A1 3.40 ± 0.01.
- **Descriptive:**
  - **T-LESS validation orbit:** A0 9.25, A1 10.71, A2 8.28.
  - **T-LESS test (PoE + orbit), R@2/5/10 and mean:**
    - A0: 4.0 / 36.0 / 73.5, 16.6°;
    - A1: 2.7 / 26.7 / 65.9, 18.2°;
    - A2: 5.7 / 48.1 / 79.0, 15.2°.
  - **SYMSOL-T test (DINOv3 orbit):** A2 2.07 vs A0 2.62.
  - **Canonical labels (A1) break orbit voting:** SYMSOL-T validation orbit is 9.73.
- **Criterion 2:** not run. The best-fit stage is out of scope (2026-10-27 amendment).
- **Decision:** A2 is the base arm for EXP-119 on both datasets.

## Amendment 2026-10-16 (DEC-018, before any run): locked recipe
This experiment uses the **EXP-109 methodology** (augmentation-trained trees, with the in-plane rotation label transform derived for the dataset's camera / crop convention; orbit PoE voting at test time). It is reported in **both** variants, PoE (DINOv3 + MoGe-2) and DINOv3-only, each with and without orbit voting. The in-plane label transform for each dataset is derived and verified by projecting its 3-D model points, before any training, as in EXP-109 (`scripts/exp109_convention_geometry.py`). Other protocol details are unchanged.

## Amendment 2026-10-27 (DEC-024, before any EXP-106 run)
The user asked for the symmetry fix ("do 1,2,3"). EXP-106 is run in the following scope; everything not listed is unchanged.
- **Tree setup: per-object trees**, not one shared tree with an object embedding. This matches EXP-064, so EXP-064's existing per-object trees are a matched **A0**: same code path (`scripts/train_xobj.py`, whose defaults reproduce EXP-064), recipe, data, seeds and checkpoint rule.
- **Arms trained now** (2 branches × seeds 0–2 per object):
  - **A1 canonical** (`--label canonical`): in every view, the equivalent rotation R·S closest to identity.
  - **A2 multi-path** (`--label multi`): the recipe's geodesic soft-target CE, averaged over the equivalent rotations R·S. All of them are used when |S| ≤ 8; otherwise 8 are drawn per row and step (unbiased). This is the level-marginal target of the whole symmetry set under teacher forcing, kept geodesically smoothed.
  - **Deviation from the text above:** the pre-registration names the EXP-044 hard-path level-marginal loss. That loss would not reduce to A0 for objects without symmetry, which the pre-registration also requires, and it would drop the recipe's soft targets. The smoothed version reduces exactly to A0 when |S| = 1.
- **Datasets:**
  - T-LESS, the pre-registered 7 objects.
  - **SYMSOL-T added** (5 shapes, LiePose split; symmetry sets from the annotations: tet 12, cube 24, icosa 60, cone 360, cyl 720). Criteria are the same, applied per dataset.
  - SYMSOL-T MoGe-2 trees use `--sd-floor 1e-2` for cube / cyl / icosa, exactly as the A0 trees (EXP-064 fix).
- **Out of scope for this run:** arm K (Kurz corner formulation) and the best-fit stage (render-and-compare, flow / diffusion scoring). They stay pre-registered and unrun.
- **Metric for criterion 1:** validation symmetry-aware mean, PoE greedy, object-balanced, 3 seeds. DINOv3-only and orbit modes are reported too.
- **Test:** T-LESS BOP19 targets and SYMSOL-T `test-5k`, reported for every arm.
- Symmetry composition was unit-checked before training: quaternion product q·s equals R·S to 1e-15, and roll commutes with right-multiplication by S.
