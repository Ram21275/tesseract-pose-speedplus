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

## Runs / Results / Decision
Not started.

## Amendment 2026-10-16 (DEC-018, before any run): locked recipe
This experiment uses the **EXP-109 methodology** (augmentation-trained trees, with the in-plane rotation label transform derived for the dataset's camera / crop convention; orbit PoE voting at test time). It is reported in **both** variants, PoE (DINOv3 + MoGe-2) and DINOv3-only, each with and without orbit voting. The in-plane label transform for each dataset is derived and verified by projecting its 3-D model points, before any training, as in EXP-109 (`scripts/exp109_convention_geometry.py`). Other protocol details are unchanged.
