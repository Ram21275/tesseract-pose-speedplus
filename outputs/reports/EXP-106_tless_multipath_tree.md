# EXP-106: Symmetry-aware multi-path Tesseract tree and best-fit selection on T-LESS

## Status
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

## Runs / Results / Decision
Not started.
