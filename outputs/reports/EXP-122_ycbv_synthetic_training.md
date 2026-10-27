# EXP-122: YCB-V with synthetic training renders

## Status
**Pre-registered 2026-10-27, before any feature extraction or training** (DEC-024, user: "do 1,2,3").

## Question
EXP-064 trained the YCB-V trees on real training videos only and they did not generalize: about 69° on test, while fitting train well. Standard YCB-V training adds the synthetic `data_syn` renders. Does adding them make the frozen-feature tree work on YCB-V?

## Data
- **Real rows:** exactly the EXP-064 YCB-V manifest: same six objects, train rows, held-out validation sequences and test keyframes.
- **Added synthetic rows:** `data_syn` (80,000 renders).
  - Six objects; instances with ≥ 1,000 visible label pixels; at most 12,000 per object (rng 0). These are the same cap and visibility rule as the real rows.
  - Crop, allocentric rotation and reprojection check as in EXP-064 (`scripts/exp122_ycbv_syn.py`).
- **Backgrounds.** The renders have black backgrounds, checked by looking at frames 000000 and 054321. Following DenseFusion / FFB6D, each render is pasted where label == 0 onto a real YCB-V frame. Background frames are drawn with rng 0 from the EXP-064 real **train** rows only, never from the validation or test sequences. Composites are written to `hdd2/RAM/ycbv_syn_composite`, and the frame→background map to `backgrounds.csv`.
- **Features:** DINOv3-L `tokens8`, MoGe-2 `normals16~and`, consensus masks. Clean plus 2 augmented views (photometric + exact roll homography), extracted for the synthetic rows only. The real rows reuse the EXP-064 caches.

## Arm (one; the control is EXP-064's real-only trees)
- **B: real + synthetic.** Locked recipe (DEC-018) and EXP-064 per-object trainer, unchanged except `--extra-train ycbvsyn`.
  - Normalization statistics come from the combined clean train rows.
  - 6 objects × 2 branches × seeds 0–2.
  - Checkpoint by validation greedy mean (held-out real sequences), as in EXP-064.

## Metrics
Raw geodesic error (this release has no symmetry metadata), object-balanced, 3-seed mean ± seed std. Reported for both variants (PoE and DINOv3-only) and all inference modes (greedy, orbit_product, orbit_vote32), on validation and test, plus acc@10 and the fraction above 90°.

## Criteria
1. **Synthetic data helps (primary):** B's validation PoE orbit_product mean is below EXP-064's (real only) by more than the larger seed std.
2. Test numbers are reported for every variant and mode, whatever the direction.
3. Descriptive: per-object test errors, and the distance to the training fit (generalization gap).

## Claim boundary
A pass shows the recipe needs view-diverse training data on YCB-V. It does not by itself match published YCB-V numbers, which use BOP metrics, detections and all 21 objects.

## Cost
Extraction of about 72k crops × 3 views × 2 backbones: about 4–5 h. Training: 36 trees, about 5–8 min each. Evaluation: under 1 h.

## Runs / Results / Decision
Not started.
