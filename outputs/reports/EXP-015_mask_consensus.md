# EXP-015: Two-way DINOv3 ↔ MoGe-2 foreground consensus (training-free)

## Status
Running. Pre-registered on 2026-10-08, before any mask metric was computed.

## Research question
MoGe-2's foreground mask fails on real images (EXP-013). Does a training-free DINOv3 foreground fix MoGe-2 where it fails, and does MoGe-2 sharpen or fix DINOv3 where DINOv3 fails? In other words, is there a **two-way** consensus?

User-requested experiment. It is outside the planned EXP list and corresponds to a Phase-2 training-free fusion question.

## Hypothesis
- **DINO → MoGe-2:** on lightbox and sunlamp, intersecting MoGe-2's mask with a DINO foreground removes MoGe-2's background over-segmentation. Precision rises, at a small recall cost.
- **MoGe-2 → DINO:** MoGe-2's 74×74 mask has sharper boundaries than DINO's upsampled 16×16 patch map, so the consensus beats DINO alone on boundary accuracy.

## Component under test
Fusion (foreground evidence)

## Track
Training-free. No learned parameters, no fitted thresholds, no dataset statistics.

## Pre-registered protocol (fixed before any result)
1. **DINO foreground.**
   - Input: DINOv3-L/16 last-layer patch tokens (16×16×1024, cached; 256-px GT crop).
   - Per image, centre the 256 tokens and project them on their first principal direction (SVD). That projection is the score.
   - Sign: flip so the mean score of the outer ring of 60 border patches is below the mean of the interior.
   - Bilinear upsample (align_corners=False) to 74×74. Binary mask = per-image Otsu threshold on the upsampled score.
2. **MoGe-2 foreground.** Cached MoGe-2 mask (area-pooled to 74×74 from 518-px binary mask), thresholded at > 0.5.
3. **Fusion rules.**
   - **Primary: AND** (MoGe-2 ∧ DINO).
   - Descriptive only: OR, and the average of (MoGe-2 fraction, per-image min–max-normalized DINO score) thresholded at > 0.5.
4. **Ground truth (evaluation only).**
   - The projected Tango body silhouette under GT pose, rasterized at 518 px with the same crop transform and area-pooled to 74×74 (> 0.5).
   - The silhouette construction (convex hull of the 8 body points, or box ∪ top-plate union) is chosen **from overlays on synthetic images only**, before any mask metric is computed.
   - Antennas are not in the silhouette. Antenna-tip coverage (3 projected tip keypoints inside the mask) is reported separately.
5. **Metrics, per domain.**
   - IoU, precision and recall against GT.
   - Boundary F-score at a 2-px tolerance (74×74 grid).
   - Antenna-tip coverage.
   - Paired per-image differences with 95% bootstrap CIs (2,000 resamples, seed 0).
6. **Case table.** A mask is "right" if IoU ≥ 0.5. Cross-tabulate MoGe-2 right/wrong against DINO right/wrong, and report AND's IoU in each cell.
7. **Synthetic-val guard.** The primary rule must not make synthetic-val IoU worse than MoGe-2 alone by more than 0.02 (mean paired ΔIoU ≥ −0.02). If it does, the primary rule is reported as failing the guard. It is not swapped.
8. **Claim rule.**
   - "DINO helps MoGe-2" if AND − MoGe-2 IoU has a 95% CI above 0 on lightbox **and** sunlamp.
   - "MoGe-2 helps DINO" if AND − DINO IoU, **or** AND − DINO boundary-F, has a 95% CI above 0 on lightbox **and** sunlamp.
   - "Two-way" only if both hold. Otherwise report one-way or none.
9. **Feeding the pose test (EXP-016).** The AND mask is the only variant carried forward.

## Known biases
- GT-centred crops put the spacecraft in the middle, which favours the "border = background" sign rule. Automatic localization would be harder.
- The GT silhouette excludes antennas, so masks that include antennas lose some precision. This applies equally to all methods.

## Results
(to be filled from `outputs/experiments/EXP-015_mask_consensus/`)
