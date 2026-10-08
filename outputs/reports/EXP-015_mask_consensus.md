# EXP-015: Two-way DINOv3 ↔ MoGe-2 foreground consensus (training-free)

## Status
Completed. Pre-registered on 2026-10-08; amendment A1 was added before any validation or test metric.


## Keep score
**4 / 5: Support.** Two-way by the pre-registered rule. DINO→MoGe-2 is large (leakage 0.52→0.12 lightbox, 0.57→0.12 sunlamp); MoGe-2→DINO is real but small on real images. The AND mask is carried forward.

_Keep score rubric (0–5): 5 strong support: pre-registered goal met with a large effect, adopt. 4 support: goal met, carry forward. 3 partial: mixed or one-sided evidence, keep for further testing. 2 weak: goal not met but an informative side result. 1 negative: fails, reject, kept only as a recorded result. 0 inconclusive or not run._

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

## Protocol amendment A1 (2026-10-08, before any validation or test mask metric)
**Why.** No exact Tango silhouette exists here: no aligned CAD model, and only 11 keypoints. The best-fitting keypoint outline (convex hull of the 8 body points, `hull8`, which beat `box+plate`, `two-quads` and `top-quad` against a bright-pixel proxy on 200 **synthetic-train** crops) failed the GT sanity check.
- On 300 **synthetic-train** crops, MoGe-2 covers it almost entirely (recall 0.99), but its median IoU with it is only 0.51 (expected ≥ about 0.7).
- Gamma-brightened overlays show `hull8` sometimes covers empty space.
- Existing on-disk "masks" are pseudo-labels from the same kind of hull, and exist only for synthetic images.
- IoU against `hull8` is therefore **not** a trustworthy primary metric.

**Amended primary metric: background leakage.**
- Envelope = convex hull of all 11 projected keypoints (8 body + 3 antenna tips), dilated by 3 cells on the 74×74 grid. Pixels outside the envelope are background beyond doubt.
- **Leakage** = the fraction of predicted-foreground pixels that fall outside the envelope. This directly measures the failure under study: background labelled as spacecraft.

**Amended retention metric: recall against `hull8`.** Its errors (over-covering) affect every method equally, so it is valid for comparing methods.

**Secondary, approximate only:** IoU, precision and boundary-F against `hull8`; antenna-tip coverage.

**Amended claim rule** (lightbox **and** sunlamp, paired bootstrap 95% CI):
- "DINO helps MoGe-2" if AND leakage < MoGe-2 leakage (CI excludes 0) **and** AND recall ≥ MoGe-2 recall − 0.05.
- "MoGe-2 helps DINO" if AND leakage < DINO leakage (CI excludes 0) **and** AND recall ≥ DINO recall − 0.05.
- "Two-way" only if both hold.

**Amended case table:** a mask is "wrong" if its leakage > 0.10 (more than 10% of its foreground is clear background).

**Synthetic-val guard:** AND recall must not fall more than 0.05 below MoGe-2's.

The fusion rules, DINO method and EXP-016 protocol are unchanged.

## Known biases
- GT-centred crops put the spacecraft in the middle, which favours the "border = background" sign rule. Automatic localization would be harder.
- The GT silhouette excludes antennas, so masks that include antennas lose some precision. This applies equally to all methods.

## Runs
| Run ID | Status | Notes |
|---|---|---|
| RUN-20261008-231024-seed0 | Failed (stopped deliberately) | About 10× slow from BLAS thread oversubscription; no results. `guarded.sh` now defaults to 1 BLAS thread. |
| RUN-20261008-234038-seed0 | Completed | About 6 min, CPU only, under `guarded.sh 16`. Wrote the AND masks to `outputs/shared_cache/consensus_masks__subset_v1/`. |

## Results
Source: `outputs/experiments/EXP-015_mask_consensus/RUN-20261008-234038-seed0/metrics/{metrics,paired_differences,case_table}.csv`. Means over images.

| domain | method | **leakage** ↓ | **recall** ↑ | IoU* | precision* | boundary-F* | antenna-tip coverage | mask area (frac of crop) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| synthetic_val | MoGe-2 | 0.279 | 0.989 | 0.581 | 0.590 | 0.487 | 0.908 | 0.575 |
| synthetic_val | DINO (PCA) | 0.116 | 0.962 | 0.615 | 0.619 | 0.463 | 0.716 | 0.372 |
| synthetic_val | **AND** | **0.070** | 0.952 | **0.775** | **0.788** | **0.775** | 0.669 | 0.294 |
| synthetic_val | OR | 0.314 | 0.999 | 0.432 | 0.433 | 0.176 | 0.955 | 0.653 |
| synthetic_val | avg | 0.278 | 0.992 | 0.572 | 0.578 | 0.485 | 0.914 | 0.578 |
| lightbox | MoGe-2 | 0.523 | 0.997 | 0.291 | 0.292 | 0.098 | 0.959 | 0.891 |
| lightbox | DINO (PCA) | 0.139 | 0.980 | 0.562 | 0.565 | 0.316 | 0.792 | 0.406 |
| lightbox | **AND** | **0.119** | 0.977 | **0.599** | **0.604** | **0.393** | 0.771 | 0.382 |
| lightbox | OR | 0.537 | 1.000 | 0.259 | 0.259 | 0.020 | 0.980 | 0.915 |
| lightbox | avg | 0.523 | 0.997 | 0.289 | 0.290 | 0.096 | 0.963 | 0.891 |
| sunlamp | MoGe-2 | 0.565 | 1.000 | 0.242 | 0.242 | 0.040 | 0.988 | 0.959 |
| sunlamp | DINO (PCA) | 0.121 | 0.994 | 0.586 | 0.589 | 0.304 | 0.637 | 0.415 |
| sunlamp | **AND** | **0.116** | 0.994 | **0.593** | **0.597** | **0.329** | 0.632 | 0.406 |
| sunlamp | OR | 0.567 | 1.000 | 0.238 | 0.238 | 0.010 | 0.993 | 0.968 |
| sunlamp | avg | 0.564 | 1.000 | 0.242 | 0.242 | 0.038 | 0.987 | 0.958 |

\* Against the approximate `hull8` silhouette (amendment A1), so relative comparisons only.

**Paired differences** (mean, 95% bootstrap CI):

| domain | AND − MoGe-2 leakage | AND − MoGe-2 recall | AND − DINO leakage | AND − DINO recall |
|---|---|---|---|---|
| synthetic_val | −0.209 [−0.222, −0.195] | −0.037 [−0.046, −0.029] | −0.046 [−0.051, −0.041] | −0.010 [−0.012, −0.009] |
| lightbox | **−0.404 [−0.418, −0.390]** | −0.020 [−0.027, −0.013] | **−0.019 [−0.024, −0.015]** | −0.003 [−0.004, −0.002] |
| sunlamp | **−0.449 [−0.459, −0.439]** | −0.006 [−0.009, −0.004] | **−0.005 [−0.007, −0.004]** | −0.000 [−0.000, −0.000] |

**Case table.** A mask is "wrong" if its leakage is > 0.10. Values are the fraction of images, with AND's leakage in brackets.

| domain | MoGe-2 ✓ DINO ✓ | MoGe-2 ✓ DINO ✗ (MoGe-2 can fix DINO) | MoGe-2 ✗ DINO ✓ (DINO can fix MoGe-2) | both ✗ |
|---|---|---|---|---|
| synthetic_val | 39.5% (0.000) | 9.8% (0.000) | 30.3% (0.008) | 20.4% (0.330) |
| lightbox | 6.1% (0.001) | 3.3% (0.004) | 53.4% (0.022) | 37.2% (0.288) |
| sunlamp | 0.2% (0.000) | 0.0% (–) | 68.9% (0.022) | 30.9% (0.325) |

**Figure:** `figures/mask_examples.jpg`. Rows: 2 synthetic_val, 2 lightbox, 2 sunlamp. Columns: crop, MoGe-2, DINO, AND; green = `hull8`.

## Pre-registered verdict
- **DINO helps MoGe-2: YES (large).** AND cuts MoGe-2's leakage from 0.52 to 0.12 on lightbox and from 0.57 to 0.12 on sunlamp (CIs far from 0). Recall drops only 0.020 / 0.006.
- **MoGe-2 helps DINO: YES under the pre-registered rule, but small on real images.** Leakage falls from 0.139 to 0.119 on lightbox and from 0.121 to 0.116 on sunlamp (CIs exclude 0), with no recall loss. On synthetic val the effect is large: leakage 0.116 → 0.070, and approximate boundary-F 0.46 → 0.78. Here MoGe-2's sharp mask trims DINO's blocky patch halo.
- **Two-way: YES by the pre-registered rule, but asymmetric.** DINO rescues MoGe-2 often: 53% of lightbox and 69% of sunlamp images fall in the "MoGe-2 wrong, DINO right" cell. MoGe-2 can only rescue DINO where MoGe-2 itself is right, which is rare on real images (9% of lightbox, 0.2% of sunlamp) but common on synthetic (49%).
- **Synthetic-val guard: passed.** AND recall is 0.037 below MoGe-2's (limit 0.05).

## Interpretation
- **The backgrounds that break MoGe-2 are textured ones** (Earth, clouds, stray light, glare). This holds on synthetic images with an Earth background too. On plain dark backgrounds MoGe-2's mask is tight and sharper than DINO's 16×16-patch mask.
- **The two sources fail in different ways:**
  - MoGe-2 over-segments textured backgrounds;
  - DINO is blocky at the boundary and occasionally picks the background as the "object" (when both fail, about 20–37% of images).
  - AND keeps only what both call object, so it fixes each source's specific failure as long as the other one is right there.
- **Cost:** AND loses antenna tips (coverage 0.91 → 0.67 on synthetic, 0.96 → 0.77 on lightbox), because DINO's coarse patches miss the thin antennas.
- **The average rule ≈ MoGe-2 and OR ≈ MoGe-2,** as predicted: MoGe-2's background mask of about 1.0 dominates them.

## Decision
Keep. The AND consensus mask is the training-free foreground carried to EXP-016 (pose) and is the baseline for EXP-017 (spectral fusion).

## Next experiment
EXP-016: does the consensus mask help pose? Then EXP-017 to EXP-020 (CASS options 1–4).
