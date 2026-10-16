# EXP-109: Real-domain generalisation: training-time augmentation (A), 16×16 DINOv3 tokens (B), in-plane orbit voting (C)

## Status
**Completed 2026-10-16. A PASS, C PASS (on both bases), B FAIL.** Augmentation-trained trees + in-plane orbit voting (A + C) is the best system so far: synthetic 3.87°, lightbox 12.3°, sunlamp 12.3° (EXP-036: 5.54 / 26.9 / 36.7). Score **5/5 Strong support**.

**Pre-registered 2026-10-15, before any code was run** (DEC-014, user: "add these A+B+C as an experiment … as soon as 050 finishes we should test transformer once with augmentations"). Order: EXP-050 → **EXP-109** → EXP-108 → EXP-107 jobs. Training on synthetic/train only; lightbox / sunlamp test-only (DEC-006).

## Question
Can the frozen-backbone Tesseract tree generalise better to real images when the training set is augmented, when finer image tokens are used, and when predictions are made consistent over in-plane rotations of the test crop? The target named by the user is 8–10° less real-domain error, i.e. making the large-flip errors rarer.

## Arms (EXP-036 recipe unless stated; both branches; equal-weight per-level PoE; seeds 0–2)
- **A, augmentation during training:**
  - The backbones are frozen, so augmentation is done at the image level and the features are re-extracted.
  - **Two augmented copies** of every synthetic-train crop. Each epoch, every image is drawn from {original, copy 1, copy 2} uniformly.
  - Photometric augmentations (labels unchanged), each with its own probability and range, all fixed now: brightness ×U(0.5, 1.5); contrast ×U(0.6, 1.4); gamma U(0.6, 1.6); Gaussian noise σ ~ U(0, 0.06); Gaussian blur σ ~ U(0, 1.5) (p 0.5); glare (1–3 saturating Gaussian blobs, p 0.3); random erasing (one rectangle, 5–20% of the area, p 0.3); JPEG quality 30–90 (p 0.5).
  - Geometric augmentation: **in-plane rotation** of the crop by θ ~ U(−180°, 180°) about the crop centre, sampled from the original image (no black corners), with the label rotated accordingly (convention fixed by the gate below).
  - A new Transformer tree (EXP-036 architecture) is trained from scratch on these features.
- **B, 16×16 tokens:** DINOv3 `tokens16` (256 × 1024, the unpooled last-layer grid already cached) instead of `tokens8`. MoGe-2 branch unchanged. A new Transformer, EXP-036 recipe. No augmentation, so B is isolated.
- **C, in-plane orbit voting at test time** (no training):
  - Each test crop is also seen rotated by 90°, 180° and 270°.
  - Candidates = the beam-8 leaves of every view, mapped back to the original frame (32).
  - Each candidate is scored by the sum over the 4 views of the PoE tree log-probability of the candidate mapped into that view (a product of experts across the orbit); prediction = argmax.
  - C is applied to the EXP-036 trees, and to arm A's trees if A is trained.
- **A+B:** run only if A and B each pass criterion 1, and only if 16×16 features for the augmented copies fit the memory guard (16×16 fp16 is 25 GB per copy). Otherwise it is reported as not run.

## Convention gate (before any augmented feature is used)
- On 512 synthetic-val crops, rotated by 90° and by 37°, the EXP-036 trees' predictions are mapped back with each of the 4 candidate conventions: q' = R_ray(±θ) ⊗ q, or q ⊗ R_ray(±θ). R_ray = rotation about the unit ray through the crop centre.
- The convention whose mapped-back error is closest to the unrotated error is used. It must be within 2× the unrotated mean (the tree has never seen rotated crops, so some degradation is expected). If none qualifies, the geometric augmentation and arm C are dropped and reported as such.
- This only checks a sign / ordering convention against the definition; it is not tuning.

## Selection and metrics
- **Selection metric (DEC-006-compliant): "synthetic val-aug"**, i.e. synthetic val with the same augmentation pipeline (one fixed copy, seed 1000), as a proxy for unseen imaging conditions, plus the clean synthetic val.
- Per domain:
  - mean / median / p95;
  - acc@5/10/20;
  - fraction > 90° and > 150°;
  - top-8 leaf recall;
  - latency (C costs 4× the backbone passes).

## Criteria
1. **A or B helps generalisation:** its 3-seed PoE mean on synthetic val-aug is below EXP-036's by more than the larger seed std, **and** its clean synthetic-val mean is not worse than EXP-036's (5.540 ± 0.063) by more than the larger seed std.
2. **C helps:** C on top of a predictor lowers that predictor's 3-seed mean on synthetic val-aug by more than the larger seed std, without being worse on clean synthetic val by more than it.
3. **Real domains:** lightbox / sunlamp are reported for every arm, whatever the direction. The user's 8–10° target is evaluated descriptively, never used for selection.

## Expectation (stated before results)
- A is the most likely to move the real numbers, because nothing in the current training resembles real lighting.
- B may help with fine asymmetric details (antenna), but costs 4× the tokens.
- C needs the flip to be inconsistent across the orbit. EXP-105 suggests the frozen features are ambiguous, so C's gain may be small.
- An 8–10° gain is not expected from any single arm.

## Resources
- Augmented extraction: 2 × 47,966 crops × 2 backbones, compact format, about 17 GB on hdd2, about 1–2 GPU-h.
- Orbit views: 3 × 21,525 test crops × 2 backbones, about 1 GPU-h.
- B's features: from the existing full cache (host-memory peak about 30 GB, guard 40).
- Trainings: 6 (A) + 3 (B, DINOv3 only) + C evaluations. Everything goes through the guarded queue, one job at a time.

## Amendment 2026-10-15 (external review; before any augmented extraction, training or evaluation)
1. **Label convention derived from geometry, not from a predictor.** The original gate (pick the convention whose mapped-back EXP-036 predictions are closest) is withdrawn as a selection rule.
   - **Derivation:** pose convention p_cam = R(q)·p_body + r (EXP-000). The crop warp rotates the content by θ (counter-clockwise on screen, cv2) about the crop centre c. With the image y axis down, a camera-frame rotation Rot(d, φ) about the unit ray d through c appears as a clockwise rotation by φ. So φ = −θ, giving R′ = Rot(d, −θ)·R and r′ = Rot(d, −θ)·r, i.e. **q′ = Exp(−θ·d) ⊗ q ("pre−")**.
   - **d** is computed from the *undistorted* crop centre (the distortion coefficients are now applied; the earlier helper ignored them).
   - **Verification** (`scripts/exp109_convention_geometry.py`, 512 synthetic-val rows × 4 angles):
     - The 11 Tango points, projected with the real distorted camera and pushed through the exact crop + rotation warp, match the projection of (q′, r′) to 0.34 px mean, 1.05 px p95, 4.75 px max (224-px crop). The other three transforms are off by 94–242 px.
     - The exact label error of the derivation (the PnP pose reproducing the warped points vs q′) is 0.06° mean, 0.20° p95, 1.14° max. This is the affine-warp vs homography/distortion approximation, well below the 2.5° leaf size.
     - The round trip q → q′ → q is within 0.004°.
   - A first run of that script (`RUN-20261015-134926`) used a pixel threshold (p95 < 1 px) chosen before measuring and reported "not verified" at 1.05 px. Its threshold was replaced, before the degree measurement was taken, by the label-error rule (p95 < 0.5°, round trip < 0.01°, pre− best by ≥ 10×), which passes (`RUN-20261015-134950`). Both runs are kept.
   - The predictor check (`scripts/exp109_gate.py`) still runs, **descriptively only**.
2. **Flip metrics added** relative to EXP-036 PoE greedy, per seed and per image:
   - n of initially > 150° images, and the fraction corrected to < 20°;
   - n of initially < 20° images, and the fraction broken to > 150°.
3. Selection rules and criteria are otherwise unchanged (synthetic val-aug + clean synthetic val; real domains descriptive).

## Runs
- Features (compact, hdd2): `exp109_train_aug1/2`, `exp109_valaug`, `exp109_valaug_r{90,180,270}`, `exp109_test3_r{90,180,270}`, gate crops.
- Convention: `RUN-20261015-134950` (derived "pre−", verified); predictor check `RUN-20261015-151441` (descriptive).
- Trainings:
  - A: DINOv3 `180909` / `190708…` / `192852…`; MoGe-2 `182205` / `192302…` / `194214…` (seeds 0–2, queue logs).
  - B: `194803`, `215520`, `235128`. B seed 0 first attempt `182752` killed by the memory watchdog (memmap RSS), fixed with pread.
- Evaluation: `RUN-20261016-014754-seed0`. Queue logs: `queue_attempt1_oomd.log`, `queue.log`.

## Results (3-seed mean ± std; flip transitions per image vs EXP-036 PoE greedy, with denominators)
| system | synth val (clean) | **synth val-aug (selection)** | lightbox | sunlamp |
|---|---|---|---|---|
| EXP-036 PoE greedy | 5.54 ± 0.07 | 30.84 ± 0.40 | 26.91 ± 0.78 | 36.66 ± 0.93 |
| **A** PoE greedy | 4.86 ± 0.03 | 9.75 ± 0.07 | 17.35 ± 0.32 | 20.16 ± 0.58 |
| B tokens16 + EXP-036 MoGe-2 | 5.56 ± 0.03 | 30.51 ± 0.20 | 27.01 ± 0.38 | 35.97 ± 0.96 |
| C orbit on EXP-036 | 4.62 ± 0.04 | 19.84 ± 0.35 | 20.25 ± 0.60 | 24.83 ± 0.87 |
| **C orbit on A** | **3.87 ± 0.03** | **5.82 ± 0.09** | **12.30 ± 0.46** | **12.28 ± 0.40** |

| system | lightbox > 150° frac. | corrected > 150° → < 20° (of 401) | broken < 20° → > 150° | sunlamp > 150° | corrected (of 206) | broken |
|---|---|---|---|---|---|---|
| EXP-036 | 0.059 | – | – | 0.074 | – | – |
| A | 0.042 | 0.51 | 0.010 | 0.033 | 0.59 | 0.010 |
| B | 0.057 | 0.14 | 0.011 | 0.073 | 0.13 | 0.012 |
| C on EXP-036 | 0.048 | 0.36 | 0.008 | 0.043 | 0.46 | 0.009 |
| **C on A** | **0.030** | **0.62** | **0.004** | **0.017** | **0.80** | **0.004** |

Single branches (training-run evaluation, greedy):
- DINOv3 + augmentation: 4.84 / 22.5 / 22.0, against 5.34 / 37.1 / 41.5 without.
- MoGe-2 + augmentation: 17.6 / 40.6 / 59.5, against 16.7 / 43.7 / 60.6.
- B DINOv3 tokens16: 5.37 / 37.9 / 39.7.

## Criteria
1. **A: PASS.** Val-aug 9.75 vs 30.84 (−21.1°, std ≤ 0.40); clean 4.86 vs 5.54, not worse (better).
2. **B: FAIL.** Val-aug 30.51 vs 30.84 (−0.33° against a std of 0.40); clean 5.56 vs 5.54.
3. **C: PASS** on both bases.
   - On A: val-aug 5.82 vs 9.75; clean 3.87 vs 4.86.
   - On EXP-036: 19.84 vs 30.84; clean 4.62 vs 5.54.
4. **Real (descriptive):** A + C lowers lightbox by 14.6° and sunlamp by 24.4° relative to EXP-036. It corrects 62% / 80% of the > 150° failures and breaks 0.4% of the good predictions. The user's 8–10° target is exceeded, and not by selection on real data.

## Interpretation
- **Augmentation** (photometric + in-plane rotation, with geometry-correct labels) is what moves real-domain error. It acts mainly on the DINOv3 branch; MoGe-2 changes little.
- **Orbit voting** turns the rotation augmentation into a test-time consistency check. A 180° flip is rarely the most probable answer in all four rotated views, while the true pose is.
- **Finer tokens (B)** add nothing at 4× the cost.
- **Cost:** C needs 4 backbone passes per image (estimated 120–150 ms at batch 1 with both backbones; to be timed). EXP-111 tests whether a DINOv3-only variant is close enough to be the fast version.

## Decision (proposed for the recipe lock; the user decides)
Adopt **A + C** (augmentation-trained trees, equal-weight PoE, in-plane orbit PoE voting) as the rotation recipe. B is rejected.
