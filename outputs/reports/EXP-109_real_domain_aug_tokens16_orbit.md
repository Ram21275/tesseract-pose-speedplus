# EXP-109: Real-domain generalisation: training-time augmentation (A), 16×16 DINOv3 tokens (B), in-plane orbit voting (C)

## Status
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

## Runs / Results / Decision
Not started.
