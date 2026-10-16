# EXP-111: DINOv3-only augmented tree + SO(3) flow head (candidate final architecture)

## Status
**Pre-registered 2026-10-16, before any run** (DEC-017, user: "yes pre-register it and run it before 108 it should be a quick experiment"). Runs right after the EXP-109 evaluation, before EXP-108.

## Question
With the augmentation-trained DINOv3 tree (EXP-109 arm A), does an EXP-042-style flow head on top improve the point estimate and give calibrated samples? How fast is the DINOv3-only pipeline (no MoGe-2)?

## Setup
- **Frozen:** DINOv3-L `tokens8` and the EXP-109 arm-A DINOv3 tree of the same seed (`--src-exp EXP-109_real_domain_aug_tokens16_orbit`).
- **Trained:** one flow head with the exact EXP-042 recipe: Haar base, geodesic flow matching, 16 pairs per image, 30 epochs, AdamW 3e-4, 2 cross-attention layers, NFE 10, M 32, KDE mode with a 5° kernel, checkpoint on a 4,000-image synthetic-val subset.
- **Training data:** the same augmented views as arm A (clean + `exp109_train_aug1/2`, uniform per sample). Labels are rotated by the geometry-derived "pre−" convention.
- Seeds 0–2. Synthetic train only; selection on synthetic val; lightbox / sunlamp descriptive.

## Metrics
- Per domain, for the flow KDE mode and for the same tree's greedy leaf (per image):
  - mean / median, acc@10, fraction > 90° / > 150°;
  - flip transitions with denominators (tree > 150° corrected to < 20°; tree < 20° broken to > 150°);
  - best-of-32 and coverage (from the trainer).
- **Timing, DINOv3 only** (batch 1 and 64, median of 5, cuda-synchronized): backbone, tree greedy, flow (1 head, NFE 10, M 32), KDE, totals. The concurrent GPU load is recorded.

## Criterion
The flow helps if its 3-seed synthetic-val mean is below the same trees' greedy mean by more than the larger seed std. Real domains are descriptive.

## Amendment 2026-10-16 (user, before any EXP-111 run): add the EXP-045 verifier arm
User: "in the same experiment can we test the 45 approach with the same aetting?"
- **Arm "verifier":** the EXP-045 / EXP-105-C3 design in the same DINOv3-only setting.
  - Candidates: the beam-8 leaves of the arm-A DINOv3 tree × {I, 180° about body x, y, z}, plus the 8 corners of each leaf cell (96).
  - Score: re-noise / return with the EXP-111 flow head of the same seed (τ 0.8, 16 draws, NFE 10).
  - Prediction: argmax. Code: `scripts/exp111_verifier.py`.
- **Criterion (added):** the verifier helps if its 3-seed synthetic-val mean is below the same tree's beam-8 top-1 mean by more than the larger seed std. Real domains descriptive, with flip transitions and denominators. Latency reported (tree included).
- The KDE-mode arm and its criterion are unchanged.

## Engineering note 2026-10-16 (before any successful run; no method change)
- The first launch (02:51) was killed by systemd-oomd (memory pressure).
- The retry (12:49) failed at start-up because the trainer's default `--centre mix` reads an error-distribution file that was not passed. With the Haar (uniform) base used here and in EXP-042, the centre is never used by the loss or the sampler, so `--centre gt` is passed. Nothing in the method changes.
- Running since 12:49:53 (`scripts/run_queue3.sh`).

## Runs / Results / Decision
Not started.
