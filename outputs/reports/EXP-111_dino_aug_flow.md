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

## Runs / Results / Decision
Not started.
