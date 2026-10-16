# EXP-111: DINOv3-only augmented tree + SO(3) flow head (candidate final architecture)

## Status
**Completed 2026-10-16.** **KDE arm: criterion PASS on synthetic val** (4.10 ± 0.04 vs its tree's 4.84 ± 0.05), **but worse on both real domains** (lightbox 25.16 vs 22.55; sunlamp 24.39 vs 22.07). **Verifier arm: PASS on synthetic val** (4.42 vs beam-8 top-1 5.26), about equal on real (22.93 / 21.98 vs 22.97 / 22.37). Neither beats orbit voting on the same tree (EXP-112 A-DINO + orbit: 4.21 / 17.40 / 15.27). Score **2/5 Weak**.

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

## Runs
Flow heads: `RUN-20261016-125000…` (seed 0), `RUN-20261016-131622-seed1`, `RUN-20261016-222956-seed2`. Summary / timing: `RUN-20261016-224135-seed0`. Verifier: `RUN-20261016-224147-seed0`.

## Results (3-seed mean ± std, °)
| system | synth val | lightbox | sunlamp | > 150° (LB / SL) |
|---|---|---|---|---|
| tree greedy (EXP-109 A DINOv3) | 4.84 ± 0.05 | 22.55 ± 0.73 | 22.07 ± 0.73 | 0.060 / 0.033 |
| **tree + flow KDE** | **4.10 ± 0.04** | 25.16 ± 0.60 | 24.39 ± 0.64 | 0.065 / 0.028 |
| tree beam-8 top-1 | 5.26 ± 0.04 | 22.97 | 22.37 | 0.060 / 0.033 |
| tree + flow verifier (EXP-045 design) | 4.42 ± 0.03 | 22.93 ± 1.05 | 21.98 ± 0.69 | 0.063 / 0.031 |
| *reference: A-DINO + orbit (EXP-112)* | *4.21* | *17.40* | *15.27* | *0.048 / 0.020* |

**Timing (batch 1 / 64, ms per image; shared GPU):**
- DINOv3 6.2 / 2.8; tree greedy 6.9 / 0.12; flow NFE 10 × M 32 8.5 / 1.2.
- Totals: tree only 13.1 / 2.9; tree + flow 21.6 / 4.1.

## Decision
- The flow and the verifier help on synthetic data but not on real images. Orbit voting on the same tree is better on every domain.
- Not adopted. The flow remains a supporting / appendix result (density, uncertainty).
