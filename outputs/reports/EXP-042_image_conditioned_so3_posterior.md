# EXP-042: Image-conditioned SO(3) flow posterior (Phase 4)

## Status
Running. Pre-registered on 2026-10-10, before any training result.

## Keep score
Pending.

## Research question
Can a global, image-conditioned SO(3) flow from a **uniform** base represent the posterior well enough to (a) give a competitive point estimate, (b) place samples on the correct mode when the tree's greedy choice is wrong, and (c) produce calibrated credible regions?

## Component under test
RFM (global posterior); input to EXP-043/044

## Track
Trainable predictor on frozen features; training-free fusion of velocities

## Fixed setup (shared by EXP-040–042)
- **Frozen predictor:** the EXP-036 Tesseract Transformers (DINOv3-L `tokens8`, MoGe-2 `normals16~and`, soft targets), seed-matched, with requires_grad False (asserted) and in eval mode. Discrete decoding is the PoE greedy leaf (DEC-003). Only the new head trains, and it sees only its own branch's frozen encoder memory, so this is **not** learned fusion.
- **Data and split:** full splits. Train on synthetic/train **only**. Checkpoint selection and the gate use synthetic val **only** (DEC-000, restored by DEC-006). Lightbox and sunlamp are test-only and descriptive. Per-run files contain descriptive `lightbox_val`/`lightbox_test` rows from the withdrawn DEC-005 split; fused evaluations report the whole of lightbox.
- **Training centre** `q_c` (`--centre mix`): per sample, either the GT leaf centre (probability ½), or the leaf containing GT ⊗ Exp(e), where e has a random axis and an angle drawn from the **synthetic-val** greedy-error distribution of the frozen PoE (stage-0 diagnostic, `RUN-20261010-015411-seed0/tables/poe_synthval_greedy_err.csv`; median 4.7°, p90 8.7°). **Why:** on synthetic **train** the frozen tree's greedy leaf is right 58% of the time (mean error 3.45°), but on val only 23% (5.64°). A head trained on the tree's own train-split predictions would see too-easy centres. At inference `q_c` is always the PoE greedy leaf.
- **Head** (`tfpose.so3flow`): a query token (Fourier features of the 3×3 rotation matrix, plus τ for flows, plus the cell centre for the local flow), then 2 pre-norm Transformer decoder layers (d 256, 4 heads, FF 512) cross-attending to the frozen memory, then Linear to R³.
- **Training:** AdamW 3e-4, weight decay 0.01, OneCycle, 30 epochs, batch 128 images, grad clip 1.0, bf16 autocast. Checkpoint selection is on synthetic-val mean point error (a fixed seeded 4,000-image subset; synthetic only). 3 seeds per branch. Run under `guarded.sh 32`, one job at a time.
- **Log-map cut locus:** pairs whose geodesic angle exceeds π − 10⁻³ are masked from the loss.
- **Fused evaluation** (`scripts/exp04x_fused_eval.py`, fixed rules): residual = the mean of the two branches' δ at the PoE leaf (the EXP-035 rule). Flow = velocity average (ω_A + ω_B)/2 at the same state. This is a heuristic, **not** an exact product of experts. Single-branch rows are reported too.
- **Control:** the frozen EXP-036 PoE greedy leaf of the same seeds, reported as `leaf_mean_deg` of the fused rows.
- **Pre-registered claim** (point estimate; rewritten by DEC-006 before any result): the variant **helps** if its fused synthetic-val mean error, averaged over 3 seeds, is below the control's by more than the larger seed std. Lightbox and sunlamp are reported but never used.

## Changed variable
`--mode flow --base uniform`.
- **Base and conditioning:** R₀ ~ Haar(SO(3)). The head does **not** see the cell; it is image-conditioned only.
- **Why uniform:** the stage-0 diagnostic (EXP-040 report) shows that real-domain errors include near-180° modes that a local Gaussian base (σ ≤ 1 rad, e.g. LiePose) cannot reach.
- **Training, sampling and statistics:** as in EXP-041 (16 pairs per image, NFE 10, M = 32, KDE mode, same reported statistics).
- **EXP-043 input:** fused seed-0 endpoint samples are saved for synthetic val, lightbox and sunlamp (`--save-samples`).

## Additional pre-registered secondary claim (multimodality)
Descriptive only (DEC-006): on lightbox, any-sample-within-10° of the fused flow is compared with the tree's best-of-8 within 10° (0.752, stage 0); this cannot be used for selection. Coverage is called "calibrated" only if the measured coverage on synthetic val is within ±0.05 of nominal at 0.8 and 0.9.

## Runs
Planned: 6 head runs and 1 fused evaluation (`scripts/run_phase4.sh`).

## Quantitative results
Pending.

## Decision
Pending.
