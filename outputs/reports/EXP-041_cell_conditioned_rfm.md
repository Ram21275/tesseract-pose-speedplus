# EXP-041: Tesseract + separate cell-conditioned SO(3) flow refiner (Phase 4)

## Status
Running. Pre-registered on 2026-10-10, before any training result.

## Keep score
Pending.

## Research question
Does a probabilistic, cell-conditioned SO(3) flow, started around the predicted cell, refine better than the tangent residual (EXP-040)? Does it give useful sample spread and coverage? The cell classifier loss (the frozen EXP-036) and the flow loss are separate.

## Component under test
RFM (separate refiner)

## Track
Trainable predictor on a frozen Phase-3 tree; training-free PoE fusion

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
`--mode flow --base local`.
- **Base:** R₀ = q_c ⊗ Exp(n), with n ~ N(0, σ_b² I) and **σ_b = 3°** per axis. This gives a median norm of about 4.6°, matching the synthetic-val greedy-error median of 4.7°; synthetic only.
- **Path:** the geodesic R_τ = R₀ Exp(τa), with a = Log(R₀⁻¹ R_GT).
- **Loss:** ‖ω_θ(R_τ, τ, q_c | mem) − a‖², with 16 (τ, R₀) pairs per image per step.
- **Sampling:** Euler on the group, **NFE 10**, **M = 32** endpoint samples per image.
- **Point estimate:** the Gaussian-KDE mode of the samples (bandwidth 5°).
- **Also reported:**
  - best-of-M;
  - mean sample error;
  - any-sample-within 5/10/20°;
  - spread (mean distance to the mode);
  - ball-around-mode coverage at nominal 0.5/0.8/0.9/0.95;
  - latency.

## Additional pre-registered comparison
EXP-041 vs EXP-040, fused, on synthetic-val mean, under the same seed-std rule.

## Runs
Planned: 6 head runs and 1 fused evaluation (`scripts/run_phase4.sh`).

## Quantitative results
Pending.

## Decision
Pending.
