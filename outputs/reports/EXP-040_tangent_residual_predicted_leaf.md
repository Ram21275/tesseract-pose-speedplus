# EXP-040: Tesseract + tangent residual on the predicted leaf (Phase 4)

## Status
Running. Pre-registered on 2026-10-10, before any training result. The stage-0 diagnostic (no training) has run.

## Keep score
Pending.

## Research question
Can a one-step tangent correction q = q_c ⊗ Exp(δ) applied at the frozen predictor's **predicted** leaf reduce rotation error? This is the Phase-4 low-cost continuous baseline. EXP-035 trained only on the GT leaf, which cannot repair wrong-cell choices.

## Component under test
Predictor (continuous refinement)

## Track
Trainable predictor on a frozen Phase-3 tree; training-free PoE fusion

## Fixed setup (shared by EXP-040–042)
- **Frozen predictor:** the EXP-036 Tesseract Transformers (DINOv3-L `tokens8`, MoGe-2 `normals16~and`, soft targets), seed-matched, with requires_grad False (asserted) and in eval mode. Discrete decoding is the PoE greedy leaf (DEC-003). Only the new head trains, and it sees only its own branch's frozen encoder memory, so this is **not** learned fusion.
- **Data and split:** full splits; train on synthetic/train. Selection and gates follow **DEC-005**: synthetic val plus `lightbox_val` (a seeded 20% of lightbox). `lightbox_test` and sunlamp are untouched tests.
- **Training centre** `q_c` (`--centre mix`): per sample, either the GT leaf centre (probability ½), or the leaf containing GT ⊗ Exp(e), where e has a random axis and an angle drawn from the **synthetic-val** greedy-error distribution of the frozen PoE (stage-0 diagnostic, `RUN-20261010-015411-seed0/tables/poe_synthval_greedy_err.csv`; median 4.7°, p90 8.7°). **Why:** on synthetic **train** the frozen tree's greedy leaf is right 58% of the time (mean error 3.45°), but on val only 23% (5.64°). A head trained on the tree's own train-split predictions would see too-easy centres. At inference `q_c` is always the PoE greedy leaf.
- **Head** (`tfpose.so3flow`): a query token (Fourier features of the 3×3 rotation matrix, plus τ for flows, plus the cell centre for the local flow), then 2 pre-norm Transformer decoder layers (d 256, 4 heads, FF 512) cross-attending to the frozen memory, then Linear to R³.
- **Training:** AdamW 3e-4, weight decay 0.01, OneCycle, 30 epochs, batch 128 images, grad clip 1.0, bf16 autocast. Checkpoint selection is on synthetic-val mean point error (a fixed seeded 4,000-image subset; synthetic only). 3 seeds per branch. Run under `guarded.sh 32`, one job at a time.
- **Log-map cut locus:** pairs whose geodesic angle exceeds π − 10⁻³ are masked from the loss.
- **Fused evaluation** (`scripts/exp04x_fused_eval.py`, fixed rules): residual = the mean of the two branches' δ at the PoE leaf (the EXP-035 rule). Flow = velocity average (ω_A + ω_B)/2 at the same state. This is a heuristic, **not** an exact product of experts. Single-branch rows are reported too.
- **Control:** the frozen EXP-036 PoE greedy leaf of the same seeds, reported as `leaf_mean_deg` of the fused rows.
- **Pre-registered claim** (point estimate, per experiment): the variant **helps** if its fused mean error, averaged over 3 seeds, is (i) below the control on `lightbox_val` by more than the larger seed std, **and** (ii) not above the control on synthetic val by more than the larger seed std. Lightbox_test and sunlamp are reported but never used.

## Changed variable
`--mode residual`: δ̂ = head(mem, q_c). Loss: Huber (δ = 1) on (δ̂ − Log(q_c⁻¹ q_GT)) / τ₅, where τ₅ = 2.51° (the L5 cell size). The output layer is zero-initialised, so the head starts at the leaf centre.

## Stage-0 diagnostic (frozen EXP-036, seed 0, beam 8; no training)
Source: `outputs/experiments/EXP-040_tangent_residual_predicted_leaf/RUN-20261010-015411-seed0/metrics/metrics.csv`.

| PoE(A,B) | greedy mean / median | leaf acc | greedy err > 90° | top-8 leaf recall | best-of-8 mean | best-of-8 ≤ 10° | best-of-8 ≤ 20° |
|---|---|---|---|---|---|---|---|
| synthetic train (12k sample) | 3.45 / – | 0.578 | 0.000 | 0.979 | 2.48 | 1.000 | 1.000 |
| synthetic val | 5.64 / 4.68 | 0.234 | 0.002 | 0.693 | 3.28 | 0.991 | 0.998 |
| lightbox_val | 27.5 | 0.069 | 0.111 | 0.284 | 18.0 | 0.740 | 0.852 |
| lightbox_test | 27.4 | 0.066 | 0.108 | 0.282 | 17.4 | 0.755 | 0.860 |
| sunlamp | 37.2 | 0.042 | 0.161 | 0.191 | 23.3 | 0.616 | 0.779 |

**Reading:**
- The train/val gap motivates the `mix` centres.
- On the real domains a candidate within 20° is in the top 8 beams for 85% (lightbox) and 78% (sunlamp) of images, although greedy is that close far less often. Wrong-mode choices (often near-180° flips) are largely *recoverable in principle*. A one-step residual at the greedy leaf cannot do that; it only removes within-mode error.

## Runs
Planned: 6 head runs (2 branches × 3 seeds) and 1 fused evaluation. Chain: `scripts/run_phase4.sh`.

## Quantitative results
Pending.

## Decision
Pending.
