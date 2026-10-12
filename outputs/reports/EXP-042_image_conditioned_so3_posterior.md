# EXP-042: Image-conditioned SO(3) flow posterior (Phase 4)

## Status
Completed (2026-10-10). Pre-registered before any training result; the gate was rewritten to synthetic val only by DEC-006 before any result.

## Keep score
**4 / 5: Support.** The point estimate passes on synthetic val (−0.32°). Its 32 samples contain a near-GT rotation far more often than the tree's greedy path on the real domains (best-of-32 on lightbox 9.7° vs greedy 26.9°). That is the property EXP-043/044 need. The KDE-mode point estimate is worse on real data (descriptive), and the regions over-cover on synthetic val, so they are not calibrated.

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
- 6 head runs (`RUN-20261010-055931/061700` seed 0, `065119/070241` seed 1, `072215/073325` seed 2) and 1 fused evaluation with saved endpoint samples (`outputs/experiments/EXP-042_image_conditioned_so3_posterior/RUN-20261010-075303-seed0`, `samples/*_fused_seed0_fp16.npy` + `*_index.npy`, M = 32), all completed.

## Quantitative results
Mean over 3 seeds; source `outputs/experiments/EXP-042_image_conditioned_so3_posterior/RUN-20261010-075303-seed0/metrics/per_seed.csv`. Uniform (Haar) base, NFE 10, M = 32, point estimate = KDE mode (5° bandwidth). Lightbox and sunlamp are descriptive.

| method | synthetic_val mean ± std (leaf) | median | best-of-32 | any sample ≤ 10° | spread | coverage @0.5 / 0.8 / 0.9 | lightbox mean (leaf) / best-of-32 | sunlamp mean (leaf) / best-of-32 | ms/img |
|---|---|---|---|---|---|---|---|---|---|
| A: DINOv3 | **4.568 ± 0.128** (5.348) | 3.96 | 2.64 | 0.992 | 3.7° | 0.10 / 0.42 / 0.71 | 38.43 (37.08) / 17.3 | 42.50 (41.46) / 16.3 | 2.9 |
| B: MoGe-2 | 19.380 ± 0.620 (16.724) | 5.57 | 5.20 | 0.915 | 33.2° | 0.40 / 0.88 / 0.96 | 48.52 (43.69) / 12.2 | 65.82 (60.58) / 16.3 | 9.8 |
| **fused** (velocity average) | **5.219 ± 0.149** (5.540 ± 0.066) | **3.75** | 2.60 | 0.985 | 13.0° | 0.37 / 0.88 / 0.98 | 29.85 (26.88) / **9.7** | 37.61 (36.72) / **10.6** | 12.7 |

## Pre-registered verdicts
- **Point estimate: PASS.** 5.219° vs 5.540°, a gap of 0.321° > 0.149° (the larger seed std).
- **Secondary, multimodality (descriptive only, DEC-006):** on lightbox, any-sample-within-10° is 0.742 vs the tree's best-of-8 within 10° of 0.752. It is roughly equal and **does not exceed** the beam. However, best-of-32 has a mean of 9.7° vs the tree's best-of-8 of 17.5°: when the flow's samples miss, they miss by less.
- **Calibration: NOT calibrated.** Synthetic-val coverage is 0.876 at a nominal 0.8 and 0.977 at 0.9, more than 0.05 above nominal (over-dispersed after velocity averaging). The DINOv3 head alone under-covers (0.42 / 0.71).

## Interpretation
- **A global posterior from a uniform base is learnable from frozen features.** The DINOv3 head alone reaches 4.57° on synthetic val, the lowest of any system so far, with median 3.96°.
- **On real data the samples spread over several modes** (fused spread 41° on lightbox), and the true mode is usually among them. The KDE mode then picks the wrong one more often than the tree's greedy path, so the real-domain point error is +3° (descriptive). This is the EXP-104 symmetry problem in a different form. Mode **selection** remains the open issue, and EXP-043/044 (flow-induced Tesseract masses, distillation) and the proposed beam/template verifier (EXP-103, awaiting approval) target exactly that.
- **Velocity averaging** of a sharp head (DINOv3) and a broad head (MoGe-2) gives intermediate spread; it is a heuristic, not a product of experts.

## Qualitative results
`outputs/experiments/EXP-042_image_conditioned_so3_posterior/RUN-20261010-075303-seed0/figures/best_worst_{synthetic_val,lightbox,sunlamp}.png`

## Decision
Keep it. It provides the endpoint samples for EXP-043 (flow-induced masses) and EXP-044 (distillation). Do not use its KDE-mode point estimate on real data without a better mode selector.

## Next experiment
EXP-043: project the saved fused samples into per-level Tesseract masses (exact parent–child consistency) and measure GT-cell recall by depth.
