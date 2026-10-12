# EXP-043: Flow-induced Tesseract masses from the same endpoint samples (Phase 4)

## Status
Completed (2026-10-13). Pre-registered before any result. No training: it re-samples the frozen EXP-042 flows and the frozen EXP-036 trees.

## Keep score
**3 / 5: Partial.** The sanity criterion holds exactly. Flow-mass discrete decoding is worse than the tree (criterion 2 FAIL). On synthetic val, though, the flow picks the correct coarse cells (levels 2–5) more often than the tree's greedy path, which is the signal EXP-044 needs. On the real domains the tree is better at every level (descriptive).

## Research question
Do the endpoint samples of the image-conditioned SO(3) flow (EXP-042) define per-level Tesseract masses that:
- are exactly parent–child consistent;
- place more probability on the ground-truth cell than the tree does, at some depth;
- give a better discrete decoding than the tree's greedy path?

## Component under test
Posterior (flow-induced Tesseract masses); input to EXP-044.

## Track
Analysis on frozen predictors (EXP-042 flow heads, EXP-036 trees); training-free PoE / velocity-average fusion.

## Fixed setup
- **Samples:** for each seed 0/1/2, the EXP-042 fused flow (both branch heads, velocity average) is re-sampled with M = 32 endpoints per image, from a Haar base, with NFE 10 (Euler on the group). Run: `exp04x_fused_eval.py --exp EXP-042_image_conditioned_so3_posterior --run-exp EXP-043_flow_induced_tesseract_masses --fused-only --save-samples --save-all-seeds`. EXP-042 itself saved samples for seed 0 only.
- **Groups:** synthetic val (decision) and lightbox / sunlamp (descriptive, DEC-006).
- **Masses:** μ_l(c) = (1/M) · #{m : prefix_l(R_m) = c}. The **same** 32 samples give every level l = 0 (root chart) … 5 (leaf), with the q ~ −q canonical encoding of `tesseract.encode`. Boundary fluxes are not estimated (plan, ground rules).
- **Tree comparison:** the frozen EXP-036 PoE tree (same seed). Its mass on the GT prefix at level l is the product of the PoE conditional probabilities along the GT path (teacher-forced, read-only). Its top-1 at level l is "greedy prefix = GT prefix".

## Reported metrics
- **Per level and domain** (3-seed mean):
  - flow vs tree mass on the GT cell;
  - flow recall (GT cell has mass > 0);
  - flow top-1 (most-massive cell = GT) vs tree greedy top-1;
  - flow occupancy (distinct cells with mass);
  - parent–child consistency error (max |μ(parent) − Σ μ(children)|, checked exhaustively on about 200 images per group);
  - mass self-consistency vs M (mean TV distance between the masses from the first m ∈ {4, 8, 16} samples and from all 32);
  - seam subset (GT chart margin < 0.05): flow vs tree root top-1.
- **Point estimates** (synthetic-val decision, real descriptive):
  - tree greedy leaf;
  - flow KDE mode (EXP-042's estimator);
  - **flow-mass greedy descent:** the most-massive root, then the most-massive child among the samples in the chosen prefix, down to the leaf centre; ties go to the smallest id.

## Pre-registered criteria
1. **Sanity (required):** the parent–child consistency error must be exactly 0. A non-zero value means an encoding bug, and EXP-044 does not start.
2. **Discrete decoding:** flow-mass greedy descent "helps" if its synthetic-val mean is below the tree greedy mean by more than the larger seed std.
3. **Phase-4 path:** EXP-044 (distilling these masses into the tree) proceeds per the plan if criterion 1 holds, whatever the outcome of criterion 2. Criterion 2 and the per-level mass comparisons are reported as evidence about whether the flow carries information the tree lacks.

## Runs
- **Sample stage:** `outputs/experiments/EXP-043_flow_induced_tesseract_masses/RUN-20261013-004254-seed0` (fused EXP-042 flow, seeds 0–2, M = 32, NFE 10, Haar base; `samples/*_fused_seed{0,1,2}_fp16.npy`).
  - **First attempt:** failed with CUDA OOM (log `samples_stage_oom_attempt1.log`), because other jobs held 31 GB of the shared GPU.
  - **Rerun:** succeeded in low-memory mode (`--feat-cpu --queries 1024`), with numerically the same procedure.
- **Analysis:** `outputs/experiments/EXP-043_flow_induced_tesseract_masses/RUN-20261013-010411-seed0` (`scripts/exp043_flow_masses.py`). Logs: `samples_stage.log`, `analysis_stage.log`.

## Results: per level (3-seed mean; source `outputs/experiments/EXP-043_flow_induced_tesseract_masses/RUN-20261013-010411-seed0/metrics/per_level.csv`)
Level 0 = root chart (4 cells), level 5 = leaf (131,072 cells). Lightbox and sunlamp are descriptive.

| domain | level | flow mass on GT | tree mass on GT | flow recall (mass > 0) | flow top-1 | tree greedy top-1 | flow occupancy (cells) | TV(m=8 vs 32) |
|---|---|---|---|---|---|---|---|---|
| synthetic_val | 0 | 0.899 | 0.940 | 0.999 | 0.966 | 0.963 | 2.0 | 0.06 |
| synthetic_val | 1 | 0.792 | 0.805 | 0.997 | 0.913 | 0.908 | 3.4 | 0.11 |
| synthetic_val | 2 | **0.672** | 0.652 | 0.986 | **0.832** | 0.787 | 5.4 | 0.17 |
| synthetic_val | 3 | **0.520** | 0.506 | 0.947 | **0.700** | 0.661 | 8.1 | 0.25 |
| synthetic_val | 4 | 0.338 | 0.337 | 0.843 | **0.498** | 0.470 | 11.4 | 0.33 |
| synthetic_val | 5 | 0.154 | 0.161 | 0.626 | **0.250** | 0.237 | 15.2 | 0.42 |
| lightbox | 0 / 2 / 5 | 0.686 / 0.332 / 0.024 | **0.809 / 0.467 / 0.051** | 0.995 / 0.863 / 0.225 | 0.813 / 0.520 / 0.049 | **0.832 / 0.571 / 0.074** | 2.9 / 11.9 / 25.8 | 0.12 / 0.36 / 0.64 |
| sunlamp | 0 / 2 / 5 | 0.621 / 0.239 / 0.010 | **0.759 / 0.381 / 0.031** | 0.996 / 0.839 / 0.136 | 0.765 / 0.427 / 0.023 | **0.783 / 0.469 / 0.044** | 3.2 / 14.3 / 28.6 | 0.14 / 0.42 / 0.70 |

- **Parent–child consistency error:** **0.0** at every level, domain and seed (criterion 1 PASS).
- **Seam subset** (GT chart margin < 0.05; 1,380 synthetic-val images), root top-1: flow 0.761 vs tree 0.745 on synthetic val; lightbox 0.538 vs 0.545; sunlamp 0.435 vs 0.475.

## Results: point estimates (source `outputs/experiments/EXP-043_flow_induced_tesseract_masses/RUN-20261013-010411-seed0/metrics/point_estimates.csv`)

| estimate | synthetic_val mean ± std / median | lightbox mean / median | sunlamp mean / median |
|---|---|---|---|
| tree greedy (PoE, EXP-036) | 5.541 ± 0.063 / 4.62 | **26.9 / 9.0** | **36.7 / 13.0** |
| flow KDE mode (EXP-042 estimator) | **5.193 ± 0.145 / 3.77** | 30.1 / 10.8 | 37.2 / 15.1 |
| flow-mass greedy descent | 6.210 ± 0.127 / 4.58 | 31.5 / 12.2 | 39.7 / 18.5 |

## Pre-registered verdicts
1. **Sanity: PASS.** Masses from the same samples are exactly parent–child consistent, so EXP-044 may start.
2. **Discrete decoding: FAIL.** Flow-mass greedy descent gives 6.210° vs the tree's 5.541° on synthetic val, which is worse.

## Interpretation
- **Why counting fails while the KDE mode works.** With M = 32 samples spread over about 15 leaf cells on synthetic val (26–29 on real), leaf-level counts are noisy: the TV distance between masses from 16 and from 32 samples is still 0.26 at the leaf. Greedy descent then breaks near-ties arbitrarily and returns a cell centre. The continuous KDE mode uses the same samples without discretizing and is better.
- **Synthetic val: the flow carries routing information the tree lacks.** At levels 2–5 its most-massive cell is the GT cell more often (level 2: 0.832 vs 0.787; level 3: 0.700 vs 0.661), and it does better at chart seams. This is exactly what EXP-044 is meant to transfer into the tree.
- **Real domains (descriptive): the tree is better at every level.** The flow's samples spread over more cells (EXP-042 spread about 41° on lightbox) and put less mass on the GT cell. Distilling flow masses could therefore help synthetic routing while hurting real-domain routing; EXP-044 must report both.

## Decision
Keep the flow-induced masses as the EXP-044 distillation target (criterion 1 holds). Do not use flow-mass counting as a decoder at M = 32.

## Next experiment
EXP-044: train the tree with GT-path loss + β · KL(stopgrad(flow masses) ‖ tree level marginals), comparing β = 0 with a small validated set; selection on synthetic val (DEC-006).
