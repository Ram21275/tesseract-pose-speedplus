# EXP-025: Phase-2 basic fusion controls: single branches vs training-free posterior fusion

## Status
Completed. Pre-registered on 2026-10-09, before any result. This is the plan's Phase-2 EXP-020 ("basic controls"), renumbered under DEC-002.


## Keep score
**5 / 5: Strong support.** The pre-registered goal is met. On synthetic val, training-free PoE beats the strongest single branch by 1.8° (more than 2× the seed std). The effect is large on real images (descriptive): lightbox mean −13.6°, median −20.9°. Fusion is carried into Phase 3.

_Keep score rubric (0–5): 5 strong support: pre-registered goal met with a large effect, adopt. 4 support: goal met, carry forward. 3 partial: mixed or one-sided evidence, keep for further testing. 2 weak: goal not met but an informative side result. 1 negative: fails, reject, kept only as a recorded result. 0 inconclusive or not run._

## Research question
Does a training-free combination of the two branches' Tesseract posteriors beat the strongest single branch?
- Branch A: DINOv3-L `grid4`.
- Branch B: the consensus MoGe-2 `normals16~and`.

## Component under test
Fusion (score level)

## Track
Training-free. No new parameters; the existing probes are reused.

## Pre-registered protocol
- **Models:** the EXP-010 `dinov3_vitl16:grid4` and EXP-016 `moge2_vitl:normals16~and` checkpoints, seeds 0/1/2, paired by seed. Each uses its own train-split feature standardization, recomputed exactly as in training.
- **Fusion of the per-level conditional distributions** (root over 4 charts, then 8 children given the parent), applied inside the beam search:
  - **PoE:** log p = log p_A + log p_B (fixed product of experts, equal weights).
  - **AVG:** log p = log(½ p_A + ½ p_B) (equal posterior averaging, applied per level to the conditionals).
  - **Controls:** A alone, B alone.
- **Decoding:** greedy and beam 4, L5, no GT path.
- **Primary metric:** mean geodesic error per domain.
- **Claim:** fusion "beats the strongest branch" if its synthetic-val mean error is below A-alone by more than A's seed std (0.8°). Lightbox and sunlamp are descriptive.
- **Phase-2 gate:**
  - if neither fusion rule passes, DINOv3-L `grid4` alone is frozen as the representation for Phase 3;
  - the conditional EXP-026–029 (learned gates, cross-attention) are skipped, because training-free fusion failed and complementarity was weak (EXP-014: no geometry-specific complementarity).

## Runs
`RUN-20261009-005238-seed0`: completed under `guarded.sh 24`. The source checkpoints are listed in `source_runs.txt`. The single-branch rows reproduce EXP-010 / EXP-016 exactly (27.679° and 47.449°), which confirms the checkpoint loading and standardization.

## Results
Mean over 3 paired seeds. Source: `metrics/metrics.csv` and `metrics/per_seed.csv`.

| decoding | method | synthetic_val mean ± std / median | lightbox mean / median | sunlamp mean / median | acc@20 (synth / lightbox / sunlamp) |
|---|---|---|---|---|---|
| greedy | A: DINOv3-L `grid4` | 27.7 ± 0.8 / 17.2 | 76.1 / 62.2 | 79.1 / 70.3 | 0.585 / 0.145 / 0.109 |
| greedy | B: consensus `normals16~and` | 47.4 ± 1.0 / 18.9 | 77.1 / 68.7 | 87.1 / 88.3 | 0.521 / 0.241 / 0.160 |
| greedy | **PoE(A,B)** | **25.9 ± 0.5 / 14.7** | **62.5 / 41.3** | **71.4 / 56.5** | **0.656 / 0.264 / 0.169** |
| greedy | AVG(A,B) | 27.2 ± 0.7 / 15.3 | 65.5 / 46.4 | 73.7 / 61.2 | 0.638 / 0.246 / 0.168 |
| beam 4 | A | 28.1 ± 0.8 / 17.2 | 75.9 / 62.1 | 79.3 / 71.1 | 0.586 / 0.150 / 0.108 |
| beam 4 | **PoE(A,B)** | 25.9 ± 0.7 / 14.7 | 62.2 / 40.8 | 71.6 / 57.1 | 0.656 / 0.266 / 0.169 |
| beam 4 | AVG(A,B) | 33.0 ± 0.7 / 16.5 | 72.1 / 58.7 | 79.4 / 72.4 | 0.587 / 0.205 / 0.143 |

acc@10 (greedy): A 0.201 / 0.036 / 0.015; **PoE 0.284 / 0.082 / 0.034**.

## Pre-registered verdict
- **PoE beats the strongest branch: YES.** Synthetic-val mean is 25.9° vs 27.7°, a gain of 1.8° against A's 0.8° std.
- **AVG: NO** on synthetic val (−0.4°, below 1 std), though it does improve the real domains. With beam 4, AVG degrades.
- **Phase-2 gate:** fusion is **kept**. The fused representation for Phase 3 is the PoE of the DINOv3-L `grid4` probe and the consensus-normals probe. EXP-026/027 (training-free reliability weighting) follow, as in the plan. The learned-fusion EXP-028/029 run only if those fail to add anything.

## Interpretation
- **Combining predictions works where feature-level and mask-level fusion did not** (EXP-016–019). The two probes err on different images (EXP-014: errors weakly correlated). Multiplying their per-level distributions lets confident, mutually consistent levels dominate and suppresses each branch's isolated mistakes.
- **The gain is largest on real images, where each branch alone is weakest.** DINOv3 transfers coarse layout, while the DINO-fixed MoGe-2 normals transfer near-correct orientation (EXP-016 acc@20). PoE keeps both.
- **Earlier statements need qualifying.** "Fusion doesn't help overall" was true for the fusions tried before (masks into DINO, attention injection, spectral), but not for posterior fusion.

## Decision
Keep. PoE(DINOv3-L `grid4`, MoGe-2 `normals16~and`) is the Phase-2 fusion of record, pending EXP-026/027.
