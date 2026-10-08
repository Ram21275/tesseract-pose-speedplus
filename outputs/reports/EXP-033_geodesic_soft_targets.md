# EXP-033: Hard vs geodesic soft targets (Phase 3)

## Status
Completed. Pre-registered on 2026-10-09, before any result.


## Keep score
**4 / 5: Support.** It passes: PoE synthetic val 6.068° vs 6.242° (−0.17°, more than 2× std). It is the best Phase-3 variant and is frozen by the gate (DEC-004). One caveat: under soft targets, DINOv3 *alone* (5.69°) beats the PoE on synthetic val. Fusion still wins clearly on real domains, which raises a selection-policy question (see DEC-004).

_Keep score rubric (0–5): 5 strong support: pre-registered goal met with a large effect, adopt. 4 support: goal met, carry forward. 3 partial: mixed or one-sided evidence, keep for further testing. 2 weak: goal not met but an informative side result. 1 negative: fails, reject, kept only as a recorded result. 0 inconclusive or not run._

## Research question
Do geodesic soft targets (the plan's EXP-033) beat hard path labels?

## Component under test
Predictor

## Track
Trainable predictor; training-free PoE fusion (DEC-003)

## Pre-registered protocol
- **Variant:** `--targets soft` (MLP decoder, no residual). Each level's target over the 4 charts or the 8 children of the GT parent is softmax(−d²/τ_l²), where d is the SO(3) distance from the GT rotation to each candidate centre. τ_l = the EXP-002 mean own-cell error per level (39.3, 39.3, 20.0, 10.0, 5.0, 2.5°), fixed a priori.
- **Data and control:** full splits. The control is EXP-030 (MLP tree, hard targets, no residual), which gives PoE synthetic-val mean **6.24 ± 0.07°**.
- **Branches:** the variant is trained on **both** branches (DINOv3-L `grid4` and MoGe-2 `normals16~and`), 3 seeds each, 100 epochs. Otherwise the EXP-030 hyperparameters are unchanged (`scripts/train_probe_p3.py`, features stored as fp16 on the GPU).
- **Evaluation:** single branches and the equal-weight PoE (DEC-003) via `scripts/exp03x_fused_eval.py`, at greedy and beam 4.
- **Claim:** the variant "helps" if its PoE synthetic-val mean error is below the EXP-030 PoE by more than the larger seed std of the two. Lightbox and sunlamp are descriptive.
- **Phase-3 gate:** the configuration with the lowest PoE synthetic-val mean among the variants that help is frozen; if none helps, the EXP-030 configuration is kept.

## Runs
- 6 training runs (2 branches × 3 seeds) under `guarded.sh 32`, all completed. Chain log: `outputs/experiments/EXP-034_beam_decoding/phase3_chain.log`.
- Fused evaluation: `RUN-20261009-025557-seed0`.

## Results
Greedy decoding, mean over 3 seeds. Source: `RUN-20261009-025557-seed0/metrics/metrics.csv`. Lightbox and sunlamp are descriptive.

| method | synthetic_val mean ± std / median | lightbox mean / median | sunlamp mean / median | acc@10 (synth / lightbox / sunlamp) | acc@20 (synth / lightbox / sunlamp) |
|---|---|---|---|---|---|
| A: DINOv3-L `grid4` | 5.691 ± 0.013 / 4.92 | 37.0 / 13.0 | 48.1 / 23.6 | 0.941 / 0.377 / 0.193 | 0.994 / 0.654 / 0.443 |
| B: MoGe-2 `normals16~and` | 16.478 ± 0.052 / 5.97 | 41.9 / 11.1 | 58.5 / 19.6 | 0.791 / 0.457 / 0.294 | 0.891 / 0.651 / 0.504 |
| **PoE(A,B)** | 6.068 ± 0.067 / 4.81 | 28.2 / 9.8 | 42.1 / 16.6 | 0.926 / 0.508 / 0.292 | 0.986 / 0.742 / 0.565 |
| EXP-030 control, PoE (MLP, hard, greedy) | 6.242 ± 0.072 / 5.00 | 31.1 / 10.4 | 43.7 / 17.0 | 0.932 / 0.479 / 0.290 | 0.989 / 0.729 / 0.553 |

## Pre-registered verdict
**PASS.** PoE synthetic val 6.068 ± 0.067° vs 6.242 ± 0.072°.

## Interpretation
- **Soft targets help the deep levels:** neighbouring cells get partial credit, so the probe learns the metric structure across cell seams.
  - DINOv3 alone improves the most: 6.64° → **5.69 ± 0.01°**.
  - Real domains improve (descriptive): PoE lightbox 31.1° → 28.2°, sunlamp 43.7° → 42.1°.
- **Caveat:** with soft targets, DINOv3 alone beats the PoE on the selection split (5.69° vs 6.07°), while PoE remains far better on lightbox (28.2° vs 37.0°) and sunlamp (42.1° vs 48.1°).
  - Under DEC-000, only synthetic val may select, and it now prefers DINOv3 alone.
  - The real-domain evidence for fusion cannot be used to choose without a recorded real-domain validation split.
  - Flagged to the user in DEC-004.

## Decision
Frozen by the Phase-3 gate (DEC-004): MLP tree with geodesic soft targets and greedy decoding.
