# EXP-033: Hard vs geodesic soft targets (Phase 3)

## Status
Running. Pre-registered on 2026-10-09, before any result.

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

## Results
(to be filled)
