# EXP-035: Tangent residual on the Tesseract leaf (Phase 3)

## Status
Running. Pre-registered on 2026-10-09, before any result.

## Research question
Does a continuous tangent residual (the plan's EXP-035) remove the L5 quantization floor (2.5° mean)?

## Component under test
Predictor

## Track
Trainable predictor; training-free PoE fusion (DEC-003)

## Pre-registered protocol
- **Variant:** `--residual` (MLP decoder, hard targets). The residual head predicts δ ∈ R³ from [trunk h, Fourier(leaf centre)], trained teacher-forced on the GT leaf (loss ‖(δ̂ − δ*)/τ_5‖², weight 1, δ* = Log(q_leaf⁻¹ ⊗ q_gt)). Inference: q = q_leaf ⊗ Exp(δ̂) on the *predicted* leaf. For the PoE, δ = the mean of both branches' residuals evaluated at the fused leaf. Both the discrete and the refined errors are reported.
- **Data and control:** full splits. The control is EXP-030 (MLP tree, hard targets, no residual), which gives PoE synthetic-val mean **6.24 ± 0.07°**.
- **Branches:** the variant is trained on **both** branches (DINOv3-L `grid4` and MoGe-2 `normals16~and`), 3 seeds each, 100 epochs. Otherwise the EXP-030 hyperparameters are unchanged (`scripts/train_probe_p3.py`, features stored as fp16 on the GPU).
- **Evaluation:** single branches and the equal-weight PoE (DEC-003) via `scripts/exp03x_fused_eval.py`, at greedy and beam 4.
- **Claim:** the variant "helps" if its PoE synthetic-val mean error is below the EXP-030 PoE by more than the larger seed std of the two. Lightbox and sunlamp are descriptive.
- **Phase-3 gate:** the configuration with the lowest PoE synthetic-val mean among the variants that help is frozen; if none helps, the EXP-030 configuration is kept.

## Results
(to be filled)
