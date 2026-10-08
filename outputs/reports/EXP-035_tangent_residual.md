# EXP-035: Tangent residual on the Tesseract leaf (Phase 3)

## Status
Completed. Pre-registered on 2026-10-09, before any result.


## Keep score
**1 / 5: Negative.** The residual moves the fused error by only 0.05° (6.27° discrete → 6.21° refined); compared with the control it gains 0.03°, within std. The L5 floor is not the binding error: the predicted leaf is correct only about 18% of the time, and a residual trained on the GT leaf cannot repair wrong-cell choices.

_Keep score rubric (0–5): 5 strong support: pre-registered goal met with a large effect, adopt. 4 support: goal met, carry forward. 3 partial: mixed or one-sided evidence, keep for further testing. 2 weak: goal not met but an informative side result. 1 negative: fails, reject, kept only as a recorded result. 0 inconclusive or not run._

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

## Runs
- 6 training runs (2 branches × 3 seeds) under `guarded.sh 32`, all completed. Chain log: `outputs/experiments/EXP-034_beam_decoding/phase3_chain.log`.
- Fused evaluation: `RUN-20261009-030321-seed0`.

## Results
Greedy decoding, mean over 3 seeds. Source: `RUN-20261009-030321-seed0/metrics/metrics.csv`. Lightbox and sunlamp are descriptive.

| method | synthetic_val mean ± std / median | lightbox mean / median | sunlamp mean / median | acc@10 (synth / lightbox / sunlamp) | acc@20 (synth / lightbox / sunlamp) | discrete synth mean |
|---|---|---|---|---|---|---|
| A: DINOv3-L `grid4` | 6.579 ± 0.037 / 5.37 | 42.5 / 14.4 | 52.7 / 25.7 | 0.914 / 0.337 / 0.174 | 0.989 / 0.611 / 0.417 | 6.586 |
| B: MoGe-2 `normals16~and` | 17.904 ± 0.122 / 6.25 | 45.5 / 12.2 | 62.8 / 23.9 | 0.769 / 0.423 / 0.269 | 0.881 / 0.620 / 0.465 | 17.971 |
| **PoE(A,B)** | 6.214 ± 0.054 / 4.98 | 32.2 / 10.4 | 43.2 / 16.8 | 0.932 / 0.482 / 0.295 | 0.988 / 0.724 / 0.554 | 6.269 |
| EXP-030 control, PoE (MLP, hard, greedy) | 6.242 ± 0.072 / 5.00 | 31.1 / 10.4 | 43.7 / 17.0 | 0.932 / 0.479 / 0.290 | 0.989 / 0.729 / 0.553 | — |

## Pre-registered verdict
**FAIL.** PoE synthetic val 6.214 ± 0.054° (refined) vs 6.242° (control); the refinement over its own discrete leaf is only 0.055°.

## Interpretation
- **Most error comes from selecting the wrong (usually neighbouring) cell,** not from quantization inside the right cell. Full L5 path accuracy is about 0.18 (EXP-030).
- **The residual is trained teacher-forced on the GT leaf.** At inference it is applied to the predicted leaf, so it only helps in the ~18% of cases where that leaf is correct (acc@1 stays below 1%).
- **A residual trained on the *predicted* leaf** (self-consistent) is the natural follow-up; it was not pre-registered here.

## Decision
Reject the GT-leaf residual. If continuous refinement is pursued (Phase 4), train it on predicted leaves or use the flow refiner.
