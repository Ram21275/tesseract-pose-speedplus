# EXP-031: GRU path decoder (Phase 3)

## Status
Completed. Pre-registered on 2026-10-09, before any result.


## Keep score
**3 / 5: Partial.** The PoE gain on synthetic val (−0.093°) falls just short of the pre-registered threshold (0.098°), so it does not count. DINOv3 alone does improve beyond noise (6.29° vs 6.64°), and real domains improve clearly (descriptive: lightbox median 8.98° vs 10.40°, sunlamp mean 40.6° vs 43.7°). Keep for a combined test.

_Keep score rubric (0–5): 5 strong support: pre-registered goal met with a large effect, adopt. 4 support: goal met, carry forward. 3 partial: mixed or one-sided evidence, keep for further testing. 2 weak: goal not met but an informative side result. 1 negative: fails, reject, kept only as a recorded result. 0 inconclusive or not run._

## Research question
Does modelling the sequence of previous child decisions with a GRU (the plan's EXP-031) improve the deep levels over the MLP's parent-centre conditioning?

## Component under test
Predictor

## Track
Trainable predictor; training-free PoE fusion (DEC-003)

## Pre-registered protocol
- **Variant:** `--decoder gru` (hard targets, no residual). The GRU (256 units) takes the previous child, the chart, the level and the parent-centre Fourier code; logits come from [GRU state, trunk].
- **Data and control:** full splits. The control is EXP-030 (MLP tree, hard targets, no residual), which gives PoE synthetic-val mean **6.24 ± 0.07°**.
- **Branches:** the variant is trained on **both** branches (DINOv3-L `grid4` and MoGe-2 `normals16~and`), 3 seeds each, 100 epochs. Otherwise the EXP-030 hyperparameters are unchanged (`scripts/train_probe_p3.py`, features stored as fp16 on the GPU).
- **Evaluation:** single branches and the equal-weight PoE (DEC-003) via `scripts/exp03x_fused_eval.py`, at greedy and beam 4.
- **Claim:** the variant "helps" if its PoE synthetic-val mean error is below the EXP-030 PoE by more than the larger seed std of the two. Lightbox and sunlamp are descriptive.
- **Phase-3 gate:** the configuration with the lowest PoE synthetic-val mean among the variants that help is frozen; if none helps, the EXP-030 configuration is kept.

## Runs
- 6 training runs (2 branches × 3 seeds) under `guarded.sh 32`, all completed. Chain log: `outputs/experiments/EXP-034_beam_decoding/phase3_chain.log`.
- Fused evaluation: `RUN-20261009-024435-seed0`.

## Results
Greedy decoding, mean over 3 seeds. Source: `RUN-20261009-024435-seed0/metrics/metrics.csv`. Lightbox and sunlamp are descriptive.

| method | synthetic_val mean ± std / median | lightbox mean / median | sunlamp mean / median | acc@10 (synth / lightbox / sunlamp) | acc@20 (synth / lightbox / sunlamp) |
|---|---|---|---|---|---|
| A: DINOv3-L `grid4` | 6.289 ± 0.072 / 5.19 | 42.1 / 12.1 | 49.7 / 19.0 | 0.930 / 0.413 / 0.247 | 0.991 / 0.655 / 0.519 |
| B: MoGe-2 `normals16~and` | 16.929 ± 0.059 / 5.93 | 43.2 / 10.1 | 59.9 / 18.2 | 0.811 / 0.497 / 0.330 | 0.895 / 0.661 / 0.517 |
| **PoE(A,B)** | 6.149 ± 0.098 / 4.87 | 31.0 / 9.0 | 40.6 / 13.4 | 0.944 / 0.553 / 0.374 | 0.990 / 0.754 / 0.628 |
| EXP-030 control, PoE (MLP, hard, greedy) | 6.242 ± 0.072 / 5.00 | 31.1 / 10.4 | 43.7 / 17.0 | 0.932 / 0.479 / 0.290 | 0.989 / 0.729 / 0.553 |

## Pre-registered verdict
**FAIL (narrowly).** PoE synthetic val 6.149 ± 0.098° vs 6.242 ± 0.072°. The gain (0.093°) is smaller than the larger seed std (0.098°).

## Interpretation
- **Conditioning on the decoded history helps the deep levels:** DINOv3 alone improves from 6.64° to 6.29°.
- **The fused gain is smaller.** PoE already corrects many of the same deep-level mistakes.
- **Real domains improve the most** (descriptive): about +7 percentage points acc@10 on lightbox and +8 on sunlamp.

## Decision
Not frozen alone. Candidate for a combined soft-target + GRU test.
