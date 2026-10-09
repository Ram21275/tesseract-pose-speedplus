# EXP-032: Lightweight cross-attentive Tesseract Transformer, hard targets (Phase 3)

## Status
Completed. Pre-registered on 2026-10-09, before any result.

This experiment was required by the plan but skipped in the first Phase-3 batch; the gap was recorded on 2026-10-09 and the experiment is now run at the user's request.


## Keep score
**4 / 5: Support.** It passes clearly: PoE synthetic val 5.685° vs 6.242° (−0.56°, more than 5× std). It is the best Phase-3 result so far on the selection split, also below EXP-033 (6.07°). Real domains are mixed (descriptive): sunlamp improves (41.8° vs 43.7°), lightbox is unchanged (31.6° vs 31.1°).

_Keep score rubric (0–5): 5 strong support: pre-registered goal met with a large effect, adopt. 4 support: goal met, carry forward. 3 partial: mixed or one-sided evidence, keep for further testing. 2 weak: goal not met but an informative side result. 1 negative: fails, reject, kept only as a recorded result. 0 inconclusive or not run._

## Research question
Does an autoregressive Transformer decoder whose path queries cross-attend to spatial tokens beat the MLP tree on a pooled 4×4 vector (EXP-030)?

## Component under test
Predictor / decoder

## Track
Trainable predictor; training-free PoE fusion

## Pre-registered protocol
- **Variant:** `--decoder transformer`, hard targets.
- **Architecture** (`phase3.HierTransformer`):
  - frozen visual tokens → LayerNorm → Linear to d = 256, plus 2-D sinusoidal positions → 1 Transformer encoder layer (the memory);
  - a 2-layer causal decoder over the path: position 0 is [ROOT] and predicts the chart (4); position j embeds the cell reached after j−1 decisions (level embedding + Fourier(cell centre)) and predicts the next child (8);
  - cross-attention to the memory; 4 heads, feed-forward 512, dropout 0.1, pre-norm.
- **Tokens per branch** (each Transformer sees only its own branch, so this is **not** learned fusion):
  - DINOv3-L: the last-layer 16×16×1024 tokens pooled to **8×8×1024** (`dinov3_vitl16:tokens8`; 16×16 at full data, 36 GB, does not fit in memory);
  - MoGe-2: the 16×16 grid of (masked normal, consensus mask) tokens, **256×4** (`moge2_vitl:normals16~and`, the same information as the MLP feature).
- **Training:** full splits, 3 seeds, 100 epochs, AdamW 1e-3 / 0.05, batch 256, train-split standardization, selection on synthetic-val mean, as in EXP-030.
  - Implementation details (speed and memory only): bf16 autocast for the probe; features stored as fp16 on the GPU.
- **Evaluation:** single branches and the equal-weight PoE (DEC-003), greedy and beam 4 (`scripts/exp03x_fused_eval.py`).
- **Claim:** the variant helps if its PoE synthetic-val mean is below the **EXP-030** PoE (6.242 ± 0.072°) by more than the larger seed std. Lightbox and sunlamp are descriptive.

## Runs
- 6 training runs (3 seeds × 2 branches), all completed under `guarded.sh 32`. About 9 min per DINOv3 run and 19 min per MoGe-2 run.
- Chain log: `outputs/experiments/EXP-032_transformer_decoder/chain.log`.
- Fused evaluation: `RUN-20261009-111025-seed0`.
- **Smoke-test history** (in `EXP-099_p3_smoke`):
  - two memguard aborts (27 GB) while building DINOv3 tokens, fixed by chunk-wise fp16 pooling and unpinned host memory;
  - bf16 autocast halved the epoch time.

## Results
Greedy decoding, mean over 3 seeds. Source: `RUN-20261009-111025-seed0/metrics/metrics.csv`. Beam 4 changes the PoE synthetic-val mean by −0.02° only.

| method | synthetic_val mean ± std / median | lightbox mean / median | sunlamp mean / median | acc@10 (synth / lightbox / sunlamp) | acc@20 (synth / lightbox / sunlamp) |
|---|---|---|---|---|---|
| A: DINOv3-L Transformer (8×8 tokens) | 6.084 ± 0.111 / 5.23 | 43.7 / 15.9 | 52.8 / 26.2 | 0.924 / 0.287 / 0.159 | 0.993 / 0.587 / 0.407 |
| B: MoGe-2 Transformer (16×16 normal tokens) | 18.403 ± 0.364 / 6.14 | 47.1 / 11.3 | 65.1 / 22.9 | 0.787 / 0.453 / 0.278 | 0.886 / 0.630 / 0.477 |
| **PoE(A,B)** | 5.685 ± 0.103 / 4.82 | 31.6 / 10.4 | 41.8 / 15.4 | 0.950 / 0.482 / 0.304 | 0.993 / 0.742 / 0.597 |
| *EXP-030 control PoE (MLP, hard)* | *6.242 ± 0.072 / 5.00* | *31.1 / 10.4* | *43.7 / 17.0* | *0.932 / 0.479 / 0.290* | *0.989 / 0.729 / 0.553* |
| *EXP-033 PoE (MLP, soft; frozen by DEC-004)* | *6.068 ± 0.067 / 4.81* | *28.2 / 9.8* | *42.1 / 16.6* | *0.926 / 0.508 / 0.292* | *0.986 / 0.742 / 0.565* |

## Pre-registered verdict
**PASS.** PoE synthetic val is 5.685 ± 0.103° against the EXP-030 control's 6.242 ± 0.072°. The gain (0.557°) exceeds the larger std (0.103°).

## Interpretation
- **Cross-attention to the spatial token grid helps the DINOv3 branch.** It improves to 6.08° from the MLP's 6.64°, with 8×8 tokens where the MLP had a pooled 4×4 vector. This is the best single-branch result at hard targets.
- **For MoGe-2, the Transformer is on par with the MLP** (18.4° vs 17.8°). Its 16×16 normal-and-mask tokens already carry the same information as the MLP's flattened input.
- **The fused gain over the MLP (−0.56°) is larger than the soft-target gain (EXP-033, −0.17°)** on synthetic val.
- **On real domains (descriptive):** the Transformer helps sunlamp but not lightbox, where soft targets did better (28.2°). The two changes are complementary candidates, which is what EXP-036 (Transformer + soft targets) tests.

## Decision
Keep. It is the current best Phase-3 configuration on synthetic val. The Phase-3 gate (DEC-004) will be updated after EXP-036 with the lower PoE synthetic-val mean of EXP-032 and EXP-036.
