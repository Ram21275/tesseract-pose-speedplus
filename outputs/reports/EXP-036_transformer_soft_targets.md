# EXP-036: Tesseract Transformer + geodesic soft targets (Phase 3)

## Status
Completed. Pre-registered on 2026-10-09, before any result. User request: the Transformer combined with the best Phase-3 target choice (EXP-033).


## Keep score
**5 / 5: Strong support.** PoE synthetic val is 5.540 ± 0.063° vs EXP-033's 6.068° (−0.53°, more than 7× std), and it beats EXP-032 (5.685°). It is best on every domain (descriptive): lightbox 26.9° (median 9.0°), sunlamp 36.6° (median 13.0°). It is frozen by the Phase-3 gate (DEC-004 update).

_Keep score rubric (0–5): 5 strong support: pre-registered goal met with a large effect, adopt. 4 support: goal met, carry forward. 3 partial: mixed or one-sided evidence, keep for further testing. 2 weak: goal not met but an informative side result. 1 negative: fails, reject, kept only as a recorded result. 0 inconclusive or not run._

## Research question
Does the Transformer decoder improve on the frozen Phase-3 configuration (MLP tree + geodesic soft targets, EXP-033) when it is also trained with soft targets?

## Component under test
Predictor / decoder (combined with targets)

## Track
Trainable predictor; training-free PoE fusion

## Pre-registered protocol
- **Variant:** `--decoder transformer --targets soft` (τ_l as in EXP-033).
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
- **Claim:** the variant helps if its PoE synthetic-val mean is below the **EXP-033** PoE (6.068 ± 0.067°) by more than the larger seed std. If it helps, it replaces EXP-033 as the frozen Phase-3 predictor (DEC-004 update).

## Runs
- 6 training runs, all completed under `guarded.sh 32`.
- Chain log: `outputs/experiments/EXP-032_transformer_decoder/chain.log`.
- Fused evaluation: `outputs/experiments/EXP-036_transformer_soft_targets/RUN-20261009-124320-seed0`.

## Results
Greedy decoding, mean over 3 seeds. Lightbox and sunlamp are descriptive.

| method | synthetic_val mean ± std / median | lightbox mean / median | sunlamp mean / median | acc@10 (synth / lightbox / sunlamp) | acc@20 (synth / lightbox / sunlamp) |
|---|---|---|---|---|---|
| A: DINOv3-L Transformer, soft | **5.346 ± 0.105 / 4.80** | 37.1 / 13.5 | 41.4 / 20.5 | 0.955 / 0.358 / 0.204 | 0.996 / 0.650 / 0.490 |
| B: MoGe-2 Transformer, soft | 16.727 ± 0.106 / 5.89 | 43.7 / 10.4 | 60.5 / 18.3 | 0.805 / 0.485 / 0.312 | 0.896 / 0.656 / 0.517 |
| **PoE(A,B)** | 5.540 ± 0.063 / **4.62** | **26.9 / 9.0** | **36.6 / 13.0** | 0.947 / **0.555 / 0.374** | 0.990 / **0.776 / 0.652** |
| *EXP-033 PoE (MLP, soft)* | *6.068 ± 0.067 / 4.81* | *28.2 / 9.8* | *42.1 / 16.6* | *0.926 / 0.508 / 0.292* | *0.986 / 0.742 / 0.565* |
| *EXP-032 PoE (Transformer, hard)* | *5.685 ± 0.103 / 4.82* | *31.6 / 10.4* | *41.8 / 15.4* | *0.950 / 0.482 / 0.304* | *0.993 / 0.742 / 0.597* |

## Pre-registered verdict
**PASS.** PoE synthetic val is 5.540 ± 0.063° vs EXP-033's 6.068 ± 0.067°. It also has the lowest PoE synthetic-val mean of all Phase-3 variants, so the gate freezes it.

## Interpretation
- **The Transformer decoder and soft targets stack:** −0.53° over the soft-target MLP and −0.15° over the hard-target Transformer.
- **The real-domain gains are the largest yet** (descriptive): lightbox 31.1° → 26.9° and sunlamp 43.7° → 36.6° against the EXP-030 control.
- **The selection-policy issue returns.** On synthetic val, DINOv3 alone (5.35°) is again below the PoE (5.54°), while PoE is much better on lightbox (26.9° vs 37.1°) and sunlamp (36.6° vs 41.4°). The real-domain validation split the user picked (pending clarification) would settle this.

## Decision
Frozen as the Phase-3 predictor (DEC-004 update): Transformer decoder, geodesic soft targets, greedy decoding, PoE fusion (DEC-003).
