# EXP-032: Lightweight cross-attentive Tesseract Transformer, hard targets (Phase 3)

## Status
Running. Pre-registered on 2026-10-09, before any result.

This experiment was required by the plan but skipped in the first Phase-3 batch; the gap was recorded on 2026-10-09 and the experiment is now run at the user's request.

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

## Results
(to be filled)
