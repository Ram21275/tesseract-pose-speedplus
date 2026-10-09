# EXP-102: LoRA adapters to specialize DINOv3 and MoGe-2 (backbone adaptation): DINOv3 only, MoGe-2 only, fused

## Status
**Cancelled by the user on 2026-10-09, before any training run.** Pre-registered on 2026-10-09. No adapter or control run was executed (the chain was still waiting for EXP-032/036). The code (`src/tfpose/adapters.py`, `scripts/train_adapter.py`, `scripts/exp102_fused_eval.py`), its tests, and the init-equivalence check are kept for possible future use. User-requested side experiment (DEC-002 numbering).

**Separately named backbone-adaptation experiment** (ground rules §2.4): its results are not mixed into the frozen-backbone comparison.

## Research question
Does lightly specializing the backbones with LoRA adapters, trained on SPEED+ synthetic train images, improve rotation prediction over frozen backbones with the same head and training budget? Does it help each branch, and the PoE fusion of the two adapted branches?

## Component under test
Backbone adaptation + predictor

## Track
Trainable (backbone adapters + head); training-free PoE fusion

## Pre-registered protocol
- **Adapters** (`src/tfpose/adapters.py`):
  - LoRA, rank 8, α = 16, on every block's attention `qkv` and `proj` Linear layers (24 blocks).
  - Base weights stay frozen; B = 0 at init, so the adapted model starts exactly at the frozen backbone. This was verified: init features match the cached frozen features to within bf16 rounding (DINOv3 relative error 0.97%; MoGe-2 normals mean |Δ| 0.004).
  - 1.18M trainable LoRA parameters per backbone.
- **Branches and features** (computed live on GT crops; identical construction to the frozen pipeline):
  - **DINOv3-L:** last-layer tokens → 4×4 grid (`grid4`).
  - **MoGe-2 encoder:** normals at 518 → 74×74, masked by the **cached** EXP-015 consensus mask (held fixed), → 16×16 (`normals16~and`).
  - Both are standardized with the **frozen** features' train-split statistics.
- **Head:** the simple MLP Tesseract predictor (EXP-030 `HierMLP`, L5), hard targets, greedy decoding.
- **Optimization:**
  - AdamW: LoRA lr 1e-4 (no weight decay), head lr 1e-3 (wd 0.05).
  - One-cycle schedule (5% warm-up), gradient clip 1.0, bf16 autocast.
  - Checkpoint selected by synthetic-val mean error each epoch.
- **Compute caps** (fixed now, from throughput and memory only; no accuracy seen):
  - **DINOv3:** all of synthetic train, **5 epochs**, batch 32.
  - **MoGe-2:** a fixed seeded **50%** of synthetic train (`--train-frac 0.5`, rng 0), **2 epochs**, batch 4 × gradient accumulation 2, gradient checkpointing. 518-px MoGe-2 backprop does not fit at larger batches on the shared GPU.
- **Arms:**
  1. adapted DINOv3 alone;
  2. adapted MoGe-2 alone;
  3. **fused** = equal-weight PoE of the two adapted heads (DEC-003 rule).
- **Matched frozen controls:** the same head, optimizer, batch, epochs, train subset and gradient clip, on the cached frozen features (`train_probe_p3.py --epochs/--batch/--train-frac/--clip`). 3 seeds per branch, plus their PoE.
- **Seeds:**
  - **Stage 1:** seed 0 for both adapter branches (pilot).
  - **Stage 2:** seeds 1–2 for both adapters, run only if the Stage-1 adapted PoE beats the matched-control PoE on synthetic val by more than 2× the control's seed std.
- **Claim:** adaptation "helps" an arm if its synthetic-val mean is below the matched control's by more than the larger seed std (3-seed claims need Stage 2). Lightbox and sunlamp are descriptive.
- **Context, not a control:** EXP-030 (frozen, 100 epochs). The cost of adaptation (GPU hours, peak memory) is reported.
- **Scheduling:** runs start only after the EXP-032/036 chain finishes. Running both would exceed the shared GPU's memory: a smoke run hit OOM at 19.6 GB.

## Results
None (cancelled).

## Keep score
**0 / 5: Not run** (cancelled).
