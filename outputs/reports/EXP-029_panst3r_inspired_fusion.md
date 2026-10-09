# EXP-029: PanSt3R-inspired learned feature fusion vs training-free PoE

## Status
**Queued.** Pre-registered on 2026-10-10, before any result. It starts automatically after the Phase-4 chain (EXP-040–042) finishes (`scripts/run_exp029.sh`, which waits for `run_phase4_resume.sh`). ID and placement: DEC-007 (the plan's Phase-2 EXP-024 slot, reopened).

## Keep score
Pending.

## Research question
Does learned joint feature fusion of frozen DINOv3 and MoGe-2 tokens into one Tesseract Transformer improve rotation prediction over the existing training-free per-level product of experts (DEC-003)?

## Reference and verification
- **Paper:** PanSt3R (Zust et al., ICCV 2025, arXiv:2506.21348), Sec. 3.1, plus the official code `naver/panst3r`. Sections read: 3.1, the training details and the 2D/3D backbone ablation; the rest of the paper was not needed.
- **v1, the paper's version:** the DINOv2-L encoder (1024), MUSt3R encoder (1024) and MUSt3R decoder (768) token maps (one token per 16×16 patch) are concatenated **raw**, with no per-source normalization. The 2,816-d result goes to `PixelShuffleUpscaler.proj_16 = Mlp(2816 → 4·2816 → 768, GELU)`, which gives the joint tokens used in cross-attention. Backbones are frozen.
- **v2:** `InputMixer` = Linear(2816 → 768) + 3 RoPE Transformer blocks + LayerNorm.
- **Ours is PanSt3R-*inspired*:**
  - two frozen sources instead of three;
  - geometry is the existing consensus-masked MoGe-2 normals, not latent decoder features;
  - inputs are z-scored per dimension (our existing rule) and LayerNorm'd before the projection;
  - the MLP hidden width is 1024, not 4 × input, to keep it "small" (3.45M vs 2.39M total parameters for the linear arm);
  - the joint token is 256-d to match our decoder width.

## Component under test
Fusion (learned, Track B), with the Phase-3 predictor recipe held fixed.

## Track
Trainable fusion + trainable predictor; frozen backbones; no LoRA or backbone fine-tuning.

## Fixed setup
- **Data:** full splits. Train on synthetic/train. Checkpoints and the keep/reject decision use synthetic/validation only (DEC-006). Lightbox and sunlamp are descriptive.
- **Grid:** a common **8×8** grid.
  - S_i = DINOv3-L last-layer token, 16×16 → 8×8 average pooled (`dinov3_vitl16:tokens8`, 1024-d).
  - G_i = MoGe-2 [masked-normal sum (3), consensus-mask coverage (1)], adaptive average pooled to 8×8 (`moge2_vitl:normals8~and`). This is exactly the `normals16~and` definition at 8×8: the coverage channel is the pooled binary EXP-015 AND mask, and the normal channels are the mask-weighted normal sum, with norm ≤ coverage (verified).
- **Alignment:** both come from the same GT square crop (`features.load_crops`, identical crop_cx/cy/side), with row-major (y, x) token order.
  - Held-out check (subset v1): a DINO foreground direction fit on half the images correlates with MoGe coverage at 0.877 aligned, vs 0.70/0.72/0.71 flipped vertically / flipped horizontally / transposed, and 0.65/0.68 shifted by one cell.
  - MoGe maps are adaptive-pooled from their native resolution, so bin edges may differ from DINO's by less than one source pixel.
- **Normalization:** per-dimension z-score with synthetic-train statistics (`featsets.standardize_to_tensor`). It is affine and invertible, so it preserves the meaning of the normals and coverage. For concatenated inputs it equals standardizing each source separately.
- **Recipe (all arms, = EXP-036):** `HierTransformer` (1 encoder layer + 2-layer causal path decoder, d 256, 4 heads, FF 512, dropout 0.1), depth 5, geodesic soft targets, 100 epochs, AdamW lr 1e-3, weight decay 0.05, OneCycle, batch 256, bf16 autocast, best synthetic-val checkpoint, greedy decoding, seeds 0/1/2.

## Arms

| # | Arm | Input to the predictor | Runs | Params |
|---|---|---|---|---|
| 1 | DINO-only | 64×1024 | **reused EXP-036 DINO runs** (identical config and code path) | 2.39M |
| 2 | Geometry-only 8×8 | 64×4 | new (EXP-029) | 2.13M |
| 3 | PoE8 = 1 + 2, equal-weight per-level PoE (DEC-003 rule) | two predictors | – | 4.52M (two predictors) |
| 4 | Concat + linear: [S‖G] 64×1028 → LayerNorm → Linear(1028→256) → one Transformer | 64×1028 | new, `--fuse linear` | 2.39M |
| 5 | Concat + MLP: [S‖G] → LayerNorm → Linear(1028→1024) → GELU → Linear(1024→256) → one Transformer | 64×1028 | new, `--fuse mlp` | 3.45M |
| ref | EXP-036 PoE (DINO tokens8 + MoGe `normals16~and`, 16×16 geometry) | two predictors | historical | ~4.5M |

Parameter differences are reported, not equalized. The two-predictor PoE has about twice the decoder capacity of the single-predictor fusion arms.

## Pre-registered decision rule (synthetic val only)
- **Selected fusion arm:** the better of arms 4 and 5 by mean synthetic-val error (3-seed mean).
- **Fusion wins reproducibly** if the selected arm's 3-seed mean synthetic-val error is below the PoE8 (arm 3) mean by more than the larger of the two seed stds, **and** it is lower than PoE8 in each of the 3 seed-paired comparisons.
  - Gaps larger than one std are reported as such, not as a formal significance test.
  - The selection between arms 4 and 5 makes this a best-of-two comparison; that is stated in the report.
- **If fusion wins:** record a revised architecture decision (a DEC entry). Later continuous-refinement experiments on the new representation then need matched controls.
- **Otherwise:** keep PoE (DEC-003) and record the negative result.
- **Also reported (descriptive):** arm 3 vs the historical 16×16 PoE, which shows the cost of the coarser geometry grid.

## Reported metrics
Per arm and domain (synthetic val, lightbox, sunlamp):
- mean / median / p95 error;
- acc@5/10/20;
- fraction of errors above 90° and above 150°;
- parameters, latency (ms per image, batch 1024, GPU) and peak inference memory;
- seed mean ± std;
- paired per-image comparisons vs PoE8 and vs PoE16: per seed and on seed-averaged errors, the mean and median difference, fraction better or worse by more than 1°, and flips fixed or broken (> 90°).

Script: `scripts/exp029_eval.py`.

## Commands
`scripts/run_exp029.sh`: 9 training runs under `guarded.sh 40`, one at a time, then `scripts/exp029_eval.py`.

## Runs
Pending.

## Quantitative results
Pending.

## Decision
Pending.

## Next experiment permitted by the gate
EXP-043 (flow-induced Tesseract masses) on the representation selected here, with matched controls if the representation changes.
