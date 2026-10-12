# EXP-029: PanSt3R-inspired learned feature fusion vs training-free PoE

## Status
Completed (2026-10-10, 08:22–09:40). Pre-registered before any result. All 9 training runs and the evaluation completed under `guarded.sh 40`. Decision: training-free PoE kept (DEC-008).

## Keep score
**2 / 5: Weak.** The pre-registered rule technically passes, but the learned fusion adds nothing measurable over DINOv3 alone. Its "gain" over PoE is the known DINOv3-vs-PoE synthetic gap (DEC-004), and it is much worse on the real domains (descriptive).

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
- **New runs:** `RUN-20261010-0822`–`0909` (arms 2 / 4 / 5 × seeds 0–2).
- **Reused:** arm 1 and the PoE16 reference are the EXP-036 runs (listed in `outputs/experiments/EXP-029_panst3r_inspired_fusion/RUN-20261010-091704-seed0/source_runs.txt`).
- **Evaluation:** `outputs/experiments/EXP-029_panst3r_inspired_fusion/RUN-20261010-091704-seed0`. Chain log: `outputs/experiments/EXP-029_panst3r_inspired_fusion/chain.log`.

## Quantitative results
Greedy, mean over 3 seeds; source `outputs/experiments/EXP-029_panst3r_inspired_fusion/RUN-20261010-091704-seed0/metrics/metrics.csv`. Lightbox and sunlamp are descriptive (DEC-006).

| arm | params | synth mean ± std | median | p95 | acc@10 | >90° | lightbox mean / median / >90° | sunlamp mean / median / >90° |
|---|---|---|---|---|---|---|---|---|
| 1 DINO-only | 2.39M | 5.345 ± 0.103 | 4.80 | 9.79 | 0.955 | 0.001 | 37.1 / 13.5 / 0.153 | 41.5 / 20.5 / 0.159 |
| 2 Geometry-only 8×8 | 2.13M | 20.309 ± 0.285 | 6.52 | 131.0 | 0.752 | 0.086 | 48.5 / 12.2 / 0.258 | 66.0 / 24.5 / 0.367 |
| 3 **PoE8** (control) | 4.52M (two predictors) | 5.582 ± 0.103 | 4.70 | 10.34 | 0.944 | 0.002 | **27.9 / 9.5 / 0.110** | **37.9 / 14.1 / 0.156** |
| 4 Concat + linear | 2.39M | 5.381 ± 0.077 | 4.83 | 9.89 | 0.953 | 0.001 | 38.3 / 13.4 / 0.163 | 41.2 / 19.9 / 0.162 |
| 5 Concat + MLP | 3.45M | **5.341 ± 0.031** | 4.73 | 9.78 | 0.955 | 0.001 | 40.8 / 15.1 / 0.170 | 46.2 / 22.4 / 0.186 |
| ref PoE16 (EXP-036) | ~4.52M | 5.543 ± 0.064 | 4.63 | 10.19 | 0.947 | 0.002 | 26.9 / 9.0 / 0.105 | 36.7 / 13.0 / 0.154 |

**Per-seed synthetic-val mean** (seed 0 / 1 / 2):
- PoE8: 5.700 / 5.535 / 5.511
- concat + MLP: 5.307 / 5.346 / 5.369
- concat + linear: 5.458 / 5.305 / 5.381
- DINO-only: 5.455 / 5.249 / 5.332

**Paired per-image comparisons, seed-averaged errors** (source `outputs/experiments/EXP-029_panst3r_inspired_fusion/RUN-20261010-091704-seed0/metrics/paired.csv`):
- **Concat + MLP vs PoE8:** mean difference −0.24° on synthetic val. Median difference 0.00°; 25% of images better and 24% worse by more than 1°. On lightbox the mean is +12.8° (median +3.8°), with 2.6% of > 90° failures fixed and 9.0% newly broken.
- **Concat + MLP vs DINO-only:** −0.15 / +0.10 / +0.04° on synthetic val (seeds 0 / 1 / 2). The per-image error correlation is only about 0.5, so the two disagree per image but have the same average.
- **PoE16 vs PoE8 (cost of the coarser geometry grid):** −0.04° on synthetic val; −1.1° on lightbox and −1.2° on sunlamp.

**Resources** (`latency_ms_per_image_b1024`, `peak_infer_mem_mb_b1024` in metrics.csv): the single-predictor fusion arms run one decoder instead of two.

## Pre-registered verdict
- **Selected fusion arm:** concat + MLP (5.341° vs concat + linear's 5.381°).
- **vs PoE8:** below it by 0.241° > 0.103° (the larger seed std), and lower in all 3 seed pairs. By the pre-registered rule, **fusion "wins"**. This is a best-of-two selection, and a gap larger than one std is not a significance test.

## Interpretation (observation vs speculation)
- **Observed:** the fusion arms reach exactly DINO-only accuracy on synthetic val (5.34–5.38° vs 5.35°), and on the real domains they behave like DINO-only, or worse (MLP). The pre-registered rule compared fusion with PoE8 only. Its "win" is the same synthetic gap by which DINOv3 alone already beats PoE (DEC-004: 5.35° vs 5.54°). **The learned fusion extracts no measurable extra value from the geometry tokens.**
- **Observed (descriptive):** the training-free PoE keeps its large real-domain advantage (lightbox 27.9° vs 40.8°; sunlamp 37.9° vs 46.2°).
- **Speculation:** with z-scored inputs and a single synthetic-trained projection, the network relies on the dominant 1024-d DINO channels and ignores the 4 geometric channels. Late (posterior) fusion keeps the geometry expert's independent vote, which is what helps under domain shift.

## Decision
**Keep PoE; learned feature fusion not adopted (DEC-008, 2026-10-13: the user said "continue" after the recommendation below).** The pre-registered "if fusion wins, record a revised architecture" would switch to a predictor that is equivalent to DINO-only on synthetic val and much worse on real data. That conflicts with ground rule 15 ("do not choose a method solely because it has the best synthetic validation result"), and it is the same selection-policy question as DEC-004.
- **Recommended:** keep PoE (DEC-003) and record EXP-029 as a negative result for learned feature fusion. Optionally run an EXP-029 follow-up that rebalances the sources (per-source LayerNorm or a learned source gate) before concatenation.
- **Alternative:** adopt concat + MLP per the literal rule.

## Next experiment permitted by the gate
EXP-043 (flow-induced Tesseract masses) on the representation decided above. If the representation changes, matched controls are required.
