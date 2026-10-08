# EXP-010: DINOv3-only frozen representation + hierarchical MLP predictor

## Status
Completed (subset v1 sanity run)


## Keep score
**4 / 5: Support.** DINOv3-L grid4 is the strongest branch (17.2° synthetic-val median). The large synthetic→real gap (62–70° real-domain medians) keeps this from a 5.

_Keep score rubric (0–5): 5 strong support: pre-registered goal met with a large effect, adopt. 4 support: goal met, carry forward. 3 partial: mixed or one-sided evidence, keep for further testing. 2 weak: goal not met but an informative side result. 1 negative: fails, reject, kept only as a recorded result. 0 inconclusive or not run._

## Research question
How much rotation information do frozen DINOv3 features carry? Which model size, layer and pooling works best, and how much does performance fall from synthetic to lightbox and sunlamp?

## Hypothesis
Deeper layers carry more pose-relevant information. Spatially pooled patch tokens beat global CLS or mean pooling, because rotation is a spatial-layout property.

## Component under test
Fusion input representation: single branch (DINOv3)

## Track
Trainable predictor. The backbone is frozen, and there is no fusion.

## Fixed setup
- **Dataset and exact split:** SPEED+ subset v1 (`outputs/data_manifests/subset_v1.csv`, hash `bbdb3e2e3f5b70a1`), seeded (seed 0) and drawn from the official splits.
  - **Train:** 6,000 synthetic/train images.
  - **Model selection:** 1,500 synthetic/validation images.
  - **Scored only, never used for selection:** 1,000 lightbox/test and 1,000 sunlamp/test images.
- **DINOv3 weights:** Meta's LVD-1689M checkpoints, loaded through the timm hub (ungated) because the `facebook/dinov3-*` repos need a token:
  - `timm/vit_base_patch16_dinov3.lvd1689m` (86M params; blob `1f9ed8a2…`);
  - `timm/vit_large_patch16_dinov3.lvd1689m` (blob sha256 `45172f20…`).
  - They have not yet been cross-checked numerically against `facebook/dinov3-*`; that needs a valid HF token.
- **Layers** (`forward_intermediates`, `norm=True`): ViT-B blocks 3, 7, 11; ViT-L blocks 11, 17, 23.
- **VGGT variant/output/layer:** n/a.
- **Predictor:** common hierarchical MLP (`src/tfpose/predictor.py`).
  - Trunk: LN → Linear(d, 512) → GELU → Linear(512, 512).
  - Root head: 4 charts.
  - Child head: [trunk, level embedding + Fourier(parent-cell centre)] → 8.
  - Hard path labels, teacher forcing during training only.
  - AdamW (lr 1e-3, wd 0.05), one-cycle schedule, 100 epochs, batch 256, dropout 0.1.
  - Features standardized with train-split statistics.
  - The checkpoint is chosen by synthetic-val mean error, evaluated every 5 epochs with greedy decoding.
- **Tesseract depth and codebook size:** L = 5 (131,072 leaves). The L5 quantization floor is a 2.5° mean.
- **Input resolution and preprocessing:**
  - GT square crop around the 11 projected Tango keypoints (distortion applied, 15% margin per side), bilinear, zero padding.
  - Resized to 256×256, grayscale replicated to RGB, ImageNet normalization.
  - bf16 autocast for the backbone.
- **Seeds:** 0, 1, 2 for every feature set. One 5-epoch smoke run (seed 99) is excluded from all tables.
- **Compute device:** RTX 6000 Ada, shared with other users' jobs at 100% utilization.
- **Git commit:** extraction at `0115c32`; probe runs at `44cdb0d` (4 runs), `89d251a` (22) and `8e57fba` (5), per each run's `environment.txt`. The predictor and training code are identical across these three commits; later commits only add scripts and the run-directory fix.
- **Flow base distribution, solver, NFE and endpoint samples:** n/a.

## Changed variable
Feature source:
- ViT-B: layer 3, 7, 11 CLS+mean; layer 11 CLS only and mean only; the last-layer 4×4 pooled patch grid (`grid4`).
- ViT-L: layer 11, 17, 23 CLS+mean; last-layer `grid4`.

## Method
1. Extract features once into an immutable cache: `outputs/shared_cache/dinov3_vit{b,l}16__subset_v1__<preprocess-hash>__<manifest-hash>/`.
2. Train the same predictor on each feature set with 3 seeds.
3. Decode with greedy and beam search (widths 2, 4, 8) without the GT path.
4. Score geodesic error for the decoded L5 leaf centre, plus root, per-level and full-path accuracy and top-5 leaf recall.

## Commands
```
python scripts/make_subset_manifest.py --name v1 --seed 0
python scripts/extract_features.py --backbone dinov3_vitb16 --batch 32
python scripts/extract_features.py --backbone dinov3_vitl16 --batch 32
python scripts/train_probe.py --exp EXP-010_dinov3_baseline --features <backbone>:<featset> --seed {0,1,2}
python scripts/summarize_probe.py --exp EXP-010_dinov3_baseline
```

## Runs
31 runs: 30 sweep runs plus 1 smoke run. All completed in 35–85 s each. The full list is in `outputs/experiments/EXP-010_dinov3_baseline/run_list.csv`.
- **Lost attempt:** the first `dinov3_vitb16:l11_cls+mean` seed-1 attempt crashed before creating its run directory. A concurrent sweep claimed the same `RUN-<second>-seed1` name, giving `FileExistsError` in `runlog.Run`.
  - The fix (atomic directory claim with retry) is committed.
  - That configuration was rerun as `RUN-20261005-182546-seed1`.
- **Excluded:** `RUN-20261005-181032-seed99` is the 5-epoch pipeline smoke test and is not used in any table.

| Features | Trainable params | Runs (seed 0/1/2) |
|---|---:|---|
| vitb16:l3_cls+mean | 1,394,444 | 181050 / 181133 / 181227 |
| vitb16:l7_cls+mean | 1,394,444 | 181304 / 181403 / 181448 |
| vitb16:l11_cls | 999,692 | 181547 / 181634 / 181729 |
| vitb16:l11_mean | 999,692 | 181821 / 181913 / 182005 |
| vitb16:l11_cls+mean | 1,394,444 | 182054 / 182546 / 182154 |
| vitb16:grid4 | 6,920,972 | 182243 / 182350 / 182443 |
| vitl16:l11_cls+mean | 1,657,612 | 181241 / 181321 / 181416 |
| vitl16:l17_cls+mean | 1,657,612 | 181509 / 181600 / 181656 |
| vitl16:l23_cls+mean | 1,657,612 | 181743 / 181843 / 181928 |
| vitl16:grid4 | 9,026,316 | 182025 / 182153 / 182313 |

## Quantitative results
Greedy decoding, mean ± std over 3 seeds. Errors are in degrees; accuracies are fractions. Source: `outputs/experiments/EXP-010_dinov3_baseline/summary_by_features.csv`.

| features | domain | mean | median | acc@10 | acc@20 | root acc | path acc L2 | path acc L3 |
|---|---|---|---|---|---|---|---|---|
| vitb16:l3_cls+mean | synthetic_val | 79.98 ± 0.22 | 64.21 ± 0.51 | 0.049 ± 0.006 | 0.186 ± 0.005 | 0.617 ± 0.009 | 0.210 ± 0.004 | 0.076 ± 0.003 |
| vitb16:l7_cls+mean | synthetic_val | 55.16 ± 0.56 | 30.14 ± 0.54 | 0.097 ± 0.006 | 0.342 ± 0.007 | 0.765 ± 0.000 | 0.358 ± 0.007 | 0.137 ± 0.012 |
| vitb16:l11_cls | synthetic_val | 36.62 ± 0.98 | 19.90 ± 0.49 | 0.166 ± 0.020 | 0.502 ± 0.013 | 0.854 ± 0.004 | 0.450 ± 0.001 | 0.206 ± 0.010 |
| vitb16:l11_mean | synthetic_val | 46.46 ± 0.28 | 22.81 ± 0.64 | 0.156 ± 0.009 | 0.441 ± 0.010 | 0.811 ± 0.001 | 0.413 ± 0.006 | 0.196 ± 0.006 |
| vitb16:l11_cls+mean | synthetic_val | 37.52 ± 0.32 | 19.52 ± 0.13 | 0.174 ± 0.008 | 0.513 ± 0.003 | 0.853 ± 0.010 | 0.456 ± 0.005 | 0.216 ± 0.011 |
| vitb16:grid4 | synthetic_val | 29.94 ± 0.43 | 18.26 ± 0.55 | 0.184 ± 0.010 | 0.558 ± 0.016 | 0.893 ± 0.001 | 0.550 ± 0.004 | 0.260 ± 0.005 |
| vitl16:l11_cls+mean | synthetic_val | 64.10 ± 1.20 | 39.35 ± 0.42 | 0.071 ± 0.003 | 0.267 ± 0.002 | 0.705 ± 0.005 | 0.290 ± 0.005 | 0.110 ± 0.001 |
| vitl16:l17_cls+mean | synthetic_val | 47.08 ± 0.44 | 24.66 ± 0.33 | 0.127 ± 0.004 | 0.410 ± 0.006 | 0.805 ± 0.008 | 0.419 ± 0.008 | 0.182 ± 0.002 |
| vitl16:l23_cls+mean | synthetic_val | 38.47 ± 0.89 | 20.58 ± 0.35 | 0.164 ± 0.010 | 0.486 ± 0.009 | 0.844 ± 0.007 | 0.443 ± 0.009 | 0.197 ± 0.010 |
| **vitl16:grid4** | synthetic_val | **27.68 ± 0.80** | **17.20 ± 0.47** | **0.201 ± 0.021** | **0.585 ± 0.017** | **0.910 ± 0.003** | **0.574 ± 0.003** | **0.276 ± 0.019** |

Chance level for a uniformly random rotation: 126.5° mean, root accuracy 0.25.

## Domain-wise results
Greedy, mean ± std over 3 seeds.

| features | synthetic_val median | lightbox mean / median | lightbox acc@20 | lightbox root | sunlamp mean / median | sunlamp acc@20 | sunlamp root |
|---|---|---|---|---|---|---|---|
| vitb16:l3_cls+mean | 64.2 | 120.5 / 125.7 | 0.009 | 0.302 | 122.5 / 128.9 | 0.008 | 0.299 |
| vitb16:l7_cls+mean | 30.1 | 108.0 / 112.8 | 0.033 | 0.382 | 114.7 / 120.9 | 0.017 | 0.330 |
| vitb16:l11_cls | 19.9 | 96.9 / 96.3 | 0.049 | 0.432 | 103.5 / 106.5 | 0.037 | 0.441 |
| vitb16:l11_mean | 22.8 | 99.4 / 102.2 | 0.056 | 0.440 | 105.0 / 109.7 | 0.040 | 0.414 |
| vitb16:l11_cls+mean | 19.5 | 96.5 / 97.8 | 0.060 | 0.449 | 104.8 / 108.1 | 0.042 | 0.424 |
| vitb16:grid4 | 18.3 | 81.3 / 74.2 | 0.111 | 0.574 | 82.6 / 75.4 | 0.115 | 0.591 |
| vitl16:l11_cls+mean | 39.3 | 112.4 / 118.8 | 0.020 | 0.384 | 114.1 / 118.5 | 0.013 | 0.377 |
| vitl16:l17_cls+mean | 24.7 | 106.1 / 111.1 | 0.043 | 0.421 | 105.8 / 106.8 | 0.016 | 0.498 |
| vitl16:l23_cls+mean | 20.6 | 95.3 / 95.6 | 0.062 | 0.473 | 99.1 / 100.0 | 0.044 | 0.513 |
| **vitl16:grid4** | **17.2** | **76.1 / 62.2** | **0.145** | **0.612** | **79.1 / 70.3** | **0.109** | **0.629** |

Beam search for vitl16:grid4 (beam widths 1/2/4/8; scored nodes per image 44/84/164/292):
- synthetic_val mean error 27.7 / 28.1 / 28.1 / 28.1°;
- lightbox 76.1 / 75.9 / 75.9 / 75.9°;
- sunlamp 79.1 / 79.4 / 79.3 / 79.3°;
- full-path (L5) accuracy ≤ 0.013;
- top-5 leaf recall at beam 8: 0.044 on synthetic val, 0.008 on lightbox, 0.004 on sunlamp.

## Qualitative results
Per-image greedy predictions (GT path, predicted path, quaternion, error) are saved in `predictions/predictions_greedy.csv` for every run. No figure has been made yet; error-by-object-size and glare analyses are in EXP-012.

## Resource results
- **Parameters:** DINOv3 ViT-B/16 86M and ViT-L/16 about 300M, both frozen and asserted to have no gradients (`features.assert_frozen`).
- **Trainable parameters:** 1.0M (single vector) to 9.0M (vitl16 grid4).
- **Peak memory:** extraction peak GPU 654 MB (ViT-B) and 1,652 MB (ViT-L) at batch 32; probe training peak GPU < 1 GB.
- **Latency:**
  - Extraction 14.7 ms/img (ViT-B) and 13.5 ms/img (ViT-L), including JPEG decode and crop on a shared GPU.
  - The predictor head is under 1 ms/img.
- **Cache size (subset, 9,500 images):** 3.6 GB (ViT-B) and 4.9 GB (ViT-L). The full 16×16 last-layer grid is kept.
- **Tesseract nodes visited:** greedy 44; beam 8: 292.
- **Flow sampling latency and NFE:** n/a.
- **Parent-child mass consistency error:** n/a.
- **Credible-region coverage:** n/a.

## Comparison with control
No external control yet. EXP-011 (VGGT) uses the identical predictor, budget, subset and seeds.

## Interpretation
Observations:
1. **Depth helps monotonically.** On synthetic val:
   - ViT-B median falls from 64° (L3) to 30° (L7) to 20° (L11);
   - ViT-L median falls from 39° (L11) to 25° (L17) to 21° (L23).
2. **Spatial pooling helps most, and clearly most off-domain.** grid4 beats CLS+mean for both backbones:
   - on synthetic val, ViT-L mean error drops from 38.5° to 27.7°;
   - on lightbox, the ViT-L median drops from 95.6° to 62.2° and root accuracy rises from 0.47 to 0.61.

   Rotation depends on the spatial layout of parts, which global pooling discards.
3. **ViT-L grid4 is the best DINO representation** on synthetic val, which is the selection criterion. It also has the best real-domain numbers, though these were not used for selection.
4. **The synthetic→real gap is very large.**
   - The best lightbox/sunlamp medians (62° / 70°) are about 4× the synthetic-val median (17°).
   - acc@20 drops from 0.585 to 0.145 / 0.109.
   - Real-domain root accuracy (0.61–0.63) is well above chance (0.25), so some transferable signal exists.
5. **Absolute accuracy is far from usable**, about 17° median on synthetic val. The probe memorizes its 6,000 training images (train loss about 0.02) while validation plateaus by about epoch 45. This regime is data-limited.
   - Deeper Tesseract levels are barely learned (L3 path accuracy 0.28; full L5 path accuracy 0.01).
   - Beam search therefore has nothing to recover (≤0.4° change).

Speculation, not tested here: more training data (the full 48k synthetic set), geodesic soft targets and augmentation are the obvious levers for the data-limited regime. None of these are Phase-1 variables.

## Failure analysis
- One run attempt was lost to a run-directory name collision. It was fixed and rerun; there is no effect on results.
- Real-domain performance is poor across all DINO feature sets. This is a measured result, not a bug: GT crops are used and the convention was verified in EXP-000.

## Decision
Keep **DINOv3 ViT-L/16 grid4** as the strongest DINO branch for Phase 1 and the complementarity audit (EXP-012). These are subset-scale sanity results. They should be repeated on the full synthetic training split before any Phase-2 decision.

## Next experiment
EXP-011 (VGGT controls with the identical predictor and budget), then EXP-012.

## Artifact index
- Metrics JSON: `outputs/experiments/EXP-010_dinov3_baseline/RUN-*/metrics/metrics.json`
- CSV: `outputs/experiments/EXP-010_dinov3_baseline/summary_by_features.csv` (aggregate); `RUN-*/metrics/metrics.csv`
- TSV: `outputs/experiments/EXP-010_dinov3_baseline/summary_by_features.tsv`; `RUN-*/metrics/metrics.tsv`
- Figures: none yet
- Predictions: `RUN-*/predictions/predictions_greedy.csv`
- Checkpoint: `RUN-*/checkpoints/best.pt` (gitignored, local only)
- Logs: `RUN-*/logs/train.log`, `status.json`, `environment.txt`, `feature_source.json`; extraction logs in `outputs/shared_cache/logs/`
