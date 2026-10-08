# EXP-013: MoGe-2 single-image geometry controls (VGGT replacement candidate)

## Status
Completed (subset v1 sanity run)

## Research question
Is a model trained for **single-image** geometry a better second branch than VGGT? MoGe-2 is tested with the same predictor, budget, subset and seeds as EXP-010 and EXP-011.

## Hypothesis
VGGT is trained for multi-view reconstruction and runs here on a single frame. A single-image geometry model should give more pose-informative tokens and maps. Its surface normals in particular encode surface orientation relative to the camera.

## Component under test
Fusion input representation: single branch (MoGe-2)

## Track
Trainable predictor. The backbone is frozen, and there is no fusion.

## Fixed setup
- **Scope change, user-approved on 2026-10-08:** MoGe-2 is evaluated as a replacement candidate for VGGT as the geometric branch. VGGT's results (EXP-011/012) are kept unchanged.
- **Dataset and exact split:** subset v1 (hash `bbdb3e2e3f5b70a1`).
  - Train: 6,000 synthetic/train images.
  - Selection: 1,500 synthetic/validation images.
  - Scored only: 1,000 lightbox and 1,000 sunlamp images.
- **DINOv3 variant/layer:** n/a. MoGe-2's own encoder is DINOv2 ViT-L/14, fine-tuned by Microsoft for geometry; it is a different model from our DINOv3 branch.
- **MoGe-2:** checkpoint `Ruicheng/moge-2-vitl-normal` (331M params, MIT licence, public). Code from github.com/microsoft/MoGe at `74fbce0`, plus `utils3d_moge` at `62f09d5`.
  - Encoder layers 5, 11, 17, 23: CLS token and mean patch token, 1024-d, from DINOv2 `get_intermediate_layers` with the final norm.
  - Last-layer 37×37 patch grid, pooled to 8×8.
  - `infer` outputs: depth, camera-frame point map, surface normals, foreground mask, each area-pooled to 74×74.
- **Predictor:** identical to EXP-010/011: hierarchical MLP, hidden 512, 100 epochs, AdamW lr 1e-3 / wd 0.05, batch 256, hard labels, checkpoint selected on synthetic-val mean error.
- **Tesseract depth and codebook size:** L = 5 (131,072).
- **Input resolution and preprocessing:**
  - The same GT square crop, resized to 518×518, grayscale replicated to RGB, values in [0, 1].
  - `num_tokens` = 37×37, the same token grid as VGGT.
  - Field of view predicted by the model (no intrinsics given), matching the VGGT control.
  - fp16 autocast; `apply_mask=False`, so all maps stay finite.
- **Seeds:** 0, 1, 2.
- **Compute device:** RTX 6000 Ada (shared; about 4–6 GB free during extraction, so batch 4).
- **Git commit:** extraction at `576f052`; all 30 probe runs at `9310141` (clean tree).
- **Safety:**
  - every job ran alone through `scripts/guarded.sh` (cgroup cap: 24 GB for extraction, 16 GB per probe, no swap, TasksMax 4096) plus the in-process memguard;
  - extraction retries a batch after a CUDA OOM.

## Changed variable
MoGe-2 feature source. It mirrors the EXP-011 list, plus one feature that VGGT cannot provide:
- `l{5,11,17,23}_cls+mean`, `l23_cls`, `l23_mean`, `grid4`;
- `depthconf16`: depth divided by the masked median, background zeroed, plus the mask;
- `points16`: mask-weighted centring and RMS scaling, plus the mask;
- **`normals16`**: masked surface normals plus the mask, pooled to 16×16.

## Method
Same as EXP-011: cached features, then 3 seeds of the common predictor, then greedy and beam decoding without the GT path.

## Commands
```
scripts/guarded.sh 24 -- python scripts/extract_features.py --backbone moge2_vitl --batch 4
scripts/run_exp013_sweep.sh          # each run: scripts/guarded.sh 16 -- python scripts/train_probe.py ...
python scripts/summarize_probe.py --exp EXP-013_moge2_controls
```

## Runs
All 30 runs completed (`outputs/experiments/EXP-013_moge2_controls/run_list.csv`), averaging 45–51 s each.

| Features | Feature dim | Trainable params | Runs (RUN-20261008-HHMMSS, seeds 0/1/2) |
|---|---:|---:|---|
| l5_cls+mean | 2048 | 1,657,612 | 155245 / 155332 / 155421 |
| l11_cls+mean | 2048 | 1,657,612 | 155509 / 155559 / 155646 |
| l17_cls+mean | 2048 | 1,657,612 | 155734 / 155821 / 155911 |
| l23_cls+mean | 2048 | 1,657,612 | 155958 / 160047 / 160129 |
| l23_cls | 1024 | 1,131,276 | 160219 / 160306 / 160356 |
| l23_mean | 1024 | 1,131,276 | 160443 / 160533 / 160621 |
| grid4 | 16384 | 9,026,316 | 160711 / 160802 / 160858 |
| depthconf16 | 512 | 868,108 | 160949 / 161041 / 161127 |
| points16 | 1024 | 1,131,276 | 161221 / 161307 / 161404 |
| normals16 | 1024 | 1,131,276 | 161449 / 161544 / 161629 |

## Quantitative results
Greedy decoding on synthetic validation, mean ± std over 3 seeds. Errors are in degrees. Source: `outputs/experiments/EXP-013_moge2_controls/summary_by_features.csv`.

| features | synthetic_val mean | synthetic_val median | acc@10 | acc@20 | root acc | L3 path acc |
|---|---|---|---|---|---|---|
| l5_cls+mean | 85.17 ± 0.55 | 79.53 ± 2.69 | 0.035 ± 0.002 | 0.134 ± 0.008 | 0.583 ± 0.008 | 0.052 ± 0.004 |
| l11_cls+mean | 53.81 ± 0.45 | 28.34 ± 1.00 | 0.108 ± 0.004 | 0.360 ± 0.015 | 0.763 ± 0.004 | 0.150 ± 0.010 |
| l17_cls+mean | 45.23 ± 0.75 | 22.58 ± 0.89 | 0.152 ± 0.004 | 0.455 ± 0.010 | 0.825 ± 0.002 | 0.206 ± 0.004 |
| l23_cls+mean | 52.15 ± 0.38 | 25.91 ± 1.38 | 0.122 ± 0.005 | 0.391 ± 0.008 | 0.811 ± 0.003 | 0.158 ± 0.012 |
| l23_cls | 52.32 ± 0.21 | 26.78 ± 0.47 | 0.120 ± 0.004 | 0.384 ± 0.010 | 0.808 ± 0.005 | 0.162 ± 0.004 |
| l23_mean | 74.27 ± 1.33 | 61.45 ± 3.86 | 0.067 ± 0.003 | 0.220 ± 0.011 | 0.701 ± 0.011 | 0.085 ± 0.008 |
| **grid4** | **39.60 ± 0.61** | **21.23 ± 0.41** | 0.154 ± 0.001 | **0.471 ± 0.010** | **0.838 ± 0.002** | **0.212 ± 0.004** |
| depthconf16 | 66.11 ± 0.69 | 37.91 ± 2.25 | 0.136 ± 0.008 | 0.349 ± 0.003 | 0.605 ± 0.007 | 0.155 ± 0.004 |
| points16 | 65.98 ± 0.81 | 33.99 ± 1.42 | 0.134 ± 0.007 | 0.362 ± 0.006 | 0.630 ± 0.004 | 0.157 ± 0.006 |
| normals16 | 56.38 ± 0.76 | 24.82 ± 1.04 | **0.179 ± 0.009** | 0.442 ± 0.010 | 0.722 ± 0.003 | 0.208 ± 0.008 |

## Domain-wise results
Greedy, mean over 3 seeds. These are descriptive only and were not used for any selection.

| features | lightbox mean / median | lightbox acc@20 | lightbox root | sunlamp mean / median | sunlamp acc@20 | sunlamp root |
|---|---|---|---|---|---|---|
| l5_cls+mean | 124.8 / 129.9 | 0.003 | 0.252 | 124.1 / 133.4 | 0.007 | 0.265 |
| l11_cls+mean | 112.2 / 120.1 | 0.022 | 0.363 | 111.1 / 115.5 | 0.020 | 0.397 |
| l17_cls+mean | 107.8 / 114.8 | 0.048 | 0.406 | 106.8 / 110.0 | 0.037 | 0.414 |
| l23_cls+mean | 101.1 / 103.9 | 0.050 | 0.438 | 101.8 / 102.7 | 0.032 | 0.491 |
| l23_cls | 102.9 / 106.6 | 0.046 | 0.427 | 105.4 / 109.1 | 0.026 | 0.462 |
| l23_mean | 108.5 / 114.6 | 0.033 | 0.403 | 107.8 / 110.0 | 0.021 | 0.410 |
| grid4 | 83.3 / 78.0 | 0.113 | 0.568 | 83.9 / 81.2 | 0.101 | 0.580 |
| depthconf16 | 95.1 / 102.1 | 0.139 | 0.451 | 102.9 / 110.8 | 0.090 | 0.425 |
| points16 | 94.4 / 98.5 | 0.153 | 0.467 | 103.1 / 110.1 | 0.088 | 0.426 |
| normals16 | 84.0 / 84.6 | 0.202 | 0.550 | 100.1 / 105.0 | 0.092 | 0.464 |

## Qualitative results
Visual check of normals, mask and depth on 2 crops per domain (`figures/moge2_maps_examples.jpg`):
- **Synthetic:** MoGe-2 gives clean geometry. Each spacecraft face has a distinct normal, the foreground mask is tight, and depth is coherent.
- **Lightbox and sunlamp: the foreground mask fails.** It labels **89% (lightbox) and 96% (sunlamp)** of crop pixels as object, against 58% on synthetic. Sensor noise, stray light and the Earth-like backdrop are read as a surface. The normals still show the face structure on the object, but the background becomes a flat plane.
- **Consequence:** the mask-weighted dense features (`depthconf16`, `points16`, `normals16`) are corrupted off-domain. This is recorded as a failure mode. It was observed on test-domain images, so the features were **not** changed in response.

## Resource results
- **Parameters:** MoGe-2 ViT-L 331M, frozen (`assert_frozen` on every batch).
- **Trainable parameters:** 0.87M to 9.0M.
- **Peak memory:** extraction about 2.9 GB GPU at batch 4; probes < 1 GB GPU; host RSS inside the 16–24 GB caps (no memguard or cgroup kill occurred).
- **Latency:**
  - extraction 128 ms/img on the shared GPU, including JPEG decode and crop;
  - VGGT-1B: 246 ms/img;
  - DINOv3-L: 13.5 ms/img.
- **Cache:** 2.1 GB for 9,500 images.
- **Tesseract nodes visited:** greedy 44.
- **Flow sampling latency and NFE:** n/a.
- **Parent-child mass consistency error:** n/a.
- **Credible-region coverage:** n/a.

## Comparison with control
Best per branch, chosen by the same synthetic-val mean rule:

| branch (best feature set) | synthetic_val mean | median | acc@20 | root | extraction ms/img |
|---|---:|---:|---:|---:|---:|
| DINOv3-L `grid4` (EXP-010) | **27.7** | **17.2** | **0.585** | **0.910** | 13.5 |
| **MoGe-2 `grid4`** (EXP-013) | 39.6 | 21.2 | 0.471 | 0.838 | 128 |
| VGGT `l11_cam+mean` (EXP-011) | 48.7 | 25.4 | 0.397 | 0.813 | 246 |

MoGe-2 beats VGGT on synthetic val by 9.1° mean error (39.6 vs 48.7; seed std ≤ 0.8°). Tokens and the spatial grid improve on VGGT; dense depth and points are on par:

| feature family | MoGe-2 mean | VGGT mean |
|---|---:|---:|
| tokens (best layer) | 45.2 | 48.7 |
| `grid4` | 39.6 | 49.1 |
| point map | 66.0 | 67.2 (≈1 std, on par) |
| depth | 66.1 | 66.1 (tie) |

MoGe-2 is still clearly behind DINOv3. Extraction timings (13.5 / 128 / 246 ms per image) were measured at different times, batch sizes and loads on a GPU shared at 100% with other users, so they are indicative only and are not used as decision evidence.

## Interpretation
Observations, decided on synthetic validation:
1. **The single-image geometry model is a better single geometric branch than VGGT.** Its best token, grid and normal features beat VGGT's best; dense depth and points are on par.
2. **Surface normals are the most pose-informative dense map.** `normals16` (1,024 numbers) reaches acc@10 0.179, the highest of any MoGe-2 or VGGT feature, against VGGT's best at 0.122. Its median of 24.8° approaches DINOv3 grid4's 17.2°. This supports the reason for choosing MoGe-2: normals encode face orientation.
3. **Layer profile.** MoGe-2's best single layer is 17 (median 22.6°), similar to VGGT, where a middle layer was best. Its last-layer mean-pooled token is poor (median 61.4°), so its pose information sits in the CLS token and the spatial layout.
4. **Off-domain (descriptive).** MoGe-2 `grid4` is better than VGGT's best tokens on lightbox and sunlamp (means 83–84° vs 101–107°) but still worse than DINOv3 (76–79°). The foreground-mask failure limits the dense maps there.

## Failure analysis
- No run failed. The extraction batch was reduced from 8 to 4 after a CUDA OOM in a smoke test; other users held about 42 GB of the 47 GB GPU. The OOM-retry logic was added afterwards and was not triggered during the real extraction.
- The foreground-mask failure on real images is a model-domain-gap failure, not a bug. It is documented above.

## Decision
**Adopted (DEC-001, user decision 2026-10-08): MoGe-2 replaces VGGT as the geometric branch.**
- Primary: `grid4` (rule-selected).
- Secondary: `normals16` (highest acc@10, lowest dimension).

Whether a geometric branch is kept at all depends on EXP-014 (including its non-geometric control) and on Phase-2 fusion beating DINOv3 alone. No architecture change is made until the user decides.

## Next experiment
EXP-014: DINOv3–MoGe-2 complementarity, the EXP-012 audit repeated.

## Artifact index
- Metrics JSON: `outputs/experiments/EXP-013_moge2_controls/RUN-*/metrics/metrics.json`
- CSV: `outputs/experiments/EXP-013_moge2_controls/summary_by_features.csv`; `run_list.csv`
- TSV: `outputs/experiments/EXP-013_moge2_controls/summary_by_features.tsv`
- Figures: `outputs/experiments/EXP-013_moge2_controls/figures/moge2_maps_examples.jpg` (rows: 2 synthetic, 2 lightbox, 2 sunlamp; columns: crop, normals, mask, depth)
- Predictions: `RUN-*/predictions/predictions_greedy.csv`
- Checkpoint: `RUN-*/checkpoints/best.pt` (gitignored, local)
- Logs: `sweep.log`, `RUN-*/logs/train.log`, `status.json`, `environment.txt`; `outputs/shared_cache/logs/moge2_vitl.log`
