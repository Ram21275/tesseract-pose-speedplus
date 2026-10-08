# EXP-011: VGGT-token and VGGT-geometry controls

## Status
Completed (subset v1 sanity run)

## Research question
With the predictor, budget, subset and seeds held equal to EXP-010, how much rotation information do frozen VGGT aggregator tokens, depth/confidence summaries and point-map summaries carry?

## Hypothesis
VGGT's geometry-trained tokens and its depth and point-map outputs carry 3D shape information that is useful for rotation. They may also be more robust to illumination than DINOv3, since glare changes appearance more than geometry.

## Component under test
Fusion input representation: single branch (VGGT)

## Track
Trainable predictor. The backbone is frozen, and there is no fusion.

## Fixed setup
- **Dataset and exact split:** subset v1, identical to EXP-010 (hash `bbdb3e2e3f5b70a1`).
  - Train: 6,000 synthetic/train images.
  - Selection: 1,500 synthetic/validation images.
  - Scored only: 1,000 lightbox and 1,000 sunlamp images.
- **DINOv3 variant/layer:** n/a.
- **VGGT:** `facebook/VGGT-1B` (public; code at GitHub commit `a288dd0`).
  - Single frame (S = 1), run as a full forward pass with a hook on the aggregator.
  - Aggregator layers 4, 11, 17, 23. Each is 2048-d, the frame-attention and global-attention halves concatenated.
  - Camera token: index 0. Patch tokens: 37×37.
  - DPT depth, depth confidence, world points and point confidence, area-pooled to 74×74.
  - With S = 1 the camera head carries no relative-pose signal, so it is not used.
- **Predictor:** identical to EXP-010: hierarchical MLP, hidden 512, 100 epochs, AdamW lr 1e-3 / wd 0.05, batch 256, hard labels, selection on synthetic-val mean error.
- **Tesseract depth and codebook size:** L = 5 (131,072).
- **Input resolution and preprocessing:**
  - The same GT square crop as EXP-010, resized to 518×518, grayscale replicated to RGB, values in [0, 1]. VGGT normalizes internally.
  - bf16 autocast for the aggregator; the heads run in fp32, as in VGGT's own forward pass.
- **Seeds:** 0, 1, 2.
- **Compute device:** RTX 6000 Ada (shared).
- **Git commit:** extraction at `89d251a`; all 27 probe runs at `3361c5e` (the training code is identical to EXP-010).
- **Flow base distribution, solver, NFE and endpoint samples:** n/a.

## Changed variable
VGGT feature source:
- `l{4,11,17,23}_cam+mean`: camera token + mean patch token per layer;
- `l23_cam`, `l23_mean`;
- `grid4`: the layer-23 patch grid, pooled to 8×8 at extraction and then 4×4;
- `depthconf16`: depth divided by the per-image median depth, plus log depth-confidence, both pooled to 16×16;
- `points16`: the point map centred by its confidence-weighted mean and scaled to unit confidence-weighted RMS, plus log point-confidence, pooled to 16×16.

## Method
Same as EXP-010: cached features, then 3 seeds of the common predictor, then greedy and beam decoding without the GT path.

## Commands
```
python scripts/extract_features.py --backbone vggt_1b --batch 8
scripts/run_phase1_sweeps.sh        # EXP-011 section
python scripts/summarize_probe.py --exp EXP-011_vggt_controls
```

## Runs
All 27 runs completed (`outputs/experiments/EXP-011_vggt_controls/run_list.csv`). Mean runtime was 36–58 s per run.

| Features | Feature dim | Trainable params | Runs (RUN-20261005-HHMMSS, seeds 0/1/2) |
|---|---:|---:|---|
| l4_cam+mean | 4096 | 2,710,284 | 185228 / 185305 / 185351 |
| l11_cam+mean | 4096 | 2,710,284 | 185423 / 185512 / 185545 |
| l17_cam+mean | 4096 | 2,710,284 | 185630 / 185714 / 185749 |
| l23_cam+mean | 4096 | 2,710,284 | 185840 / 185909 / 190000 |
| l23_cam | 2048 | 1,657,612 | 190037 / 190118 / 190205 |
| l23_mean | 2048 | 1,657,612 | 190236 / 190328 / 190357 |
| grid4 | 32768 | 17,447,692 | 190446 / 190555 / 190651 |
| depthconf16 | 512 | 868,108 | 190745 / 190836 / 190912 |
| points16 | 1024 | 1,131,276 | 190955 / 191041 / 191118 |

## Quantitative results
Greedy decoding on synthetic validation, mean ± std over 3 seeds. Errors are in degrees. Source: `outputs/experiments/EXP-011_vggt_controls/summary_by_features.csv`.

| features | synthetic_val mean | synthetic_val median | acc@20 | root acc | L3 path acc |
|---|---|---|---|---|---|
| l4_cam+mean | 54.78 ± 1.21 | 29.47 ± 1.36 | 0.356 ± 0.021 | 0.789 ± 0.009 | 0.148 ± 0.002 |
| **l11_cam+mean** | **48.70 ± 0.65** | **25.35 ± 0.56** | **0.397 ± 0.006** | **0.813 ± 0.007** | 0.169 ± 0.012 |
| l17_cam+mean | 66.18 ± 0.39 | 44.28 ± 1.54 | 0.267 ± 0.013 | 0.711 ± 0.006 | 0.105 ± 0.009 |
| l23_cam+mean | 69.52 ± 0.97 | 50.93 ± 1.67 | 0.254 ± 0.013 | 0.665 ± 0.010 | 0.102 ± 0.002 |
| l23_cam | 72.62 ± 1.14 | 57.33 ± 1.29 | 0.235 ± 0.022 | 0.646 ± 0.009 | 0.099 ± 0.006 |
| l23_mean | 69.90 ± 0.87 | 51.71 ± 1.83 | 0.268 ± 0.005 | 0.669 ± 0.012 | 0.110 ± 0.009 |
| grid4 | 49.12 ± 0.74 | 27.06 ± 0.16 | 0.380 ± 0.003 | 0.799 ± 0.005 | 0.167 ± 0.007 |
| depthconf16 | 66.14 ± 0.72 | 36.18 ± 0.49 | 0.358 ± 0.007 | 0.643 ± 0.002 | 0.157 ± 0.006 |
| points16 | 67.23 ± 0.82 | 38.61 ± 0.95 | 0.330 ± 0.011 | 0.648 ± 0.011 | 0.143 ± 0.003 |

For reference, the best DINOv3 configuration (EXP-010, `vitl16:grid4`) scores 27.68 ± 0.80 mean, 17.20 ± 0.47 median and 0.585 acc@20 on synthetic val.

## Domain-wise results
Greedy, mean over 3 seeds.

| features | lightbox mean / median | lightbox acc@20 | lightbox root | sunlamp mean / median | sunlamp acc@20 | sunlamp root |
|---|---|---|---|---|---|---|
| l4_cam+mean | 104.9 / 110.5 | 0.057 | 0.441 | 110.1 / 113.8 | 0.028 | 0.419 |
| l11_cam+mean | 101.5 / 104.7 | 0.054 | 0.474 | 106.9 / 109.8 | 0.031 | 0.457 |
| l17_cam+mean | 103.3 / 106.6 | 0.050 | 0.467 | 107.7 / 111.9 | 0.025 | 0.444 |
| l23_cam+mean | 104.5 / 108.6 | 0.047 | 0.448 | 112.0 / 118.1 | 0.024 | 0.410 |
| l23_cam | 107.4 / 113.6 | 0.044 | 0.427 | 110.7 / 117.8 | 0.033 | 0.408 |
| l23_mean | 104.6 / 109.0 | 0.051 | 0.444 | 114.9 / 122.9 | 0.019 | 0.362 |
| grid4 | 86.7 / 84.3 | 0.103 | 0.568 | 92.5 / 91.6 | 0.066 | 0.551 |
| depthconf16 | 97.2 / 103.2 | **0.132** | 0.450 | 99.6 / 106.2 | **0.112** | 0.449 |
| points16 | 103.1 / 111.0 | 0.095 | 0.429 | 106.4 / 116.8 | 0.069 | 0.417 |

Reference, DINOv3 `vitl16:grid4`: lightbox 76.1 / 62.2, acc@20 0.145, root 0.612; sunlamp 79.1 / 70.3, acc@20 0.109, root 0.629.

## Qualitative results
Per-image greedy predictions are in `RUN-*/predictions/predictions_greedy.csv`. Error by object size, posterior agreement and the frequency audit are in EXP-012.

## Resource results
- **Parameters:** VGGT-1B, frozen; `assert_frozen` passed on every batch.
- **Trainable parameters:** 0.87M (depthconf16) to 17.4M (grid4).
- **Peak memory:** extraction peak GPU 7.8 GB at batch 8; probe peak GPU < 1 GB.
- **Latency:** extraction 246 ms/img on the shared GPU, including JPEG decode and crop. That is about 17× DINOv3-L's 13.5 ms/img.
- **Cache:** 3.3 GB for 9,500 images.
- **Tesseract nodes visited:** greedy 44.
- **Flow sampling latency and NFE:** n/a.
- **Parent-child mass consistency error:** n/a.
- **Credible-region coverage:** n/a.

## Comparison with control
Against the strongest DINOv3 branch, under the same predictor, budget, subset and seeds, the best VGGT tokens (`l11_cam+mean` and `grid4`, tied within noise) are clearly worse in every domain:
- synthetic-val mean 48.7° vs 27.7°;
- lightbox median 84–105° vs 62°;
- sunlamp median 92–110° vs 70°.

## Interpretation
Observations:
1. **Middle layers are best for VGGT.** Layer 11 beats layer 4, while layers 17 and 23 degrade sharply (median 44–57°). This is the opposite of DINOv3, where the last layer is best. One possible reason, not tested: VGGT's late layers specialize for its multi-view heads, which a single frame does not exercise.
2. **Spatial pooling (`grid4`) matters most off-domain.** On synthetic val it matches the best vector (49.1° vs 48.7° mean), but it is far better on lightbox (median 84° vs 105°) and sunlamp (92° vs 110°).
3. **Dense geometry (depth/confidence and point maps) is weaker on synthetic** (mean about 66–67°). However, `depthconf16` has the **highest real-domain acc@20 of any VGGT feature**, 0.132 lightbox and 0.112 sunlamp, comparable to DINOv3's 0.145 and 0.109, from only 512 numbers. That small, illumination-insensitive geometry summary is the one VGGT signal worth carrying into the complementarity audit and the Phase-2 reliability signals.
4. **VGGT costs about 17× more extraction time than DINOv3-L for a weaker branch.**

## Failure analysis
None of the runs failed. VGGT's weak real-domain performance is measured, not a bug: the same crops and conventions were used as in EXP-010.

## Decision
Keep VGGT `l11_cam+mean` (primary, selected by the synthetic-val mean rule) and `grid4` (statistically tied) as the VGGT representatives for EXP-012. Flag `depthconf16` as a candidate reliability or geometry signal. Whether VGGT stays at all is decided by EXP-012 and the Phase-1 gate.

## Next experiment
EXP-012: complementarity and frequency audit.

## Artifact index
- Metrics JSON: `outputs/experiments/EXP-011_vggt_controls/RUN-*/metrics/metrics.json`
- CSV: `outputs/experiments/EXP-011_vggt_controls/summary_by_features.csv`; `run_list.csv`
- TSV: `outputs/experiments/EXP-011_vggt_controls/summary_by_features.tsv`
- Figures: none yet
- Predictions: `RUN-*/predictions/predictions_greedy.csv`
- Checkpoint: `RUN-*/checkpoints/best.pt` (gitignored, local)
- Logs: `RUN-*/logs/train.log`, `status.json`, `environment.txt`, `feature_source.json`; `sweeps.log`; `outputs/shared_cache/logs/vggt_1b.log`
