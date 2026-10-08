# EXP-014: DINOv3–MoGe-2 complementarity (repeat of EXP-012 with MoGe-2)

## Status
Completed (subset v1)

## Research question
Does MoGe-2 add information beyond DINOv3? Is it more complementary than VGGT was in EXP-012?

## Hypothesis
MoGe-2 is a single-image geometry model, and its encoder is DINOv2, not DINOv3. It may fail on different images than DINOv3, and its agreement with DINOv3 may mark reliable predictions. Because both encoders are DINO-family models, the overlap could also be higher than VGGT's.

## Component under test
Fusion (pre-fusion audit)

## Track
Training-free analysis. No new training; it reuses EXP-010 and EXP-013 predictions.

## Fixed setup
- **Dataset and exact split:** subset v1 (hash `bbdb3e2e3f5b70a1`). Scored on synthetic_val (1,500), lightbox (1,000) and sunlamp (1,000).
- **DINOv3:** ViT-L/16 `grid4` (EXP-010 best).
- **Second branch:** MoGe-2 `grid4`, the EXP-013 pick by the synthetic-val mean rule.
- **Predictor:** EXP-010/013 checkpoints via their saved greedy predictions. Seed pairs (0,0), (1,1), (2,2), with metrics averaged over the 3 pairs.
- **Tesseract depth and codebook size:** L = 5.
- **Input resolution and preprocessing:** as in EXP-010/013.
- **Seeds:** 0, 1, 2 (paired).
- **Compute device:** CPU.
- **Git commit:** `c8f2d94` (clean tree).
- **Safety:** run alone under `scripts/guarded.sh 24` plus memguard. Peak RSS 3.9 GB.
- **Flow base distribution, solver, NFE and endpoint samples:** n/a.

## Changed variable
Second branch: MoGe-2 `grid4` instead of VGGT `l11_cam+mean` (EXP-012).

## Method
Identical to EXP-012 (same script, `scripts/exp012_complementarity.py`, now parameterized by the second branch). The band-energy audit uses MoGe-2's 74×74 surface-normal map.

## Commands
```
scripts/guarded.sh 24 -- python scripts/exp012_complementarity.py --dino dinov3_vitl16:grid4 \
    --vggt moge2_vitl:grid4 --second-exp EXP-013_moge2_controls --second-map normal \
    --run-exp EXP-014_moge2_complementarity
```

## Runs
| Run ID | Status | Peak RAM | Artifact directory |
|---|---|---:|---|
| RUN-20261008-161752-seed0 | Completed | 3.9 GB | outputs/experiments/EXP-014_moge2_complementarity/RUN-20261008-161752-seed0 |

## Quantitative results
DINOv3-L `grid4` with MoGe-2 `grid4`, mean over 3 seed pairs. Source: `RUN-20261008-161752-seed0/metrics/metrics.csv`.

| domain | DINO mean° | MoGe-2 mean° | Spearman(err) | both <10° | DINO only | MoGe-2 only | neither | oracle-of-two acc@10 | root agree | L2 agree | agree frac (dist<20°) | DINO acc@10 when agree / disagree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| synthetic_val | 27.7 | 39.6 | 0.376 | 0.066 | 0.135 | 0.088 | 0.711 | 0.289 | 0.837 | 0.462 | 0.367 | 0.355 / 0.112 |
| lightbox | 76.1 | 83.3 | 0.329 | 0.003 | 0.033 | 0.017 | 0.947 | 0.053 | 0.567 | 0.121 | 0.067 | 0.133 / 0.030 |
| sunlamp | 79.1 | 83.9 | 0.284 | 0.001 | 0.014 | 0.014 | 0.971 | 0.029 | 0.554 | 0.109 | 0.061 | 0.056 / 0.012 |

DINO-alone acc@10: 0.201 synthetic_val, 0.036 lightbox, 0.015 sunlamp.

## Domain-wise results
**Head-to-head with the VGGT audit (EXP-012, primary pick `l11_cam+mean`).** The decision rests on synthetic validation. Real-domain columns are descriptive only.

| metric | VGGT (EXP-012) synth-val | **MoGe-2 (EXP-014) synth-val** | VGGT lightbox / sunlamp | MoGe-2 lightbox / sunlamp |
|---|---:|---:|---|---|
| second-branch mean error | 48.7° | **39.6°** | 101.5 / 106.9 | **83.3 / 83.9** |
| images only the second branch solves (<10°) | 7.7% | **8.8%** | 0.8 / 0.3% | **1.7 / 1.4%** |
| oracle-of-two acc@10 (DINO alone 0.201) | 0.278 | **0.289** | 0.045 / 0.018 | **0.053 / 0.029** |
| error correlation (Spearman) | **0.312** (less correlated) | 0.376 | 0.276 / 0.166 | 0.329 / 0.284 |
| fraction of images where the branches agree | 29.8% | **36.7%** | 4.3 / 1.6% | 6.7 / 6.1% |
| DINO acc@10 when agree / disagree | 0.354 / 0.137 (2.6×) | 0.355 / 0.112 (**3.2×**) | 0.134 / 0.032 | 0.133 / 0.030 |

**Object scale**, mean error by GT crop-size tertile (DINO / MoGe-2):
- synthetic_val: small 28.0 / 39.6, medium 24.6 / 36.1, large 30.4 / 43.1;
- lightbox: small 77.2 / 89.6, medium 75.9 / 83.5, large 75.2 / 76.9;
- sunlamp: small 85.0 / 93.1, medium 76.3 / 81.1, large 76.0 / 77.3.

**Feature drift** (mean shift / spread):

| features | synthetic_val | lightbox | sunlamp |
|---|---:|---:|---:|
| DINOv3 grid4 | 0.029 | 0.479 | 0.927 |
| MoGe-2 grid4 | 0.031 | 0.646 | 0.880 |
| VGGT l11, from EXP-012 | 0.030 | 0.539 | 0.956 |

**Band energy** of the MoGe-2 normal map (74×74), low / mid / high:
- synthetic 0.832 / 0.101 / 0.068;
- lightbox 0.840 / 0.098 / 0.062;
- sunlamp 0.851 / 0.090 / 0.058.

The DINOv3 token map is as in EXP-012. The two maps are different quantities, so they are not compared across backbones.

## Qualitative results
See the EXP-013 figure (`outputs/experiments/EXP-013_moge2_controls/figures/moge2_maps_examples.jpg`). It shows MoGe-2's foreground mask covering almost the whole crop on real images.

## Resource results
- Parameters: n/a (analysis)
- Trainable parameters: 0
- Peak memory: 3.9 GB RSS
- Mean/median latency: n/a
- Tesseract nodes visited: n/a
- Flow sampling latency and NFE: n/a
- Parent-child mass consistency error: n/a
- Credible-region coverage: n/a

## Comparison with control
The control is DINO alone. With a perfect per-image selector (an upper bound, not an achievable fusion), adding MoGe-2 raises acc@10 by:
- synthetic_val: +8.8 points (0.201 → 0.289), against +7.7 points for VGGT;
- lightbox and sunlamp: +1.7 and +1.4 points (descriptive), against +0.9 and +0.3 for VGGT.

## Interpretation
On synthetic validation:
1. **MoGe-2 is at least as complementary as VGGT.** It solves more images that DINO misses (8.8% vs 7.7%) and gives a higher oracle-of-two bound (0.289 vs 0.278), even though its errors are slightly more correlated with DINO's (0.38 vs 0.31). That higher correlation fits its DINOv2 encoder overlapping somewhat more with DINOv3.
2. **The agreement signal is stronger and covers more images.** When DINO and MoGe-2 agree, DINO is correct below 10° 3.2× as often as when they disagree (VGGT: 2.6×). The agreement set is also larger (36.7% vs 29.8% of images). This is the most direct use of a second branch, both for the Phase-2 reliability-weighted fusion (EXP-021 "agreement") and for Phase-5 adaptive depth.
3. **Off-domain** (descriptive), MoGe-2 is better than VGGT but both branches remain near chance on most real images (95–97% solved by neither). Drift to lightbox is a little larger for MoGe-2 (0.65 vs 0.54). The real-domain mask failure documented in EXP-013 is a likely contributor; this is not tested.

## Failure analysis
None. The run completed on the first attempt within the memory caps.

## Decision
**Replace VGGT with MoGe-2 as the geometric branch carried into Phase 2.** The deciding evidence is synthetic validation only:
- MoGe-2 is better as a single branch (39.6° vs 48.7° mean);
- it is at least as complementary (oracle 0.289 vs 0.278);
- it gives a stronger, wider agreement signal (3.2× on 36.7% of images, vs 2.6× on 29.8%);
- it is about 2× cheaper to extract than VGGT (128 vs 246 ms/img).

DINOv3-L `grid4` remains the primary branch. If no training-free fusion of DINOv3 and MoGe-2 beats DINOv3 alone on synthetic validation in Phase 2, the geometric branch is dropped, or kept only as an agreement and uncertainty cue.

## Next experiment
Either:
- the full-data repeat of the Phase-1 controls (DINOv3 + MoGe-2), or
- Phase 2, EXP-020, with DINOv3 + MoGe-2.

## Artifact index
- Metrics JSON: `RUN-20261008-161752-seed0/metrics/{metrics,complementarity_per_seed,scale,drift,band_energy}.json`
- CSV: same directory, `*.csv`
- TSV: same directory, `*.tsv`
- Tables: `RUN-20261008-161752-seed0/tables/*.md`
- Figures: none (see EXP-013)
- Predictions: source runs listed in `RUN-20261008-161752-seed0/source_runs.txt`
- Checkpoint: n/a
- Logs: `status.json`, `environment.txt`
