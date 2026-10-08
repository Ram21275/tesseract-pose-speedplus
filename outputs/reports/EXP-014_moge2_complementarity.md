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

A reference control was added on reviewer advice: DINOv3-B `grid4` (EXP-010 predictions, mean 29.9°) as the second branch. It is a second appearance-only model, so it shows how much complementarity *any* second model gives, with no geometric information.

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
| RUN-20261008-161752-seed0 | Completed (MoGe-2 grid4) | 3.9 GB | outputs/experiments/EXP-014_moge2_complementarity/RUN-20261008-161752-seed0 |
| RUN-20261008-225350-seed0 | Completed (reproduction of RUN-20261008-161752 via the new `configs/branches.yaml` defaults; metrics bit-identical) | – | outputs/experiments/EXP-014_moge2_complementarity/RUN-20261008-225350-seed0 |
| RUN-20261008-162511-seed0 | Completed (**control:** DINOv3-B grid4 as the second branch; appearance only, no geometry) | – | outputs/experiments/EXP-014_moge2_complementarity/RUN-20261008-162511-seed0 |

## Quantitative results
DINOv3-L `grid4` with MoGe-2 `grid4`, mean over 3 seed pairs. Source: `RUN-20261008-161752-seed0/metrics/metrics.csv`.

| domain | DINO mean° | MoGe-2 mean° | Spearman(err) | both <10° | DINO only | MoGe-2 only | neither | oracle-of-two acc@10 | root agree | L2 agree | agree frac (dist<20°) | DINO acc@10 when agree / disagree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| synthetic_val | 27.7 | 39.6 | 0.376 | 0.066 | 0.135 | 0.088 | 0.711 | 0.289 | 0.837 | 0.462 | 0.367 | 0.355 / 0.112 |
| lightbox | 76.1 | 83.3 | 0.329 | 0.003 | 0.033 | 0.017 | 0.947 | 0.053 | 0.567 | 0.121 | 0.067 | 0.133 / 0.030 |
| sunlamp | 79.1 | 83.9 | 0.284 | 0.001 | 0.014 | 0.014 | 0.971 | 0.029 | 0.554 | 0.109 | 0.061 | 0.056 / 0.012 |

DINO-alone acc@10: 0.201 synthetic_val, 0.036 lightbox, 0.015 sunlamp.

**Complementarity on synthetic validation, all second branches.** Mean ± std over the 3 seed pairs. Source: each run's `metrics/complementarity_per_seed.csv`.

| second branch | its mean error | only it solves (<10°) | oracle-of-two acc@10 | Spearman(err) | DINO-L acc@10 agree/disagree ratio | agree fraction |
|---|---:|---:|---:|---:|---:|---:|
| VGGT `l11_cam+mean` (EXP-012) | 48.7° | 0.077 ± 0.007 | 0.278 ± 0.022 | 0.312 ± 0.011 | 2.60 ± 0.31 | 0.298 |
| VGGT `grid4` (EXP-012b) | 49.1° | 0.068 ± 0.009 | 0.269 ± 0.016 | 0.384 ± 0.007 | 2.95 ± 0.17 | 0.297 |
| **MoGe-2 `grid4`** (EXP-014) | 39.6° | 0.088 ± 0.002 | 0.289 ± 0.020 | 0.376 ± 0.019 | 3.18 ± 0.24 | 0.367 |
| *Control: DINOv3-B `grid4`* (no geometry) | 29.9° | 0.107 ± 0.009 | 0.308 ± 0.022 | 0.421 ± 0.001 | 3.67 ± 0.05 | 0.441 |

DINO-L alone: acc@10 0.201.

## Domain-wise results
**Head-to-head with the VGGT audit (EXP-012, primary pick `l11_cam+mean`).** The decision rests on synthetic validation. Real-domain columns are descriptive only.

| metric | VGGT (EXP-012) synth-val | **MoGe-2 (EXP-014) synth-val** | VGGT lightbox / sunlamp | MoGe-2 lightbox / sunlamp |
|---|---:|---:|---|---|
| second-branch mean error | 48.7° | **39.6°** | 101.5 / 106.9 | **83.3 / 83.9** |
| images only the second branch solves (<10°) | 7.7% | **8.8%** | 0.8 / 0.3% | **1.7 / 1.4%** |
| oracle-of-two acc@10 (DINO alone 0.201) | 0.278 | **0.289** | 0.045 / 0.018 | **0.053 / 0.029** |
| error correlation (Spearman) | **0.312** (less correlated) | 0.376 | 0.276 / 0.166 | 0.329 / 0.284 |
| fraction of images where the branches agree | 29.8% | **36.7%** | 4.3 / 1.6% | 6.7 / 6.1% |
| DINO acc@10 when agree / disagree | 0.354 / 0.137 (2.6×; VGGT grid4: 3.0×) | 0.355 / 0.112 (3.2×) | 0.134 / 0.032 | 0.133 / 0.030 |

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
The control is DINO alone. With a perfect per-image selector (an upper bound, not an achievable fusion), acc@10 on synthetic_val rises from 0.201 to:
- 0.278 ± 0.022 with VGGT;
- 0.289 ± 0.020 with MoGe-2;
- 0.308 ± 0.022 with the appearance-only control, DINOv3-B.

## Interpretation
On synthetic validation:
1. **MoGe-2 and VGGT are about equally complementary to DINOv3-L.** Their oracle bounds (0.289 vs 0.278) and "only it solves" rates (8.8% vs 7.7%) differ by about one seed standard deviation or less. MoGe-2 is **clearly the better single branch** (39.6° vs 48.7°, std ≤ 0.8°).
2. **Neither geometric branch shows complementarity beyond what any second model gives.** The appearance-only DINOv3-B control gives a *higher* oracle bound (0.308), more images it alone solves (10.7%) and a stronger agreement signal (3.67×). Across the four second branches, complementarity tracks the second model's own accuracy, not whether it carries geometry. At this subset scale, these audits do not show that a geometric branch adds *geometric* information that DINOv3 lacks.
3. **Agreement remains a useful reliability cue.** DINO-L is correct 3.2× more often when it agrees with MoGe-2. Against VGGT `grid4` the ratio is 3.0×, so like for like the gap to VGGT is small. A second DINOv3 model gives the same cue (3.7×) at about one tenth of the extraction cost, though extraction timings on the shared GPU are only indicative.
4. **Off-domain** (descriptive), all second branches are near chance on most real images (94–97% solved by neither).

## Recommendation (awaiting user decision)
- **If a geometric branch is kept, use MoGe-2 rather than VGGT.** It is the better single branch, about equally complementary, and has a slightly stronger agreement signal.
- **Whether to keep a geometric branch at all is open.** This subset gives no evidence that it adds geometric information beyond a second appearance model. That should be re-tested on the full synthetic training split, and in Phase 2 against a matched non-geometric second branch (e.g. DINOv3-B), before it is fixed in the design.
- Replacing VGGT changes the project's stated DINOv3–VGGT architecture (ground rules §2.5, §16.7), so **no swap is made until the user decides.**

## Failure analysis
None. The run completed on the first attempt within the memory caps.

## Decision
Modify. MoGe-2 replaces VGGT as the geometric branch (DEC-001, user decision 2026-10-08). Whether a geometric branch is retained at all is deferred to the full-data repeat and the Phase-2 matched non-geometric control.

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
