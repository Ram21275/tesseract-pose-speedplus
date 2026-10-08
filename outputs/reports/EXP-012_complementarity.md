# EXP-012: DINOv3–VGGT complementarity and frequency audit

## Status
Completed (subset v1). The first attempts failed, one of them taking the machine down; see Failure analysis.


## Keep score
**2 / 5: Weak.** Some complementarity on synthetic data and a usable agreement cue, but almost none on real images. Superseded by EXP-014.

_Keep score rubric (0–5): 5 strong support: pre-registered goal met with a large effect, adopt. 4 support: goal met, carry forward. 3 partial: mixed or one-sided evidence, keep for further testing. 2 weak: goal not met but an informative side result. 1 negative: fails, reject, kept only as a recorded result. 0 inconclusive or not run._

## Research question
Does VGGT add information beyond DINOv3? Specifically: do the two branches fail on different images, does their agreement predict correctness, and how do they behave under illumination (sunlamp), object scale and domain shift?

## Hypothesis
VGGT's geometric evidence fails on different images than DINOv3's appearance evidence. If so, the errors will be weakly correlated, one branch will solve images the other misses, and agreement between the branches will mark reliable predictions.

## Component under test
Fusion (pre-fusion audit)

## Track
Training-free analysis. No new training; it reuses EXP-010 and EXP-011 predictions.

## Fixed setup
- **Dataset and exact split:** subset v1 (hash `bbdb3e2e3f5b70a1`). Scored on synthetic_val (1,500), lightbox (1,000) and sunlamp (1,000).
- **DINOv3:** ViT-L/16 `grid4`, the best EXP-010 branch.
- **VGGT:**
  - primary: VGGT-1B `l11_cam+mean`, the EXP-011 pick by the synthetic-val mean rule;
  - secondary: `grid4`, statistically tied with the primary.
- **Predictor:** the EXP-010/011 checkpoints, using their saved greedy predictions. Seeds 0, 1, 2 are paired DINO-seed-s with VGGT-seed-s, and metrics are averaged over the 3 pairs.
- **Tesseract depth and codebook size:** L = 5.
- **Input resolution and preprocessing:** as in EXP-010/011.
- **Seeds:** 0, 1, 2 (paired).
- **Compute device:** CPU.
- **Git commit:** `22ee2d3` (clean code tree) for both completed runs, the memory-safe version (chunked FFT, memguard, cgroup cap of 24 GB).
- **Flow base distribution, solver, NFE and endpoint samples:** n/a.

## Changed variable
VGGT representative: `l11_cam+mean` (primary) versus `grid4` (secondary).

## Method
1. **Error coupling.** Spearman correlation of per-image errors. The fraction of images solved below 10° by both branches, by DINO only, by VGGT only, or by neither. Oracle-of-two acc@10: the upper bound for a *perfect* per-image selector, which no real fusion reaches.
2. **Posterior agreement.**
   - Root-chart agreement, and level-2 prefix agreement.
   - Prediction–prediction geodesic distance.
   - Each branch's acc@10 when the predictions agree (distance < 20°) versus when they disagree.
3. **Object scale.** Error by GT crop-size tertile within each domain.
4. **Feature drift (label-free).** ‖mean shift from synthetic-train‖ divided by the synthetic-train spread, in train-standardized feature space.
5. **Spatial band energy.**
   - Fraction of FFT energy in low (r < 0.25 Nyquist), mid and high bands for DINOv3's 16×16 last-layer patch-token map and VGGT's 74×74 depth map, per domain.
   - These are different quantities on different grids. They are comparable across domains within a backbone, not across backbones.

## Commands
```
# explicit form (defaults changed to MoGe-2 after DEC-001)
scripts/guarded.sh 24 -- python scripts/exp012_complementarity.py --dino dinov3_vitl16:grid4 --vggt vggt_1b:l11_cam+mean \
    --second-exp EXP-011_vggt_controls --second-map depth --run-exp EXP-012_complementarity
scripts/guarded.sh 24 -- python scripts/exp012_complementarity.py --dino dinov3_vitl16:grid4 --vggt vggt_1b:grid4 \
    --second-exp EXP-011_vggt_controls --second-map depth --run-exp EXP-012_complementarity
```

## Runs
| Run ID | VGGT pick | Status | Runtime | Peak RAM | Artifact directory |
|---|---|---|---:|---:|---|
| RUN-20261005-191223-seed0 | l11_cam+mean | Failed: Tesseract path column read as int (`.str` accessor) | <5 s | – | outputs/experiments/EXP-012_complementarity/RUN-20261005-191223-seed0 |
| RUN-20261005-191226-seed0 | grid4 | Failed: same bug | <5 s | – | …/RUN-20261005-191226-seed0 |
| RUN-20261005-191236-seed0 | l11_cam+mean | **Failed: died with the machine.** Most likely an out-of-memory in the band-energy FFT. | – | est. ~100 GB | …/RUN-20261005-191236-seed0 |
| RUN-20261008-111426-seed0 | l11_cam+mean (primary) | Completed | 74 s | 3.1 GB | …/RUN-20261008-111426-seed0 |
| RUN-20261008-111542-seed0 | grid4 (secondary) | Completed | – | 5.9 GB | …/RUN-20261008-111542-seed0 |

## Quantitative results
Primary pair, DINOv3 `vitl16:grid4` with VGGT `l11_cam+mean`, mean over 3 seed pairs. Source: `RUN-20261008-111426-seed0/metrics/metrics.csv`.

| domain | DINO mean° | VGGT mean° | Spearman(err) | both <10° | DINO only | VGGT only | neither | oracle-of-two acc@10 | root agree | L2 agree | DINO acc@10 when agree / disagree | VGGT acc@10 when disagree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| synthetic_val | 27.7 | 48.7 | 0.312 | 0.045 | 0.156 | 0.077 | 0.722 | 0.278 | 0.808 | 0.386 | 0.354 / 0.137 | 0.041 |
| lightbox | 76.1 | 101.5 | 0.276 | 0.002 | 0.034 | 0.008 | 0.955 | 0.045 | 0.520 | 0.076 | 0.134 / 0.032 | 0.008 |
| sunlamp | 79.1 | 106.9 | 0.166 | 0.000 | 0.015 | 0.003 | 0.982 | 0.018 | 0.504 | 0.043 | 0.021 / 0.015 | 0.002 |

DINO-alone acc@10 for reference: 0.201 synthetic_val, 0.036 lightbox, 0.015 sunlamp.

The predictions agree (distance < 20°) on 29.8% of synthetic_val, 4.3% of lightbox and 1.6% of sunlamp images.

Secondary pair, VGGT `grid4` (`RUN-20261008-111542-seed0`):

| domain | Spearman(err) | DINO only <10° | VGGT only <10° | oracle-of-two acc@10 | root agree | DINO acc@10 when agree / disagree |
|---|---:|---:|---:|---:|---:|---|
| synthetic_val | 0.384 | 0.153 | 0.068 | 0.269 | 0.801 | 0.376 / 0.127 |
| lightbox | 0.298 | 0.034 | 0.025 | 0.061 | 0.544 | 0.159 / 0.028 |
| sunlamp | 0.144 | 0.014 | 0.012 | 0.027 | 0.472 | 0.063 / 0.013 |

## Domain-wise results
**Object scale**, primary pair: mean error in degrees by GT crop-size tertile.

| domain | small (≈520 px) DINO / VGGT | medium (≈740 px) | large (≈1230 px) |
|---|---|---|---|
| synthetic_val | 28.0 / 50.2 | 24.6 / 46.1 | 30.4 / 49.8 |
| lightbox | 77.2 / 108.9 | 75.9 / 102.2 | 75.2 / 93.3 |
| sunlamp | 85.0 / 109.7 | 76.3 / 106.9 | 76.0 / 103.9 |

**Feature drift** (mean shift / spread; synthetic_val ≈ 0.03 for both backbones):

| features | lightbox | sunlamp |
|---|---:|---:|
| DINOv3 vitl16:grid4 | 0.479 | 0.927 |
| VGGT l11_cam+mean | 0.539 | 0.956 |

**Band energy** (fraction of spatial FFT energy):

| map | domain | low | mid | high |
|---|---|---:|---:|---:|
| DINOv3 last-layer tokens (16×16) | synthetic_train / val | 0.471 | 0.276 | 0.253 |
| | lightbox | 0.456 | 0.253 | 0.291 |
| | sunlamp | 0.442 | 0.265 | 0.293 |
| VGGT depth (74×74) | synthetic_train / val | 0.939 | 0.042 | 0.019 |
| | lightbox | 0.957 | 0.027 | 0.016 |
| | sunlamp | 0.954 | 0.029 | 0.017 |

## Qualitative results
None rendered yet. Per-image predictions from both branches can be joined on `image_relpath` from the source runs listed in `source_runs.txt`.

## Resource results
- Parameters: n/a (analysis only)
- Trainable parameters: 0
- Peak memory: 3.1 GB and 5.9 GB RSS (the crashed version was estimated at about 100 GB)
- Mean/median latency: n/a
- Tesseract nodes visited: n/a
- Flow sampling latency and NFE: n/a
- Parent-child mass consistency error: n/a
- Credible-region coverage: n/a

## Comparison with control
The control is DINO alone. Under a hypothetical perfect selector, adding VGGT raises acc@10 by these amounts (primary / secondary pick):

| domain | primary | secondary |
|---|---|---|
| synthetic_val | +7.7 points (0.201 → 0.278) | +6.8 points |
| lightbox | +0.9 points | +2.5 points (0.036 → 0.061) |
| sunlamp | +0.3 points | +1.3 points |

These are upper bounds, not achievable fusion gains.

## Interpretation
Observations:
1. **There is complementarity on synthetic data.** The errors are only weakly correlated (Spearman 0.31–0.38). VGGT alone solves 7–8% of synthetic-val images that DINO misses, about a third as many as DINO solves alone.
2. **Off-domain, complementarity mostly vanishes**, because both branches are close to chance there: 95–98% of real images are solved by neither branch.
   - The VGGT `grid4` pick keeps a little more of it (lightbox oracle +2.5 points).
   - The low real-domain correlations (0.14–0.30) mostly reflect near-random errors, not useful independence.
3. **Agreement is a strong reliability signal.** When the two branches agree, DINO is correct below 10° about 2.6× as often on synthetic (0.354 vs 0.137) and 4–6× as often on lightbox (0.134–0.159 vs 0.028–0.032).
   - This directly supports the "DINO–VGGT agreement" reliability signal planned for EXP-021 and the adaptive-depth features in Phase 5.
   - It is the clearest positive evidence for VGGT so far.
4. **VGGT is not more domain-invariant.**
   - Its feature drift to lightbox and sunlamp is as large as DINO's (0.54 / 0.96 vs 0.48 / 0.93).
   - Sunlamp drifts about twice as far as lightbox for both backbones.
5. **Scale.** DINO is roughly scale-insensitive except for small objects under sunlamp (85° vs 76°). VGGT degrades for small objects on lightbox (109° vs 93°).
6. **Frequency.**
   - DINO's token maps shift energy toward high frequencies on the real domains (0.25 → 0.29), consistent with sensor noise and glare texture.
   - VGGT depth is overwhelmingly low-frequency (about 94–96%), and slightly more so on real data.
   - These two maps are different quantities on different grids, so they say nothing about which backbone is "low" or "high" frequency. A like-for-like comparison (e.g. both token grids at 8×8) has not been run.

## Failure analysis
- **Path dtype bug.** The two quick failures came from pandas reading Tesseract path strings as integers. Fixed with `dtype=str`.
- **Machine crash.** The third attempt computed the band-energy FFT over all 9,500 DINOv3-L grids at once: about 10 GB of float32 input and about 40 GB of complex128 output, plus copies. The estimated peak of ~100 GB was on a 123 GB host shared with other users. The host went down shortly after the run started.
  - The kernel log for that boot was truncated, so the OOM is not proven, but it is the only plausible cause in our jobs.
- **Fixes applied:**
  - chunked, float32/complex64 streaming for the FFT, and chunked grid pooling in `featsets`;
  - an in-process watchdog (`tfpose.memguard`: 32 GB per-process RSS, 16 GB system floor) started by every run, which records `failed` with the reason;
  - a hard cgroup cap via `scripts/guarded.sh` (`systemd-run --scope -p MemoryMax=…`, no swap);
  - unit tests that spawn a process, exceed a 0.5 GB limit, and check it is killed.

## Decision
Keep VGGT **conditionally**. The evidence used for this decision is **synthetic validation only**:
- the errors are only weakly correlated (Spearman 0.31);
- VGGT alone solves 7.7% of images DINO misses, and the oracle-of-two upper bound is +7.7 acc@10 points;
- agreement between the branches makes DINO correct below 10° 2.6× more often (0.354 vs 0.137).

Against it: VGGT costs about 17× DINO's extraction time. The lightbox and sunlamp numbers above are reported as description only and were not used to make this decision.

Phase-1 gate recommendation:
- Carry DINOv3-L `grid4` as the primary branch.
- Carry VGGT `l11_cam+mean` (the rule-selected pick; `grid4` is tied on synthetic val) into the Phase-2 training-free fusion tests (EXP-020/021), together with the agreement-based reliability signal.
- If no training-free fusion beats DINO alone on synthetic validation, drop VGGT, or use it only as an uncertainty cue.

All Phase-1 numbers come from a 6,000-image subset in which the probe is data-limited. Repeating EXP-010/011/012 on the full synthetic training split is advisable before committing to the Phase-2 design.

## Next experiment
Either:
- a full-data repeat of the Phase-1 controls (recommended), or
- Phase 2, EXP-020: single-branch controls, equal posterior averaging and product of experts.

## Artifact index
- Metrics JSON: `RUN-20261008-111426-seed0/metrics/{metrics,complementarity_per_seed,scale,drift,band_energy}.json`; same for `RUN-20261008-111542-seed0`
- CSV: same directories, `*.csv`
- TSV: same directories, `*.tsv`
- Tables: `RUN-*/tables/*.md`
- Figures: none
- Predictions: source runs listed in `RUN-*/source_runs.txt`
- Checkpoint: n/a
- Logs: `logs/exp012_rerun.log`, `status.json`, `environment.txt`, `memguard_abort.txt` (only if it triggers)
