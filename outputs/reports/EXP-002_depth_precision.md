# EXP-002: Tesseract fixed-depth precision (L = 1–5)

## Status
Completed

## Research question
How precise is the Tesseract grid at each depth, how uniform is it, and what does each depth cost?

## Hypothesis
Error roughly halves with each level. L∞ cells are not equal-volume on SO(3), so the grid is measurably non-uniform.

## Component under test
Tesseract grid

## Track
Training-free (no learning)

## Fixed setup
- Dataset and exact split: 2,000,000 Haar-uniform rotations (seed 0), plus all 69,491 SPEED+ GT labels
- DINOv3 variant/layer: n/a
- VGGT variant/output/layer: n/a
- Predictor: n/a
- Tesseract depth and codebook size: L = 1…5 (32 … 131,072 leaves)
- Input resolution and preprocessing: n/a
- Seeds: 0
- Compute device: RTX 6000 Ada (nearest-neighbour search), CPU otherwise
- Git commit: `9586198` (clean)
- Flow base distribution, solver, NFE and endpoint samples, if applicable: n/a

## Changed variable
Depth L.

## Method
Quantities measured, by column:
- **Own-cell error** (`haar_own_*`, `gt_own_*`): distance from a rotation to the centre of its own encoded cell. This is what the tree decoder returns.
- **Exact worst-case own-cell error** (`exact_own_cell_max`): the maximum over all cells of the centre-to-corner distance. It is exact because the angle to a fixed direction is quasiconvex on a convex cell in the projected cube, so the maximum is reached at a corner.
- **Nearest-centre covering error** (`haar_nearest_*`): the distance to the nearest centre in any cell. It is measured on 1M samples, so its maximum is a lower bound on the true covering radius.
- **Exact relative cell volume:** the L∞ chart map has density (1+|u|²)⁻² on S³, integrated per cell with 6³-point Gauss–Legendre quadrature. Haar occupancy is reported alongside as an empirical check.
- **Centre spacing:** each centre's nearest-neighbour distance.
- **Cost:** codebook memory and search cost, as nodes scored by greedy tree descent versus flat search.

## Commands
```
python scripts/exp002_depth_precision.py --n-haar 2000000 --max-depth 5 --seed 0
```

## Runs
| Run ID | Seed | Status | Runtime | Artifact directory |
|---|---:|---|---:|---|
| RUN-20261005-173345-seed0 | 0 | Completed (tracked file modified at run time) | 81 s | outputs/experiments/EXP-002_depth_precision/RUN-20261005-173345-seed0 |
| RUN-20261005-174002-seed0 | 0 | Completed (clean commit; bit-identical metrics) | 85 s | outputs/experiments/EXP-002_depth_precision/RUN-20261005-174002-seed0 |

## Quantitative results
Source: `metrics/metrics.csv`. All errors are in degrees.

| L | cells | own-cell mean | own-cell median | own-cell p95 | own-cell max (exact) | nearest mean | nearest max (sampled) | GT own-cell mean / p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 32 | 39.27 | 39.84 | 59.63 | 81.79 | 39.28 | 81.17 | 39.24 / 59.59 |
| 2 | 256 | 19.95 | 19.85 | 31.44 | 46.83 | 19.64 | 46.21 | 19.98 / 31.42 |
| 3 | 2,048 | 10.01 | 9.95 | 15.83 | 24.43 | 9.88 | 23.81 | 10.02 / 15.84 |
| 4 | 16,384 | 5.01 | 4.98 | 7.92 | 12.36 | 4.95 | 11.81 | 5.02 / 7.92 |
| 5 | 131,072 | 2.51 | 2.49 | 3.97 | 6.20 | 2.48 | 5.86 | 2.51 / 3.97 |

Uniformity and cost:

| L | cell volume max/min (exact) | volume CV (exact) | Haar occupancy CV | nearest centre ≠ own cell | centre spacing mean / min / max | spacing CV | codebook (fp32) | greedy nodes | flat nodes |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 1.00 | 0.000 | 0.004 | 0.0% | 62.0 / 62.0 / 62.0 | 0.00 | <0.01 MB | 12 | 32 |
| 2 | 4.71 | 0.494 | 0.494 | 10.3% | 30.6 / 24.8 / 44.0 | 0.18 | <0.01 MB | 20 | 256 |
| 3 | 9.62 | 0.548 | 0.550 | 10.6% | 15.8 / 11.2 / 26.5 | 0.21 | 0.03 MB | 28 | 2,048 |
| 4 | 12.81 | 0.562 | 0.569 | 10.4% | 8.3 / 5.3 / 14.0 | 0.21 | 0.25 MB | 36 | 16,384 |
| 5 | 14.44 | 0.565 | 0.620 | 10.4% | 4.3 / 2.6 / 7.1 | 0.21 | 2.0 MB | 44 | 131,072 |

At L1–L4, the empirical Haar occupancy CV matches the exact volume CV (correlation ≥ 0.987). At L5 the Haar occupancy CV is inflated by Poisson noise, at about 15 samples per cell.

## Domain-wise results
The GT own-cell error statistics match the Haar statistics to within 0.03° at every depth, because SPEED+ rotations are close to uniform. They are effectively identical across domains (EXP-001: L5 mean 2.51–2.53°).

## Qualitative results
`figures/error_vs_depth.png`: own-cell mean, p95 and exact max, and the sampled nearest-centre max, all against depth.

## Resource results
- Parameters: 0
- Trainable parameters: 0
- Peak memory: see `status.json`
- Mean/median latency: n/a
- Tesseract nodes visited: greedy 4 + 8L, i.e. 44 at L5; flat search is 131,072
- Flow sampling latency and NFE, if applicable: n/a
- Parent-child mass consistency error, if applicable: n/a
- Credible-region coverage, if applicable: n/a

## Comparison with control
See EXP-003 for matched-size Hopf and Super-Fibonacci grids.

## Interpretation
- **Error halves per level, as expected.** At L5 the mean is 2.51°, p95 3.97° and the exact worst case 6.20°.
- **The grid is markedly non-uniform.** Cell volumes differ by up to 14.4× at L5, approaching the analytic limit of 16× (density 1 at the chart centre versus 1/16 at cube corners). Volume CV is about 0.56.
  - Small cells sit near the cube corners (chart seams). Large cells sit near the chart axes.
  - This matters for classification priors and soft targets: a uniform prior over leaves is not a uniform prior over rotations.
- **The decoded cell is not the nearest centre about 10% of the time.** In those cases the decoded centre is not the closest grid hypothesis. Beam search, or a local nearest-centre or residual step, can recover part of this. The cost is small: mean 2.51° versus 2.48° at L5.

## Failure analysis
None.

## Decision
Keep. Phase 0 gate: **L_max = 5** (131,072 cells; 2.5° mean, 4.0° p95, 6.2° worst-case quantization).
- L6 would give about 1.25° mean, but with 1,048,576 cells against 47,966 training images.
- Sub-L5 precision is therefore delegated to a continuous refinement: the tangent residual in EXP-035/040, or flow in Phase 4.

## Next experiment
EXP-003: matched comparison with Hopf and a near-uniform grid.

## Artifact index
- Metrics JSON: `outputs/experiments/EXP-002_depth_precision/RUN-20261005-174002-seed0/metrics/metrics.json`
- CSV: same directory, `metrics.csv`
- TSV: same directory, `metrics.tsv`
- Tables: `.../tables/depth_table.md`
- Figures: `.../figures/error_vs_depth.png`
- Predictions: n/a
- Checkpoint, if any: n/a
- Logs: `status.json`, `environment.txt`, `command.txt`
