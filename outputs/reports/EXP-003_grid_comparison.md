# EXP-003: Tesseract vs Hopf vs near-uniform (Super-Fibonacci) grids

## Status
Completed


## Keep score
**3 / 5: Partial.** Tesseract is less precise per hypothesis than Super-Fibonacci and Hopf. Its only advantage is search cost (44 nodes at L5). Keep, but the Hopf baseline must stay in every comparison.

_Keep score rubric (0–5): 5 strong support: pre-registered goal met with a large effect, adopt. 4 support: goal met, carry forward. 3 partial: mixed or one-sided evidence, keep for further testing. 2 weak: goal not met but an informative side result. 1 negative: fails, reject, kept only as a recorded result. 0 inconclusive or not run._

## Research question
At matched hypothesis counts, and at matched covering radius, how does the Tesseract grid compare with established SO(3) grids in precision, uniformity and search cost?

## Hypothesis
Near-uniform grids need fewer points than Tesseract to reach the same error. Tesseract's advantage, if any, is its cheap 4-root, 8-ary hierarchy.

## Component under test
Tesseract grid

## Track
Training-free (no learning)

## Fixed setup
- Dataset and exact split: 1,000,000 Haar-uniform query rotations (seed 0)
- DINOv3 variant/layer: n/a
- VGGT variant/output/layer: n/a
- Predictor: n/a
- Tesseract depth and codebook size: L1–L5 (32 … 131,072)
- Input resolution and preprocessing: n/a
- Seeds: 0
- Compute device: RTX 6000 Ada (exact nearest-neighbour by |q·g|)
- Git commit: `0115c32` (clean code tree; the Phase 0 code is unchanged since `9586198`)
- Flow base distribution, solver, NFE and endpoint samples, if applicable: n/a

## Changed variable
Rotation grid family.

## Method
**Grids compared:**
- **Tesseract:** this project's grid.
- **Hopf (Yershova et al., IJRR 2010):** HEALPix S² × uniform S¹, with 72·8ʳ points at r = 0…4. HEALPix ring-scheme `pix2ang` is implemented in `src/tfpose/grids.py`.
- **Super-Fibonacci spirals (Alexa, CVPR 2022):** at exactly the Tesseract counts.

**Recorded substitution:** the ground rules name cubochoric as one matched baseline. Super-Fibonacci is used instead, under the clause allowing "another near-uniform SO(3) grid". It yields any N exactly, which makes exact count matching possible. Cubochoric can be added later if a reviewer requires it.

**Matching:**
1. **Exact count:** Tesseract versus Super-Fibonacci at the same N.
2. **Nearest count:** Hopf counts (72·8ʳ) can never equal Tesseract counts (4·8ᴸ); the ratio is always 18 or 2.25. Both neighbouring Hopf levels are reported.
3. **Matched covering radius:** the Super-Fibonacci size needed to equal the Tesseract decoded p95 error. It was estimated from the N^(−1/3) law and then measured.

**Measurements:** nearest-centre error (all grids), the decoded own-cell error for Tesseract, Voronoi occupancy CV, nearest-neighbour spacing CV, and greedy-tree versus flat search cost.

## Commands
```
python scripts/exp003_grid_comparison.py --n-haar 1000000 --seed 0
```

## Runs
| Run ID | Seed | Status | Runtime | Artifact directory |
|---|---:|---|---:|---|
| RUN-20261005-173545-seed0 | 0 | Completed (tracked file modified at run time) | 183 s | outputs/experiments/EXP-003_grid_comparison/RUN-20261005-173545-seed0 |
| RUN-20261005-174130-seed0 | 0 | Completed (clean commit; bit-identical metrics) | 173 s | outputs/experiments/EXP-003_grid_comparison/RUN-20261005-174130-seed0 |

## Quantitative results
Source: `metrics/metrics.csv`, also copied to `outputs/comparisons/rotation_grid_summary.csv`. Errors are in degrees.

| grid | level | N | nearest mean | nearest p95 | nearest max | decoded mean | decoded p95 | decoded max | spacing CV | greedy nodes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Tesseract | L1 | 32 | 39.28 | 59.62 | 81.17 | 39.28 | 59.62 | 81.17 | 0.000 | 12 |
| Super-Fib | – | 32 | 38.77 | 57.20 | 76.36 | – | – | – | 0.175 | flat |
| Hopf | r0 | 72 | 28.36 | 40.58 | 55.71 | – | – | – | 0.014 | 72 |
| Tesseract | L2 | 256 | 19.64 | 30.28 | 46.21 | 19.96 | 31.43 | 46.21 | 0.181 | 20 |
| Super-Fib | – | 256 | 19.17 | 28.15 | 37.47 | – | – | – | 0.066 | flat |
| Hopf | r1 | 576 | 14.10 | 19.90 | 27.50 | – | – | – | 0.028 | 80 |
| Tesseract | L3 | 2,048 | 9.88 | 15.52 | 23.81 | 10.01 | 15.83 | 23.81 | 0.209 | 28 |
| Super-Fib | – | 2,048 | 9.53 | 13.72 | 18.26 | – | – | – | 0.097 | flat |
| Hopf | r2 | 4,608 | 7.04 | 9.90 | 13.53 | – | – | – | 0.063 | 88 |
| Tesseract | L4 | 16,384 | 4.95 | 7.81 | 11.81 | 5.01 | 7.92 | 11.83 | 0.213 | 36 |
| Super-Fib | – | 16,384 | 4.68 | 6.72 | 8.53 | – | – | – | 0.052 | flat |
| Hopf | r3 | 36,864 | 3.52 | 4.95 | 6.67 | – | – | – | 0.064 | 96 |
| Tesseract | L5 | 131,072 | 2.48 | 3.91 | 5.86 | 2.51 | 3.97 | 5.90 | 0.206 | 44 |
| Super-Fib | – | 131,072 | 2.47 | 3.76 | 4.52 | – | – | – | 0.070 | flat |
| Hopf | r4 | 294,912 | 1.76 | 2.47 | 3.24 | – | – | – | 0.062 | 104 |

Matched covering (`metrics/matched_covering.csv`):

| Tesseract level | Tesseract N | Tesseract decoded p95 | Super-Fib N for equal p95 | Super-Fib measured p95 | Super-Fib N / Tesseract N |
|---|---:|---:|---:|---:|---:|
| L1 | 32 | 59.62 | 28 | 59.84 | 0.88 |
| L2 | 256 | 31.43 | 184 | 31.42 | 0.72 |
| L3 | 2,048 | 15.83 | 1,332 | 15.34 | 0.65 |
| L4 | 16,384 | 7.92 | 10,018 | 7.98 | 0.61 |
| L5 | 131,072 | 3.97 | 111,852 | 3.94 | 0.85 |

The L5 ratio is less reliable: the N^(−1/3) estimate was not iterated, and the measured p95 (3.94°) came out slightly better than the target.

Voronoi occupancy CV is reported in the CSV. It is dominated by Poisson noise at high N (about 3–15 samples per cell), so the uniformity comparison uses spacing CV and the exact cell volumes from EXP-002.

## Domain-wise results
n/a (grid geometry only; EXP-002 shows SPEED+ GT matches Haar statistics).

## Qualitative results
`figures/p95_vs_n.png`: p95 error against N on log–log axes for all grids.

## Resource results
- Parameters: 0
- Trainable parameters: 0
- Peak memory: see `status.json`
- Mean/median latency: n/a
- Tesseract nodes visited: greedy 4 + 8L (44 at L5). Hopf hierarchy: 72 + 8r (104 at r4). Super-Fibonacci has no hierarchy, so it is flat (N).
- Flow sampling latency and NFE, if applicable: n/a
- Parent-child mass consistency error, if applicable: n/a
- Credible-region coverage, if applicable: n/a

## Comparison with control
**At equal N, Super-Fibonacci is better than Tesseract on every statistic:**
- Its mean error is about 1–5% lower, and its p95 about 4–14% lower.
- Its worst case is much lower: 4.52° versus 5.86° at L5, and 8.53° versus 11.81° at L4.
- Its spacing CV is roughly 3× smaller.
- For equal p95 error it needs only about 0.61–0.88× the Tesseract count.

**Against Hopf, Tesseract never matches its counts.** Hopf is more uniform (spacing CV about 0.06 versus 0.21). On the p95-versus-N curve the two lie close together, with Hopf slightly better per point.

**Tesseract's distinct advantage is search cost.**
- Greedy descent scores 44 nodes at L5, against 131k for a flat search of Super-Fibonacci.
- That is fewer than Hopf's own hierarchy needs at similar precision (96–104 nodes), because Tesseract starts from 4 roots rather than 72.

## Interpretation
- **Tesseract is not the most precise grid per hypothesis.** The L∞ chart construction causes up to 14× cell-volume variation and a heavier worst-case tail.
- **Its cheap hierarchy is its defensible property.** It has 4 roots, 8 children per level, and closed-form encoding and decoding with no lookup tables.
- **The gate condition ("precision *or* search advantage") is met on search.** It is not met on precision.
- **Hopf is the stronger hierarchical competitor.** It must stay the matched baseline in predictor experiments (mandatory baseline 9), compared at matched node budgets and covering radius, not matched "level" names.

## Failure analysis
None. The cubochoric substitution is documented above.

## Decision
Keep the Tesseract as the primary rotation representation, with L_max = 5, justified by search cost. Carry the Hopf grid forward as the mandatory matched hierarchical baseline for Phase 3. Report precision honestly as worse per hypothesis than near-uniform grids.

## Next experiment
Phase 1: EXP-010 (DINOv3-only) and EXP-011 (VGGT controls).

## Artifact index
- Metrics JSON: `outputs/experiments/EXP-003_grid_comparison/RUN-20261005-174130-seed0/metrics/{metrics,matched_covering}.json`
- CSV: same directory, `*.csv`; also `outputs/comparisons/rotation_grid_summary.csv`
- TSV: same directory, `*.tsv`; also `outputs/comparisons/rotation_grid_summary.tsv`
- Tables: `.../tables/grid_table.md`
- Figures: `.../figures/p95_vs_n.png`
- Predictions: n/a
- Checkpoint, if any: n/a
- Logs: `status.json`, `environment.txt`, `command.txt`
