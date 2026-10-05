# EXP-001: Ground-truth quaternion to Tesseract path

## Status
Completed

## Research question
Does the Tesseract encoder produce correct, antipodally invariant, deterministic and invertible paths for every SPEED+ ground-truth rotation?

## Hypothesis
With first-max chart selection, positive-sign canonicalization, L∞ projection and the upper-half tie rule, every invariant listed in ground rules section 5 holds on the real labels.

## Component under test
Tesseract grid

## Track
Training-free (no learning)

## Fixed setup
- Dataset and exact split: all SPEED+ labels (synthetic train 47,966; synthetic validation 11,994; lightbox 6,740; sunlamp 2,791)
- DINOv3 variant/layer: n/a
- VGGT variant/output/layer: n/a
- Predictor: n/a
- Tesseract depth and codebook size: L = 5, 4·8⁵ = 131,072 leaves
- Input resolution and preprocessing: labels only
- Seeds: 0 (perturbation axes, seam probes)
- Compute device: CPU
- Git commit: `9586198` (clean). The first completed run was at `4f2118d` plus an uncommitted fix in `rotations.py`.
- Flow base distribution, solver, NFE and endpoint samples, if applicable: n/a

## Changed variable
None; this is a validation experiment.

## Method
Encoding (`src/tfpose/tesseract.py`):
1. Normalize q.
2. Set chart c to the first index of max |qᵢ|, and multiply q by sign(q_c).
3. Project x = q/q_c, so the three free coordinates lie in [−1, 1].
4. Split each free-coordinate interval at its midpoint, recursively; a value equal to the midpoint goes to the upper half.
5. The child id is b₀ + 2b₁ + 4b₂.
6. The cell centre is the projected cube-cell centre lifted back to S³.

Checks:
- path(q) == path(−q);
- encode(decode(path)) == path;
- finite values and a positive chart coordinate;
- prefix consistency across depths 1–5;
- exact ties in the labels;
- seam behaviour.

The seam checks have two parts:
- Every GT rotation is perturbed by 0.001°. Whenever the leaf changes, the jump between decoded centres must stay within twice the cell bound.
- 200,000 synthetic probes are placed exactly on chart seams (|qᵢ| = |qⱼ|) to test antipodal consistency and the first-max rule there.

The script also writes the derived label manifest to `outputs/data_manifests/` (relative image paths only; the dataset directory is untouched).

## Commands
```
python scripts/exp001_gt_tesseract_paths.py --depth 5 --seed 0
```

## Runs
| Run ID | Seed | Status | Runtime | Artifact directory |
|---|---:|---|---:|---|
| RUN-20261005-173305-seed0 | 0 | **Failed**: `axis_angle_to_quat` did not broadcast a scalar angle over a batch of axes (`rotations.py:91`) | 2.6 s | outputs/experiments/EXP-001_gt_tesseract_paths/RUN-20261005-173305-seed0 |
| RUN-20261005-173314-seed0 | 0 | Completed after the broadcast fix (fix not yet committed) | 4.0 s | outputs/experiments/EXP-001_gt_tesseract_paths/RUN-20261005-173314-seed0 |
| RUN-20261005-173957-seed0 | 0 | Completed (clean commit; bit-identical metrics) | 4.1 s | outputs/experiments/EXP-001_gt_tesseract_paths/RUN-20261005-173957-seed0 |

## Quantitative results
Invariant checks on all 69,491 labels (`metrics/metrics.json`):

| check | result |
|---|---|
| path(q) == path(−q) | True |
| encode(decode(path)) == path | True |
| all finite | True |
| canonical chart coordinate > 0 | True |
| prefix consistent, L1–L5 | True (all) |
| exact chart ties in labels | 0 (minimum top-2 margin 5.0e-6) |
| leaf changes under 0.001° perturbation | 21 of 69,491 (1 chart change) |
| max centre jump / (2 × cell bound) when the leaf changes | 0.587 (≤ 1, so continuous up to cell size) |
| seam probes: antipodally consistent | True |
| seam probes: chart is the first max | True |
| seam probes: error ≤ global cell bound | True |

Label statistics by domain (`metrics/label_stats.csv`):

| domain/split | n | chart 0/1/2/3 fraction | occupied leaves L3 / L4 / L5 | L5 quantization error mean / max (°) |
|---|---:|---|---|---|
| synthetic/train | 47,966 | .248/.250/.246/.257 | 2048 / 14,503 / 38,318 | 2.51 / 5.71 |
| synthetic/validation | 11,994 | .252/.253/.246/.249 | 2008 / 7,895 / 11,270 | 2.51 / 5.78 |
| lightbox/test | 6,740 | .255/.240/.260/.244 | 1847 / 5,265 / 6,519 | 2.51 / 5.49 |
| sunlamp/test | 2,791 | .234/.258/.256/.253 | 1388 / 2,519 / 2,752 | 2.53 / 5.70 |

## Domain-wise results
See above. All charts are roughly equally used in all domains. The rotation distribution is close to uniform; EXP-002 confirms the GT and Haar error statistics agree.

## Qualitative results
n/a (label-space experiment).

## Resource results
- Parameters: 0
- Trainable parameters: 0
- Peak memory: see `status.json`
- Mean/median latency: encoding all 69k labels to L5 takes well under 1 s on CPU
- Tesseract nodes visited: n/a
- Flow sampling latency and NFE, if applicable: n/a
- Parent-child mass consistency error, if applicable: n/a
- Credible-region coverage, if applicable: n/a

## Comparison with control
n/a (validation experiment).

## Interpretation
- **The encoder is correct on the real label distribution.** Every invariant holds, and the decoded centre is continuous across chart and cell seams up to cell size.
- **L5 is sparse for training.** Only 38,318 of the 131,072 leaves (29%) contain any synthetic-train image, so leaf-level supervision is roughly one example per occupied leaf. Deep levels will need geodesic soft targets and/or a continuous residual (EXP-033/035).
- **The L5 grid has a floor.** Mean quantization error is about 2.5° and the maximum about 5.8°, so a pure L5 cell prediction cannot reach the 1° accuracy band.

## Failure analysis
The first run failed on a broadcasting bug in `axis_angle_to_quat`. It was fixed and the code committed; the clean rerun is recorded.

## Decision
Keep.

## Next experiment
EXP-002: precision by depth.

## Artifact index
- Metrics JSON: `outputs/experiments/EXP-001_gt_tesseract_paths/RUN-20261005-173957-seed0/metrics/{metrics,label_stats}.json`
- CSV: same directory, `*.csv`
- TSV: same directory, `*.tsv`
- Derived manifest: `outputs/data_manifests/speedplus_tesseract_L5.{csv,tsv}`; hash in `speedplus_tesseract_L5.sha256.txt`
- Figures: n/a
- Predictions: n/a
- Checkpoint, if any: n/a
- Logs: `status.json`, `environment.txt`, `command.txt`
