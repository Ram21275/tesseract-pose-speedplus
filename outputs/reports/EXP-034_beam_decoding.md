# EXP-034: Greedy vs beam decoding at full data (Phase 3)

## Status
Completed. Pre-registered on 2026-10-09, before any result.


## Keep score
**1 / 5: Negative.** Beam search gives no gain at any width (±0.01°) and costs up to 6.6× more nodes. Greedy (44 nodes) stays.

_Keep score rubric (0–5): 5 strong support: pre-registered goal met with a large effect, adopt. 4 support: goal met, carry forward. 3 partial: mixed or one-sided evidence, keep for further testing. 2 weak: goal not met but an informative side result. 1 negative: fails, reject, kept only as a recorded result. 0 inconclusive or not run._

## Research question
Does beam search (widths 2, 4, 8) improve over greedy decoding at full data, for the single branches and for the PoE fusion?

## Component under test
Decoder

## Track
Inference only (no training)

## Pre-registered protocol
- **Models:** the EXP-030 checkpoints (3 seeds, both branches). PoE as in DEC-003.
- **Decoding:** beams 1, 2, 4, 8. Reported: mean / median error, nodes scored per image, top-k leaf recall.
- **Claim:** beam k helps if its PoE synthetic-val mean is below greedy's by more than the seed std.
- The beam width chosen here is applied to the frozen Phase-3 configuration.

## Runs
`RUN-20261009-022450-seed0` (`scripts/exp03x_fused_eval.py`, EXP-030 checkpoints). The greedy PoE row reproduces EXP-030 exactly (6.242°), which cross-checks the new evaluation script against the EXP-025 script.

## Results
Mean over 3 seeds. Source: `metrics/metrics.csv`.

| beam | nodes per branch | A synthetic_val mean | B synthetic_val mean | **PoE synthetic_val mean ± std / median** | PoE lightbox mean / median | PoE sunlamp mean / median |
|---:|---:|---:|---:|---|---|---|
| 1 | 44 | 6.639 | 17.791 | **6.242 ± 0.072 / 4.998** | 31.07 / 10.40 | 43.75 / 17.00 |
| 2 | 84 | 6.671 | 17.819 | 6.245 ± 0.077 / 5.002 | 31.06 / 10.40 | 43.65 / 16.91 |
| 4 | 164 | 6.667 | 17.818 | 6.248 ± 0.078 / 5.002 | 31.03 / 10.40 | 43.64 / 16.90 |
| 8 | 292 | 6.667 | 17.826 | 6.248 ± 0.078 / 5.002 | 31.03 / 10.40 | 43.64 / 16.90 |

## Pre-registered verdict
**No beam width helps.** The PoE synthetic-val mean changes by at most +0.006°, well within the 0.07° std.

## Interpretation
- **The trained tree is decisive.** Its top-1 child at each level almost always also leads to the best full-path score, so extra beams re-rank nothing.
- **The remaining error is not a decoding error.** Wrong branches are confidently wrong, and the quantization floor (2.5°) remains.

## Decision
Use **greedy** decoding (44 nodes per branch) in the frozen Phase-3 configuration. Beam is unnecessary.
