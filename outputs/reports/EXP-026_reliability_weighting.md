# EXP-026: Training-free reliability signals for posterior fusion, tested one at a time

## Status
Completed. Pre-registered on 2026-10-09, before any result. This is the plan's Phase-2 EXP-021, renumbered under DEC-002.


## Keep score
**2 / 5: Weak.** No reliability signal beats equal-weight PoE by the pre-registered margin. Margin weighting ties it (−0.08°). The informative side result: plain PoE is already near the best a fixed weighting can do.

_Keep score rubric (0–5): 5 strong support: pre-registered goal met with a large effect, adopt. 4 support: goal met, carry forward. 3 partial: mixed or one-sided evidence, keep for further testing. 2 weak: goal not met but an informative side result. 1 negative: fails, reject, kept only as a recorded result. 0 inconclusive or not run._

## Research question
Does weighting each branch by a fixed, training-free reliability signal improve on the equal-weight PoE of EXP-025?

## Component under test
Fusion (reliability weighting)

## Track
Training-free.

## Pre-registered protocol
- **Branches, checkpoints, seeds and standardization:** as EXP-025 (A = DINOv3-L `grid4`, B = consensus `normals16~and`, seeds 0/1/2 paired). Greedy decoding at L5.
- **Signals, each tested on its own** (`src/tfpose/fusion.py`):
  1. **Entropy:** per image and level, w_i ∝ 1/(H_i + 1e-3), with Σw = 2 (equal weights = PoE).
  2. **Margin:** w_i ∝ (p_top1 − p_top2) + 1e-3, with Σw = 2.
  3. **Agreement gate:** per image, if the greedy single-branch predictions of A and B lie within 20° → PoE, otherwise A alone. The 20° threshold is fixed a priori and matches EXP-012/014/020.
- **Plan signals not run, with reasons:**
  - VGGT confidence: VGGT is retired (DEC-001).
  - Spectral-energy reliability: no signal on synthetic val in EXP-020.
  - Query/template spectral coherence: there are no templates in this pipeline.
- **Claim:** a signal "helps" if its synthetic-val mean error is below PoE's by more than PoE's seed std (0.46°). Lightbox and sunlamp are descriptive.
- **EXP-027:** if two or more signals help, test their combination. Otherwise equal-weight PoE stays the fusion of record.

## Runs
`RUN-20261009-005422-seed0`: completed under `guarded.sh 24`; sources are in `source_runs.txt`. The A and PoE rows reproduce EXP-025 exactly.

## Results
Greedy decoding, mean over 3 paired seeds. Source: `metrics/metrics.csv`.

| method | synthetic_val mean ± std / median | lightbox mean / median | sunlamp mean / median | acc@20 (synth / lightbox / sunlamp) |
|---|---|---|---|---|
| A: DINOv3-L alone | 27.7 ± 0.8 / 17.2 | 76.1 / 62.2 | 79.1 / 70.3 | 0.585 / 0.145 / 0.109 |
| **PoE (equal weights)** | **25.9 ± 0.5 / 14.7** | 62.5 / 41.3 | **71.4 / 56.5** | 0.656 / 0.264 / 0.169 |
| margin-weighted PoE | 25.8 ± 0.4 / 14.8 | **62.3 / 40.8** | 71.9 / 58.0 | 0.652 / 0.263 / 0.176 |
| entropy-weighted PoE | 26.1 ± 0.4 / 15.0 | 63.7 / 43.7 | 71.9 / 58.3 | 0.646 / 0.258 / 0.175 |
| agreement gate (PoE if A, B within 20°, else A) | 27.4 ± 0.7 / 16.8 | 76.0 / 62.2 | 79.0 / 70.3 | 0.594 / 0.148 / 0.113 |

## Pre-registered verdict
None of the three signals helps. The threshold was synthetic-val mean below PoE by more than 0.46°:
- margin: −0.08°;
- entropy: +0.18°;
- agreement gate: +1.48°.

EXP-027 (combination) is skipped because no signal helped individually. The learned-fusion EXP-028/029 are not triggered, because training-free fusion already passed the Phase-2 gate (EXP-025).

## Interpretation
- **Equal-weight PoE already lets the more confident branch dominate.** A peaked distribution contributes more to the product than a flat one, so explicit entropy or margin weighting adds little.
- **The agreement gate hurts.** On disagreeing images, falling back to DINOv3 alone discards exactly the cases where PoE helps most: most real images disagree by more than 20°.

## Decision
Equal-weight PoE(DINOv3-L `grid4`, consensus `normals16~and`) is frozen as the Phase-2 fusion (DEC-003).
