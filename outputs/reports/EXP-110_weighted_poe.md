# EXP-110: Weighted PoE of the DINOv3 and MoGe-2 decisions

## Status
**Completed 2026-10-16. Criterion 1 PASS on its pre-registered proxy** (w* = 0.8: val-aug 23.65 ± 0.93 vs 30.84 ± 0.42 at w = 0.5; clean synthetic 5.14 vs 5.54), **but the real domains disagree** (lightbox 32.0 vs 26.9; sunlamp 36.7 vs 36.7). Score **2/5 Weak**. Not adopted on its own: see the decision.

**Pre-registered 2026-10-15, before any run** (DEC-016, user: "you can also test different weights for moge and dino decisions while PoE"). Inference only; frozen EXP-036 trees. Runs after EXP-108, in the same chain.

## Method
- At every level: log p_w = log_softmax(w·log p_DINO + (1 − w)·log p_MoGe), greedy decoding.
- w = 0.5 gives the same argmax as the current equal-weight PoE; w = 1 is DINOv3 only; w = 0 is MoGe-2 only.
- Grid fixed now: w ∈ {0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0}. Seeds 0–2.

## Selection (DEC-006-compliant)
- **Selection metric: the EXP-109 synthetic val-aug proxy** (synthetic val with the seed-1000 augmentation, labels rotated by the geometry-derived "pre−" convention).
- Clean synthetic val is not used for selection: EXP-036 found DINOv3 alone better there (5.35° vs PoE 5.54°), while PoE is much better on real images. Selecting on clean synthetic would therefore favour w → 1 for reasons unrelated to generalisation.
- Clean synthetic val, lightbox and sunlamp are reported for every w.

## Criteria
1. **A non-equal weight helps:** the w* with the lowest 3-seed val-aug mean beats w = 0.5 on val-aug by more than the larger seed std, and is not worse than w = 0.5 on clean synthetic val by more than the larger seed std.
2. Real domains are descriptive for every w, with the flip transitions vs w = 0.5 (counts and denominators).

## Expectation (stated before results)
The weight probably matters less than the flips do. If val-aug picks w > 0.5, the real-domain numbers will show whether the proxy agrees with real images. No real-domain number is used for selection.

## Runs
`EXP-110_weighted_poe` latest RUN (`scripts/exp110_weighted_poe.py`); EXP-036 trees, seeds 0–2.

## Results (3-seed mean ± std, PoE greedy with DINOv3 weight w)
| w (DINOv3) | 0 | 0.2 | 0.3 | 0.4 | **0.5** | 0.6 | 0.7 | **0.8 (w*)** | 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| synth val-aug | 75.50 | 57.48 | 47.68 | 38.26 | 30.84 ± 0.42 | 26.14 | 24.08 | **23.65 ± 0.93** | 24.73 |
| synth val | 16.71 | 10.29 | 7.90 | 6.36 | 5.54 | 5.25 | 5.14 | 5.14 | 5.35 |
| lightbox | 43.71 | 34.24 | 30.47 | 27.94 | **26.89** | 27.74 | 29.49 | 32.04 | 37.09 |
| sunlamp | 60.58 | 49.72 | 44.47 | 39.99 | 36.67 | 35.55 | **35.12** | 36.74 | 41.44 |

## Criteria
1. **PASS on the proxy:** −7.2° on val-aug, and clean synthetic improves.
2. **Real (descriptive):** the best lightbox weight is about 0.5 and the best sunlamp weight about 0.7. w* = 0.8 is 5.1° worse on lightbox.

## Interpretation
- The augmented-synthetic proxy rewards DINOv3 more than real images do. MoGe-2's geometry still carries real-domain information that the proxy does not reward.
- These are the **clean-trained EXP-036** trees. With augmentation-trained trees (EXP-109 A) the balance shifts, because DINOv3-aug improves a lot and MoGe-2 hardly at all.
- So the weight should be decided on the final trees, not here.

## Decision
- Not adopted as a standalone change: the proxy and the real domains disagree, and the recipe has moved to EXP-109 A + C.
- If the user wants it, a weight check on the A trees (with orbit voting) can be pre-registered, with selection on val-aug as here, and the real domains reported.
