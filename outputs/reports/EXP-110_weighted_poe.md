# EXP-110: Weighted PoE of the DINOv3 and MoGe-2 decisions

## Status
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

## Runs / Results / Decision
Not started.
