# EXP-114: Orbit-product scoring vs maximum confidence vs rotation-cluster voting (same views, same candidate pool)

## Status
**Pre-registered 2026-10-16, before any run** (DEC-019, user: "Same four views and candidate pool: your orbit-product score versus maximum confidence and simple rotation-cluster voting to know that our inference mechanism adds something other than just TTA"). Evaluation only (EXP-109 arm-A trees; both variants; seeds 0–2).

## Rules (all on the same 4 views and the same 32 back-mapped candidates)
- **greedy:** view 0 only (no TTA).
- **orbit_product (ours):** s(c) = Σ_k log p_k(Exp(−θ_k·d) ⊗ c), the tree likelihood of each candidate in every view.
- **orbit_maxconf:** the candidate with the highest joint beam log-probability in its own view.
- **orbit_vote1:** rotation-cluster voting among the 4 back-mapped view top-1s (5° Gaussian kernel; ties broken by confidence). This is plain TTA voting.
- **orbit_vote32:** weighted kernel voting over all 32 candidates (weights = per-view normalized probabilities).

## Criteria
1. **The product score adds something beyond TTA:** its 3-seed **synthetic-val** mean is below the best of {maxconf, vote1, vote32} by more than the larger seed std (PoE variant primary; DINOv3-only secondary).
2. Also judged on synthetic val-aug by the same rule.
3. Real domains: descriptive, with flip transitions vs greedy (denominators).
4. If a simpler rule ties, the mechanism is reported as "TTA voting", without a product-score claim.

Evaluator: `scripts/orbit_eval.py`.

## Runs / Results / Decision
Not started.
