# EXP-114: Orbit-product scoring vs maximum confidence vs rotation-cluster voting (same views, same candidate pool)

## Status
**Completed 2026-10-16. Criterion 1 (synthetic val) FAIL:** the weighted vote over the 32 candidates beats the orbit product (PoE 3.46 ± 0.03 vs 3.87 ± 0.03; DINOv3-only 3.71 vs 4.21). **Criterion 2 (val-aug) PASS for PoE, marginally** (5.82 ± 0.10 vs vote32 5.98 ± 0.13), and **FAIL for DINOv3-only** (6.25 vs 6.30, within std). Real (descriptive): product best on both domains. Max-confidence and top-1 voting are clearly worse. Score **3/5 Partial**.

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

## Runs
`RUN-20261016-231825-seed0` (`scripts/orbit_eval.py`, EXP-109 arm-A trees, seeds 0–2).

## Results (3-seed mean, °; std in brackets for synthetic sets)
| variant | rule | synth val | synth val-aug | lightbox | sunlamp |
|---|---|---|---|---|---|
| PoE | greedy | 4.85 (0.03) | 9.75 (0.06) | 17.33 | 20.15 |
| PoE | max-confidence | 5.51 (0.03) | 8.43 (0.03) | 14.63 | 15.92 |
| PoE | vote top-1 (4) | 4.37 (0.03) | 6.85 (0.11) | 13.24 | 13.38 |
| PoE | **vote weighted (32)** | **3.46 (0.03)** | 5.98 (0.13) | 12.58 | 12.88 |
| PoE | **orbit product** | 3.87 (0.03) | **5.82 (0.10)** | **12.30** | **12.25** |
| DINO | greedy | 4.84 (0.05) | 9.88 (0.22) | 22.55 | 22.00 |
| DINO | max-confidence | 5.92 (0.02) | 8.71 (0.13) | 20.02 | 17.75 |
| DINO | vote top-1 (4) | 4.65 (0.03) | 7.27 (0.13) | 18.68 | 16.50 |
| DINO | **vote weighted (32)** | **3.71 (0.05)** | 6.30 (0.06) | 18.27 | 15.83 |
| DINO | **orbit product** | 4.21 (0.04) | **6.25 (0.12)** | **17.40** | **15.27** |

**Cost:**
- Weighted voting reuses the per-view beam scores plus a 32 × 32 angle matrix (< 0.1 ms per image).
- The product re-scores 32 candidates in 4 views: 8.2 ms (DINOv3-only) / 15.3 ms (PoE) per image at batch 1; 0.4 / 1.2 ms batched (EXP-112).

## Interpretation
- **Orbit consistency, not a specific score, is the robust effect.** Both multi-candidate rules (weighted voting, product) far exceed greedy, max-confidence and top-1 voting on every domain.
- On clean synthetic data, weighted voting is more precise: it averages nearby candidates' support, giving a finer-than-leaf choice.
- On shifted data (val-aug, real), the product is as good or better, and lightest on flips.
- **Under the pre-registered rule** (synthetic val), the product score **does not** add something beyond voting. The claim is therefore limited to "multi-view orbit consistency over the hierarchical candidates", with the product and weighted voting as two instantiations.

## Decision
- The recipe keeps orbit voting. The scoring rule is reported as a choice:
  - **weighted voting** is lighter and best on synthetic;
  - the **product** is best on real (descriptive only).
- Any switch of the locked rule is the user's decision (DEC-018 locks the product). No real-domain number is used to choose.
