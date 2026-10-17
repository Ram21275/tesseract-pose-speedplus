# EXP-113: Tesseract tree vs matched flat Hopf classifier under the locked recipe (identical augmentation and orbit inference)

## Status
**Completed 2026-10-17. Criteria 1 and 2 PASS:** with identical features, augmentation, budget and orbit inference, the Tesseract tree beats the count-matched flat Hopf classifier on synthetic val (PoE + orbit 3.87 ± 0.03 vs 7.53 ± 0.01; greedy 4.85 vs 8.95) and on val-aug (5.82 vs 9.80), with 15× fewer parameters (2.4 M vs 36.1 M per branch). Real (descriptive): tree better on both domains in every configuration, though the flat model has fewer > 150° flips on lightbox with orbit voting. Score **5/5 Strong support**.

**Pre-registered 2026-10-16, before any run** (DEC-019, user: "comparison of this architecture with Final Tesseract versus matched Hopf/flat classifier, both with identical augmentation and orbit inference for claiming this discretization method improves something").

## Question
Does the Tesseract hierarchical discretization + autoregressive path decoder improve on a flat classifier over a count-matched Hopf grid (the SPACE-HOP discretization), when everything else is identical?

## Arms (both trained per branch, seeds 0–2; PoE and DINOv3-only variants)
- **Tesseract (reference):** the EXP-109 arm-A trees (already trained).
- **Flat Hopf (new):**
  - **Same network** as the tree (LN + Linear → 256, 2-D positional encoding, 1 encoder layer, 2-layer decoder with the [ROOT] query only), followed by a linear head over **131,076** anchors.
  - The grid is SPACE-HOP's Hopf construction (`tfpose.grids.spacehop_hopf_grid`), with 10,923 directions × 12 rolls, count-matched to the 131,072 Tesseract leaves (EXP-101).
  - **Targets:** geodesic soft targets over the 32 nearest anchors with the tree's leaf temperature (2.51°).
  - **Training:** same budget, optimizer, features and augmented views (clean + `exp109_train_aug1/2`, labels "pre−"). Checkpoint by the clean synthetic-val argmax mean.
  - **Size:** 36.1 M trainable parameters (head 33.6 M) vs 2.4 M for the tree. This is inherent to flat classification and is reported; no capacity-matching is attempted.
- **Inference, identical for both:**
  - greedy;
  - orbit voting with the same 4 views and a candidate pool of the top-8 per view (32), scored by the orbit product. For flat models the per-view log-probability of a candidate is that of its nearest anchor.
  - The PoE variant uses the per-anchor product across branches for the flat model.

## Metrics
- Synthetic val, synthetic val-aug, lightbox, sunlamp: mean / median, acc@10, fraction > 90° / > 150°, flip transitions vs the same family's greedy (denominators).
- Latency of the head and decoder (EXP-112-style timing for the flat head, added to the EXP-112 table if time allows).

## Criteria
1. **The tree improves on the flat grid:** the tree's 3-seed **synthetic-val** mean is below the flat model's by more than the larger seed std, in the PoE + orbit configuration (primary) and in greedy PoE (secondary).
2. Also judged on synthetic val-aug (the generalisation proxy), by the same rule.
3. Real domains: descriptive.
4. If the flat model matches or beats the tree, this is reported as is, and the contribution "the discretization improves something" is withdrawn.

Evaluator: `scripts/orbit_eval.py` (shared with EXP-114 / 115). Trainer: `scripts/train_flat_tf.py`. Model: `src/tfpose/flatgrid.py`.

## Runs
- Flat trainings: `EXP-113_tesseract_vs_flat_hopf/RUN-20261016-23…` / `20261017-00…` (seeds 0–2, both branches).
- First attempt (14:05, 2026-10-16) failed on the post-DEC-020 path bug (recorded).
- Evaluation: `RUN-20261017-042606-seed0` (`scripts/orbit_eval.py`, greedy and orbit product).

## Results (3-seed mean ± std, °; > 150° fraction)
| family | variant | inference | synth val | synth val-aug | lightbox | sunlamp |
|---|---|---|---|---|---|---|
| **tree** | PoE | greedy | **4.85 ± 0.03** | **9.75 ± 0.06** | 17.33 (0.042) | 20.15 (0.033) |
| flat Hopf | PoE | greedy | 8.95 ± 0.06 | 15.06 ± 0.20 | 20.56 (0.041) | 27.16 (0.057) |
| **tree** | PoE | orbit | **3.87 ± 0.03** | **5.82 ± 0.10** | **12.30** (0.030), med 5.26 | **12.25** (0.017), med 6.32 |
| flat Hopf | PoE | orbit | 7.53 ± 0.01 | 9.80 ± 0.09 | 13.15 (**0.017**), med 8.07 | 16.31 (0.024), med 8.68 |
| **tree** | DINOv3 | greedy | **4.84 ± 0.05** | **9.88 ± 0.22** | 22.55 (0.060) | 22.00 (0.032) |
| flat Hopf | DINOv3 | greedy | 8.61 ± 0.09 | 13.97 ± 0.17 | 27.29 (0.063) | 32.48 (0.063) |
| **tree** | DINOv3 | orbit | **4.21 ± 0.04** | **6.25 ± 0.12** | **17.40** (0.048) | **15.27** (0.020) |
| flat Hopf | DINOv3 | orbit | 7.54 ± 0.03 | 9.55 ± 0.06 | 17.71 (0.033) | 18.61 (0.027) |

Single-branch training-run numbers (flat): DINOv3 8.61 / 27.4 / 32.5; MoGe-2 20.1 / 44.7 / 61.6.

## Criteria
1. **PASS:** PoE + orbit synthetic 3.87 vs 7.53 (−3.66°, std ≤ 0.03); greedy 4.85 vs 8.95.
2. **PASS:** val-aug 5.82 vs 9.80 (PoE + orbit); 9.75 vs 15.06 (greedy).
3. **Real (descriptive):** the tree is better in mean and median in all 8 comparisons. The flat model's orbit voting reaches a lower > 150° fraction on lightbox (0.017 vs 0.030), but is much less precise (median 8.07 vs 5.26).
4. Not triggered: the flat model does not match the tree.

## Interpretation
- At equal hypothesis count (≈ 131k), the hierarchical Tesseract tree is far more precise than a flat classifier over a well-spread grid. The flat model's median is about 2× worse, with 15× the parameters: a 131k-way softmax trained with one example per pose is data-starved per anchor, while the tree shares statistics across levels.
- Orbit voting helps both families, and makes the flat model nearly as good at avoiding flips. So **flip repair comes mainly from augmentation + orbit voting, and precision mainly from the tree** (EXP-115 will separate the augmentation's role).
- Caveats:
  - The flat head's capacity is necessarily different. No offsets / regression refinement on either side (SPACE-HOP adds offsets).
  - Weighted voting was not evaluated for the flat family here (cheap follow-up).

## Decision
The discretization claim is **supported**: the Tesseract tree improves precision and accuracy over a matched flat Hopf classifier under the same recipe.
