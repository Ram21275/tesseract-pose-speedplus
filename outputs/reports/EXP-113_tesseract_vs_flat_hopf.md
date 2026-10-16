# EXP-113: Tesseract tree vs matched flat Hopf classifier under the locked recipe (identical augmentation and orbit inference)

## Status
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

## Runs / Results / Decision
Not started.
