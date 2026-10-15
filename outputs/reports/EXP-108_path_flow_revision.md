# EXP-108: Geometry-conditioned iterative path denoiser (requested as "direct path flow") — revising root and child decisions from image and rotation geometry

## Status
**Pre-registered 2026-10-15, before any code was run** (DEC-013, user request: "add an experiment direct path flow that iteratively revises the root and child decisions themselves using image and rotation geometry").

## Motivation
- The tree decodes top-down and never revisits a decision. A wrong early decision is final: a wrong chart at the root, or a 180° flip that is already decided at levels 0–2.
- EXP-042 refines in continuous rotation space. EXP-105 re-ranks a fixed candidate set.
- Neither lets the model change the discrete path itself after seeing the whole hypothesis. This experiment learns a **discrete flow on the path**: starting from the tree's path, it repeatedly re-predicts every decision (root included) jointly, conditioned on the image and on the geometry of the current hypothesis.

## Method
**Path reviser (one per branch, trained; trees and backbones frozen):**
- Bidirectional (non-causal) Transformer over the 6 decision tokens [root, c₀…c₄], with 2 layers, d = 256.
- Each token = embedding of its current decision (root: 4-way; child: 8-way) + level embedding + Fourier(cell centre of its prefix).
- One extra **geometry token**: Fourier features of the current leaf rotation (3×3 matrix), plus Fourier features of its three 180° body-axis flips. The model can then compare the hypothesis with its look-alikes.
- Cross-attention to the frozen EXP-036 encoder memory of that branch (as the flow heads do).
- Output: a distribution for every level's decision at once.

**Discrete flow-matching training (synthetic train only):**
- Sample time t ~ U(0,1). Each GT decision is kept with probability t and otherwise corrupted.
- Corruption kernel: mostly geometry-aware, with uniform noise.
  - 35%: the whole path of a 180° body-axis flip of the GT (re-encoded).
  - 25%: the path of a nearby rotation (GT perturbed by 3–15°, crossing cells and seams).
  - 40%: a uniformly random decision at that level.
- Corrupted paths are re-made consistent top-down. Loss = cross-entropy of the GT decision at every level.
- The two branches are combined as in the rest of the method: equal-weight per-level PoE of their revision distributions.

**Inference (Euler-like iterative revision):**
- Start from the PoE greedy path.
- For k = 1…K: run both revisers, form the per-level PoE, and re-decode **top-down** (argmax root, then argmax child given the new prefix, which is re-evaluated with the reviser), so the path stays valid. Stop early if the path does not change.
- K ∈ {1, 2, 4, 8} is reported; K = 4 is the pre-declared system (chosen now, not tuned).

**Budget:** EXP-036 recipe where applicable (AdamW 1e-3, wd 0.05, OneCycle, batch 256, bf16). 50 epochs (half of EXP-036's, because the encoder is frozen and reused). Checkpoint by synthetic-val mean after K = 4 revisions. Seeds 0–2. Features in host memory; guarded queue.

## Metrics
- PoE synthetic-val mean / median / p95, acc@5/10, and fraction > 90° and > 150°, per domain.
- Fraction of images whose path changes, and the per-level change rate.
- Changes split into improvements and harms.
- Latency.
- Lightbox / sunlamp descriptive (DEC-006).

## Criteria
1. **Helps (primary):** the 3-seed synthetic-val mean at K = 4 is below EXP-036 PoE greedy (5.540 ± 0.063°) by more than the larger seed std.
2. **Repairs flips (secondary, descriptive on real):** the real-domain > 90° fraction falls below EXP-036's (lightbox 0.103, sunlamp 0.154).
3. Everything is reported whatever the direction.

## Expectation (stated before results)
Synthetic val has almost no flips, so criterion 1 can only reward precision and seam repairs. On real images the reviser reads the same frozen features that EXP-105 found ambiguous between flips, so large flip repair is not expected unless the geometry token gives a cue the tree cannot use. The experiment tests this.

## Amendment 2026-10-15 (external review; before any EXP-108 training or evaluation run)
1. **Formulation, renamed.** The implementation has no corruption-time input, no defined probability path, and decodes by repeated argmax. It is therefore a **geometry-conditioned iterative path denoiser**, not discrete flow matching, and is reported as such. It is kept as the baseline. A discrete flow-matching variant (defined probability path, time conditioning, a consistent transition sampler) would be a separate, separately pre-registered experiment; it is not run here. The sections above that say "discrete flow matching" / "Euler-like" are superseded by this naming. The method itself is unchanged.
2. **Corruption geometry, measured** (`scripts/exp108_corruption_geometry.py`, `RUN-20261015-135058-seed0`; no model involved):
   - The token-wise kernel samples geometry-aware **source** paths, but mixing tokens is **not geometry-preserving**.
   - At t = 0.3–0.6, 45–53% of corrupted paths are > 30° from *both* GT and source (flip sources 59–69%).
   - Paths with a changed root average 130–150° from GT.
   - A prefix-mixing kernel (GT above depth j, source below) degrades smoothly (e.g. t = 0.6: 13° vs 68° mean error).
   - The pre-registered token-wise kernel is **kept** for the main and control arms. A prefix-mixing arm is listed as a possible follow-up and is not run unless the user asks.
3. **Matched control added:**
   - Arm **"nogeo"**: identical modules and parameter count (1,752,076 per branch, checked), with the geometry-token features zeroed, and the same corruption, budget, seeds and checkpoint rule.
   - Criterion 4 (new): **geometry adds value** if the geo arm's 3-seed synthetic-val mean at K = 4 is below the nogeo arm's by more than the larger seed std.
   - Criteria 1–3 are unchanged and are applied to the geo arm.
4. **Metrics added:**
   - Path changes are counted from the actual decisions: root, each child level, any.
   - **Flip transitions with denominators**:
     - n of initially > 150° images, and the fraction corrected to < 20°;
     - n of initially < 20° images, and the fraction broken to > 150°.
   - The existing metrics are retained.
5. Selection stays synthetic-only; lightbox / sunlamp remain descriptive.

## Runs / Results / Decision
Not started.
