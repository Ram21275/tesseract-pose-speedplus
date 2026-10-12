# EXP-043: Flow-induced Tesseract masses from the same endpoint samples (Phase 4)

## Status
Running. Pre-registered on 2026-10-13, before any result. No training: it re-samples the frozen EXP-042 flows and the frozen EXP-036 trees.

## Keep score
Pending.

## Research question
Do the endpoint samples of the image-conditioned SO(3) flow (EXP-042) define per-level Tesseract masses that:
- are exactly parent–child consistent;
- place more probability on the ground-truth cell than the tree does, at some depth;
- give a better discrete decoding than the tree's greedy path?

## Component under test
Posterior (flow-induced Tesseract masses); input to EXP-044.

## Track
Analysis on frozen predictors (EXP-042 flow heads, EXP-036 trees); training-free PoE / velocity-average fusion.

## Fixed setup
- **Samples:** for each seed 0/1/2, the EXP-042 fused flow (both branch heads, velocity average) is re-sampled with M = 32 endpoints per image, from a Haar base, with NFE 10 (Euler on the group). Run: `exp04x_fused_eval.py --exp EXP-042_image_conditioned_so3_posterior --run-exp EXP-043_flow_induced_tesseract_masses --fused-only --save-samples --save-all-seeds`. EXP-042 itself saved samples for seed 0 only.
- **Groups:** synthetic val (decision) and lightbox / sunlamp (descriptive, DEC-006).
- **Masses:** μ_l(c) = (1/M) · #{m : prefix_l(R_m) = c}. The **same** 32 samples give every level l = 0 (root chart) … 5 (leaf), with the q ~ −q canonical encoding of `tesseract.encode`. Boundary fluxes are not estimated (plan, ground rules).
- **Tree comparison:** the frozen EXP-036 PoE tree (same seed). Its mass on the GT prefix at level l is the product of the PoE conditional probabilities along the GT path (teacher-forced, read-only). Its top-1 at level l is "greedy prefix = GT prefix".

## Reported metrics
- **Per level and domain** (3-seed mean):
  - flow vs tree mass on the GT cell;
  - flow recall (GT cell has mass > 0);
  - flow top-1 (most-massive cell = GT) vs tree greedy top-1;
  - flow occupancy (distinct cells with mass);
  - parent–child consistency error (max |μ(parent) − Σ μ(children)|, checked exhaustively on about 200 images per group);
  - mass self-consistency vs M (mean TV distance between the masses from the first m ∈ {4, 8, 16} samples and from all 32);
  - seam subset (GT chart margin < 0.05): flow vs tree root top-1.
- **Point estimates** (synthetic-val decision, real descriptive):
  - tree greedy leaf;
  - flow KDE mode (EXP-042's estimator);
  - **flow-mass greedy descent:** the most-massive root, then the most-massive child among the samples in the chosen prefix, down to the leaf centre; ties go to the smallest id.

## Pre-registered criteria
1. **Sanity (required):** the parent–child consistency error must be exactly 0. A non-zero value means an encoding bug, and EXP-044 does not start.
2. **Discrete decoding:** flow-mass greedy descent "helps" if its synthetic-val mean is below the tree greedy mean by more than the larger seed std.
3. **Phase-4 path:** EXP-044 (distilling these masses into the tree) proceeds per the plan if criterion 1 holds, whatever the outcome of criterion 2. Criterion 2 and the per-level mass comparisons are reported as evidence about whether the flow carries information the tree lacks.

## Runs / Results / Decision
Pending.
