# EXP-105: Candidate expansion (180° flips, cell corners) and best-fit selection on SPEED+

## Status
**Pre-registered 2026-10-13, before any result** (DEC-010, user request). Runs after the EXP-044 chain. Absorbs the proposed EXP-103 ("rotate and match"), which becomes the synthetic-template arm here.

## Research question
The frozen Phase-3 predictor's large errors are mostly 180° flips about a body axis caused by visual near-symmetry (EXP-104). If each prediction is expanded into a small candidate set (its flips and its cell corners), can a scorer pick the correct candidate, and with which kind of evidence: the tree itself, the flow/diffusion posterior, or comparison with synthetic renders?

## Fixed inputs (all frozen)
- Trees: EXP-036 PoE (DINOv3 `tokens8` + MoGe-2 `normals16~and`), seeds 0–2.
- Flow: EXP-042 fused flow heads (velocity average, Haar base), same seed.
- Templates: synthetic/train images only (never val/test), with their GT rotations and frozen features.

## Candidates per image
Base hypotheses = the top-8 PoE beam leaves (centres). Expansion arms, kept separate:
- **C0 (control):** the 8 beam centres.
- **C1 flips:** each centre × {identity, 180° about body x, y, z} = 32.
- **C2 corners:** each centre + the 8 corners of its leaf cell = 72. This is the user's corner idea. The leaf centre–corner distance is reported first: if it is a few degrees, corners can refine precision but cannot repair a flip.
- **C3:** flips + corners.

## Scorers (each evaluated on its own; no fitted combination weights)
1. **Tree:** PoE log-probability of the candidate's leaf (teacher-forced). Control.
2. **Flow density:** exact log-density of the EXP-042 flow at the candidate (continuous normalizing flow on SO(3): integrate backwards to t = 0, Haar base density, minus the integrated divergence of the 3-D tangent velocity, computed exactly by autograd). On SO(3) the diffusion score equals the flow-matching target (LiePose analysis), so this is also the diffusion-model likelihood; no separate SO(3) diffusion model is trained.
3. **Diffusion re-noise / return:** SDEdit-style. Partially noise the candidate to time τ ∈ {0.5, 0.8} (geodesic interpolation from a Haar sample), integrate the flow to t = 1 with 16 draws, and score = −mean geodesic distance from the endpoints back to the candidate. A candidate the posterior "pulls back to" fits better.
4. **Synthetic template comparison (render-and-compare with the dataset's own renders; no SPEED+ 3-D model exists):** for each candidate, the k = 8 synthetic-train images with the nearest GT rotation (all within the reported distance). Score = mean per-token cosine similarity between the test crop and the templates, separately for DINOv3 `tokens8` (4a) and MoGe-2 normals `normals16` (4b).

## Metrics
- **Oracle ceiling** (uses GT; reported only as a ceiling): error of the candidate closest to GT, per candidate arm.
- **Flip discrimination** (selection metric, synthetic val): for each image, the GT leaf centre and its 3 body-axis flips are scored; accuracy = fraction where GT ranks first. This works even though synthetic val has almost no > 90° errors (DEC-006).
- **End-to-end:** re-ranked point estimate (top-scoring candidate) per candidate arm × scorer: mean / median / p95 error, acc@10, fraction > 90° and > 150°, per domain.
- Runtime per image per scorer.

## Pre-registered criteria
1. A scorer is **useful for flips** if its synthetic-val flip-discrimination accuracy exceeds the tree's by more than the larger seed std.
2. A (scorer, candidate arm) pair **helps** if its re-ranked synthetic-val mean is not worse than the greedy EXP-036 PoE by more than the larger seed std, and criterion 1 holds for its scorer.
3. Lightbox / sunlamp: descriptive only, reported for every pair whatever the direction.

## Baselines reported
Internal: EXP-036 greedy, EXP-042 KDE mode, best-of-8 beams (oracle), EXP-044 (if selected). External SPEED+ numbers (SPEC2021) are full-image 6-DoF and are listed as unmatched context only.

## Amendment 2026-10-13 (before any run): arm K, the original Kurz et al. (2017) corner formulation
Requested by the user (DEC-010 addendum). Added next to the other arms; nothing else changes.
- **Representation (as in the original paper):** all **8** facets of the tesseract (±w, ±x, ±y, ±z), so q and −q are both represented, and rotations are represented by the **corners** of the subdivided cubes (radially projected), not by cube centres. A corner is shared by up to 8 neighbouring cubes, so one rotation has many tree paths.
- **Tree:** the same Transformer, with an 8-way root (one per facet) and 8 children per level, depth 5.
- **Training target (multi-path):** for each training rotation, the nearest corner representative, for both q (on its facet) and −q (on the opposite facet); the target mass is spread uniformly over every leaf cube that has that corner, in both copies. Implemented with the EXP-044 level-marginal loss, using these paths in place of samples.
- **Decoding:** beam search (top 8 leaves); each leaf votes its probability to its 8 corners; votes for the same rotation (same corner, either copy) are summed; the prediction is the corner with the largest summed mass.
- **What is compared:** K vs the single-path centre tree (EXP-036 recipe, same features, budget and seeds), under each experiment's own metrics and criteria.
- **Expectation, stated before results:** q and −q are the same rotation, and corners shared between cubes are only a few degrees apart. So the redundancy is not expected by itself to resolve 180° flips (different rotations, far apart on the hypersphere); any gain is expected in precision and at chart seams. Reported whatever the outcome.

## Runs / Results / Decision
Not started.
