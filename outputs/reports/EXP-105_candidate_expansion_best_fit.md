# EXP-105: Candidate expansion (180° flips, cell corners) and best-fit selection on SPEED+

## Status
**Completed 2026-10-14.** C-arms: diffusion re-noise / return re-ranking **helps** on synthetic val (C3 + re-noise τ 0.8: 4.78 ± 0.11° vs EXP-036 5.54 ± 0.07°) and descriptively on real (sunlamp 36.6 → 32.1–32.7°, lightbox 26.6 → 25.3–26.5°), but **no scorer repairs 180° flips on real images** (the flip oracle would take lightbox to 9.8°; re-ranked flip fractions stay at about 10%). Synthetic templates fail. Arm K **passes** on synthetic val (5.20 ± 0.09° vs 5.54 ± 0.07°) but is worse on both real domains. Score **3/5 Partial**.

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

## Amendment 2, 2026-10-13 (before any arm-K run; C-arm scoring already started)
Implementation details and criterion for arm K, fixed before it is trained. The C-arm protocol is unchanged.
- **Code:** `src/tfpose/kurz8.py` (8-facet cells, corner targets, beam, corner vote), `scripts/train_kurz8.py`, `scripts/exp105_armK_eval.py`, chain `scripts/run_exp105K.sh`. Unit check: every rotation gets 8–16 target leaves (mean 16.0), both q and −q copies are always present, and the corner vote with uniform scores returns exactly the nearest grid corner. Depth-5 corner quantization (mean 2.49°, max 5.80°) is about the same as leaf-centre quantization (mean 2.50°, max 6.20°).
- **Training:** EXP-036 budget and optimizer (100 epochs, batch 256, AdamW 1e-3, wd 0.05, OneCycle, bf16). Loss = the EXP-044 level-marginal cross-entropy over the target leaves, estimated with 8 of them drawn uniformly per image per step. Checkpoint chosen by the synthetic-val corner-vote mean (single branch), as EXP-036 chooses by its greedy mean.
- **System:** equal-weight per-level PoE of the two K trees (DINOv3 `tokens8`, MoGe-2 `normals16~and`), beam 8, corner vote. Seeds 0–2.
- **Criterion:** K helps if its 3-seed PoE synthetic-val mean is below the EXP-036 PoE greedy mean (same seeds, recomputed in the same script) by more than the larger seed std. Single-branch K numbers, K's top-1 leaf centre and lightbox / sunlamp are descriptive.
- **Engineering only:** the flow scorers are evaluated in chunks of 25 candidates to fit the shared GPU. This does not change the method.

## Runs
- C-arm scoring, 3 seeds × 3 domains, 96 candidates + 4 GT-flip probes per image: `RUN-20261013-181652-seed0` (`scripts/exp105_best_fit.py --seeds 0,1,2`), log `outputs/experiments/exp105_main.log`.
- Arm K trees, (DINOv3, MoGe-2): seed 0 `020727`, `022722`; seed 1 `030004`, `031914`; seed 2 `035313`, `041153`. Best epochs 65–100. Smoke run in `EXP-099_p3_smoke/RUN-20261014-020514`. PoE corner-vote eval: `RUN-20261014-044939-seed0`. Chain `scripts/run_exp105K.sh`, log `outputs/experiments/exp105K_chain.log`.

## Results (3-seed mean ± std; lightbox / sunlamp descriptive)

### Cell geometry
Leaf-centre to cell-corner distance: mean 4.72°, max 6.20°. So a corner can only refine precision; it cannot reach a 180° flip.

### Flip discrimination (GT leaf centre vs its three 180° body-axis flips; fraction ranking GT first)
| scorer | synth val | lightbox | sunlamp | ms / image (100 candidates) |
|---|---|---|---|---|
| tree (PoE leaf log-prob) | 0.997 ± 0.001 | 0.905 | 0.867 | 2 |
| flow exact log-density | **0.999 ± 0.000** | **0.919** | **0.914** | 72–83 |
| re-noise / return τ 0.5 | **0.999 ± 0.000** | 0.915 | 0.909 | 222–255 |
| re-noise / return τ 0.8 | **0.999 ± 0.000** | 0.907 | 0.895 | 92–104 |
| templates, DINOv3 tokens | 0.631 | 0.529 | 0.612 | 8 (both template scorers) |
| templates, MoGe-2 normals | 0.783 | 0.661 | 0.575 | (included above) |

### End-to-end re-ranked mean error (°); reference = EXP-036 PoE greedy 5.54 ± 0.07 synth, 26.9 lightbox, 36.6 sunlamp (beam-8 top-1 here: 5.59 / 26.6 / 36.7)
| arm | scorer | synth val | lightbox | sunlamp |
|---|---|---|---|---|
| C0 centres (8) | ORACLE | 3.26 | 16.9 | 22.4 |
| C1 + flips (32) | ORACLE | 3.16 | **9.8** | **13.6** |
| C2 + corners (72) | ORACLE | 2.34 | 14.7 | 19.7 |
| C3 all (96) | ORACLE | **2.25** | **7.9** | **11.4** |
| C0 | flow log-density | 5.86 ± 0.09 | **25.3** | 32.9 |
| C0 | re-noise τ 0.8 | 4.99 ± 0.09 | 25.3 | 33.3 |
| C2 | re-noise τ 0.8 | 4.80 ± 0.10 | 25.4 | 33.0 |
| C3 | re-noise τ 0.5 | 4.94 ± 0.13 | 25.9 | **32.1** |
| C3 | **re-noise τ 0.8** | **4.78 ± 0.11** | 26.5 | 32.7 |
| C1 | flow log-density | 5.86 ± 0.10 | 26.3 | 32.6 |
| C1 | templates DINOv3 / normals | 56.0 / 38.1 | 81.4 / 69.2 | 70.3 / 91.3 |
| any | tree | 5.58 | 26.6 | 36.6 |

On real images the > 90° fraction after re-ranking stays at 0.096–0.112 (lightbox) and 0.12–0.14 (sunlamp), against 0.103 / 0.154 for the tree. So the gains are mostly not flip repairs, and the flip candidates are almost never chosen correctly.

### Arm K (original Kurz 2017 corner tree; 8 facets, multi-path targets, corner vote)
| system | synth val | lightbox | sunlamp |
|---|---|---|---|
| EXP-036 PoE greedy (recomputed) | 5.544 ± 0.066 (med 4.62) | 26.9 (med 9.0) | 36.7 (med 13.0) |
| **K PoE, corner vote** | **5.197 ± 0.085** (med 4.43) | 28.8 (med 10.3) | 39.1 (med 16.2) |
| K PoE, top-1 leaf centre (no vote) | 5.216 ± 0.091 | 28.7 | 39.3 |
| K DINOv3 alone (vote) | 5.436 ± 0.071 | 43.1 | 50.1 |
| K MoGe-2 alone (vote) | 14.156 ± 0.106 | 40.0 | 57.6 |

## Criteria
1. **Useful for flips (synthetic val):**
   - Flow log-density and both re-noise scorers: 0.999 vs tree 0.997, a 0.002 difference against a larger std of 0.001. **PASS, but at the ceiling.** Synthetic val has almost no flips, so the margin is small.
   - Templates: **FAIL** (0.63 / 0.78).
2. **Helps (re-ranked synth mean not worse than EXP-036 greedy, and criterion 1 holds):**
   - **PASS** for re-noise τ 0.8 and τ 0.5 on every arm (best: C3 + τ 0.8, 4.78 ± 0.11°, −0.76°).
   - **FAIL** for the flow log-density (C0 5.86°, worse by more than the std).
   - Templates are not eligible (fail criterion 1).
3. **Real domains (descriptive):**
   - Sunlamp improves by 3.5–4.5° with any flow-based scorer (best 32.1°). Lightbox improves by up to 1.6° (best 25.3°).
   - The flip oracle shows how much is left (lightbox 7.9–9.8°), but no scorer captures it.
   - On real images the flow prefers GT over its flips 91–92% of the time, vs 87–91% for the tree.
4. **Arm K:** **PASS** on its pre-registered criterion: −0.35° vs a larger std of 0.085°.
   - Almost all of the gain survives without the corner vote (top-1 centre 5.216°), so it comes from the 8-facet multi-path training, not from corner decoding.
   - Real domains (descriptive): worse (lightbox +1.9°, sunlamp +2.4°). The DINOv3-alone K tree degrades most on real (43.1° vs 37.1° in EXP-036).
   - **Confound, stated now:** arm K changes two things at once: the targets (multi-path corners) and the loss (level-marginal cross-entropy instead of per-level soft cross-entropy). This experiment cannot separate them.

## Interpretation
- **Flips are a selection problem the current evidence cannot solve on real images.** The correct flip is in the candidate set for most real failures (oracle C1: lightbox 26.6 → 9.8°), yet the tree, the flow posterior and synthetic templates all prefer the wrong one about as often as before. The likely cause is the domain gap: the real-image features are ambiguous between the flip modes, and every scorer reads the same frozen features. An independent cue (geometry with a model, temporal, or a real-image-adapted signal) would be needed. Synthetic-train templates are not that cue: they fail even on synthetic val, because the nearest-rotation renders differ in lighting and background.
- **The flow helps as a local refiner, not a flip fixer.** Re-noise / return over the tree's beams plus corners is the best re-ranking: −0.76° synth, −4.5° sunlamp. It costs about 100 ms per image for 100 candidates, versus 2 ms for the tree.
- **Arm K:** the original Kurz corner formulation trains a better synthetic predictor, but transfers worse to real images. Under DEC-006 the synthetic-val criterion decides, so K passes. The real-domain regression is a strong caveat for the recipe lock (DEC-012).

## Decision (proposed for the recipe lock, DEC-012; the user decides)
- Keep EXP-036 PoE as the base predictor. Re-noise / return re-ranking (C3, τ 0.8) becomes the candidate refinement stage, to be compared against EXP-040/041/042 in EXP-045.
- Arm K: reported as a pre-registered synthetic-val pass with real-domain regression. Whether to adopt it is put to the user at the recipe lock, because selection by synthetic val alone would pick K, while real-domain behaviour argues against it.
- Synthetic-template matching: rejected.
