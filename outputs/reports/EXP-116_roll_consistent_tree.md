# EXP-116: Roll-consistent learning of Tesseract path distributions (geometry-transported consistency)

## Status
**Completed 2026-10-18.** Pre-registered 2026-10-17, before any code or run (DEC-021, user idea 1: "When an image undergoes a known rotation, train the model's probability distribution to move through the corresponding tesseract cells correctly: p(T_θ I) ≈ K_θ p(I)").

## Question
Can the tree learn, during training, the cross-view consistency that orbit voting enforces at test time? That is, does transporting its hierarchical distribution under known in-plane rotations reduce severe flips at **single-view** inference cost, better than ordinary augmentation and a plain output-level equivariance loss?

## Transport K_θ (exact for camera roll; derived and verified in EXP-109)
- For views u, v of the same image with in-plane angles θ_u, θ_v (clean = 0), a leaf rotation c in view u corresponds to c′ = Exp(−(θ_v − θ_u)·d) ⊗ c in view v ("pre−", d = undistorted ray through the crop centre).
- p_u is transported by mapping its support: the beam-8 leaves of view u, with their normalized PoE path probabilities w, are mapped to v and re-encoded into Tesseract paths.
- Mapped paths are valid paths of real rotations, so the transport is **geometry-preserving at every level** (unlike EXP-108's token mixing). Parent–child consistency holds because prefixes of mapped leaves define the level marginals.

## Loss (per image, a random pair of distinct views u ≠ v from {clean, aug1, aug2}, both labels known)
L = L_GT(u) + L_GT(v) + λ·[ H(stopgrad K p_u → p_v) + H(stopgrad K p_v → p_u) ], where
H(q → p) = −Σ_j w_j · (1/(L+1)) Σ_l log p^(l)(prefix_l(path_j)) is the EXP-044 level-marginal cross-entropy on the transported support. **λ = 1.0, beam 8 (fixed now, not tuned).**

## Arms (EXP-109 A recipe otherwise: same views, budget, optimizer; both branches; seeds 0–2)
- **T (proposed):** hierarchical transported consistency, as above.
- **C1, ordinary augmentation with paired views:** the same two-views-per-image sampling and GT losses, λ = 0. Data-matched control for T.
- **C2, output-level equivariance loss:** the same transport, but the consistency term applied **at the leaf level only** (l = L), i.e. a standard equivariance constraint on the final distribution, without the hierarchical level terms.
- **A (reference):** the existing EXP-109 arm-A trees (one view per image per step).

## Evaluation (`scripts/orbit_eval.py`, same views as EXP-109 / 113–115)
- **Primary: single-view greedy** (same inference budget as A greedy), PoE and DINOv3-only variants.
- Also orbit (product + weighted vote) for completeness.
- Metrics: mean / median, > 150° fraction, flip transitions vs EXP-036 greedy (denominators).
- **Transformation consistency:** the fraction of orbit views whose back-mapped greedy prediction lies within 10° of view 0's.

## Criteria (synthetic val / val-aug; real descriptive)
1. **T beats the controls:** T's 3-seed single-view greedy val-aug mean is below each of A, C1 and C2 by more than the larger seed std (DINOv3-only primary; PoE secondary), without being worse than A on clean synthetic val by more than the larger seed std.
2. **Consistency improves:** T's transformation consistency on val-aug exceeds A's and C1's.
3. Real domains: descriptive, including > 150° flips.

## Prior work and claim boundary
- Equivariant pose prediction is established (e.g. Image2Sphere, ICLR 2023; Wigner-D equivariant pose regression, NeurIPS 2024). The candidate distinction is transport of a **hierarchical discrete distribution** with level-wise consistency, tested against C2. Exact for camera roll only (SO(2) about the viewing ray), not general 3-D equivariance.
- If T ≈ C2, no hierarchical-transport claim is made.

## Cost
No new features. 3 new arms × 2 branches × 3 seeds = 18 trainings (about 15–25 min each; two views per step plus beam), plus evaluation, about 6–8 GPU-hours.

## Amendment 2026-10-17 (before any EXP-116 run)
- Trees are trained **per branch** (as in EXP-109 A), so the transported distribution p_u is that branch's own beam-8 distribution, not the PoE. The PoE is used only at evaluation.
- The transformation-consistency metric is implemented in `scripts/orbit_eval.py` as `view_consistency@10`: the mean over images of the fraction of views 90/180/270 whose back-mapped top-1 is within 10° of view 0's top-1.
- Trainer: `scripts/train_consistency.py --mode T|C1|C2`.

## Runs
- Trainings (per branch, seeds 0–2): T `RUN-20261017-194457` … `RUN-20261018-020621`; C1 `RUN-20261018-040034` … `055113`; C2 `RUN-20261018-061819` … `124857` (all under `outputs/experiments/EXP-116_roll_consistent_tree/`).
- `RUN-20261017-132338-seed0` (first T attempt): stopped by memguard at epoch 85 (system-available floor); not used. Trainer memory reduced and re-run (DEC-021).
- Evaluation: `RUN-20261018-194430-seed0` (`scripts/orbit_eval.py`, families A = EXP-109 arm A, C1, C2, T; DINO and PoE; greedy / orbit_product / orbit_vote32). Tables: `tables/metrics.md`, per seed `metrics/per_seed.csv`.

## Results (3-seed means ± seed std, degrees)

### Primary: single-view greedy
| variant | arm | synth val | val-aug | lightbox | sunlamp | view_consistency@10 (val-aug) |
|---|---|---|---|---|---|---|
| DINO | A (EXP-109) | 4.84 ± 0.05 | 9.88 ± 0.22 | 22.55 | 22.00 | 0.554 |
| DINO | C1 paired aug | 4.96 ± 0.06 | 10.04 ± 0.21 | 23.75 | 22.72 | 0.541 |
| DINO | C2 leaf equivariance | **4.59 ± 0.04** | **9.43 ± 0.14** | 22.88 | **21.64** | **0.621** |
| DINO | T hierarchical transport | 4.65 ± 0.03 | 9.57 ± 0.06 | 22.84 | 22.20 | 0.603 |
| PoE | A (EXP-109) | 4.85 ± 0.03 | 9.75 ± 0.06 | 17.33 | 20.15 | 0.607 |
| PoE | C1 | 4.89 ± 0.04 | 10.00 ± 0.11 | 17.88 | 20.34 | 0.597 |
| PoE | C2 | **4.52 ± 0.04** | **9.08 ± 0.05** | **17.28** | **18.85** | **0.659** |
| PoE | T | 4.56 ± 0.02 | 9.17 ± 0.12 | 17.39 | 19.01 | 0.646 |

### Orbit inference (4 views)
| variant | arm | synth product / vote32 | val-aug product | lightbox product | sunlamp product |
|---|---|---|---|---|---|
| DINO | A | 4.21 / 3.71 | 6.25 | 17.40 | 15.27 |
| DINO | C2 | **3.92 / 3.40** | **5.66** | 17.40 | **14.47** |
| DINO | T | 4.02 / 3.45 | 5.91 | 17.55 | 14.66 |
| PoE | A | 3.87 / 3.46 | 5.82 | **12.30** | 12.25 |
| PoE | C1 | 3.94 / 3.48 | 6.04 | 12.38 | 12.45 |
| PoE | C2 | **3.71 / 3.29** | **5.39** | 12.41 | 11.74 |
| PoE | T | 3.74 / 3.29 | 5.59 | 12.49 | **11.73** |

Real-domain seed stds are 0.2–1.6° (C2 / T lightbox 0.6–1.4), so the real differences between A, C2 and T are within noise. Lightbox > 150° fraction (PoE + orbit product): A 0.030, C1 0.029, C2 0.032, T 0.032.

## Criteria
1. **T beats A, C1 and C2 on single-view greedy val-aug by more than the larger seed std: FAIL.** T beats A (9.57 vs 9.88; Δ 0.31 > 0.22) and C1 (Δ 0.47 > 0.21), but **C2 is better than T** (DINO 9.43 vs 9.57; PoE 9.08 vs 9.17). Clean synthetic val is not hurt (T 4.65 vs A 4.84).
2. **Consistency improves over A and C1: PASS** (DINO val-aug 0.603 vs 0.554 / 0.541). However, C2 is higher still (0.621).
3. Real domains (descriptive): no arm changes lightbox. C2 and T are about 0.5° better on sunlamp with orbit inference (PoE 11.74 / 11.73 vs 12.25), within about one seed std. Severe flips are unchanged.

## Decision
- **No hierarchical-transport claim** (pre-registered rule: T ≈ or worse than C2). The level-wise terms add nothing over the leaf-level transported consistency loss.
- **What does help:** a transported equivariance loss on the leaf distribution (C2), a known technique. It improves synthetic precision and val-aug consistently (PoE + orbit synth 3.71 vs 3.87; greedy 4.52 vs 4.85, Δ ≫ seed std) at zero inference cost. It does not fix real flips.
- Paired-view sampling alone (C1) is slightly worse than A, so the gain is from the consistency term, not the data pairing.
- **Not adopted into the locked recipe (DEC-018) without the user's decision.** If adopted, it is reported as a training-time equivariance regularizer (prior art: Image2Sphere, ICLR 2023; equivariant pose regression), not as a contribution. Selection would rest on synthetic val only (DEC-006).

