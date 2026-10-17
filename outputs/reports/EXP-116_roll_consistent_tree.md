# EXP-116: Roll-consistent learning of Tesseract path distributions (geometry-transported consistency)

## Status
**Pre-registered 2026-10-17, before any code or run** (DEC-021, user idea 1: "When an image undergoes a known rotation, train the model's probability distribution to move through the corresponding tesseract cells correctly: p(T_θ I) ≈ K_θ p(I)").

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

## Runs / Results / Decision
Not started.
