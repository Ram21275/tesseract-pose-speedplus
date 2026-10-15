# EXP-050: Oracle adaptive-depth targets (reduced Phase 5)

## Status
**Pre-registered 2026-10-15, before any result.** Phase 5 is reduced to this oracle check (DEC-012). EXP-051–055 run only if the criterion below passes.

## Question
Is there headroom for choosing the tree depth per image? In other words, does stopping at a shallower cell centre ever beat the depth-5 leaf, by enough to justify a controller?

## Setup
- Frozen EXP-036 PoE trees, seeds 0–2. Greedy PoE path with no GT routing, as the plan requires.
- For each image: e_l = error of the path's prefix-cell centre at depth l = 0…5.
- Oracle depth l* = argmin_l [e_l + λ·l], with λ ∈ {0, 0.1, 0.5, 1.0}° per level (cost proxy = levels decoded).
- BRANCH oracle = the best of the beam-8 leaves.
- Domains: synthetic train (targets saved: STOP early / DESCEND to leaf / BRANCH), synthetic val (selection), lightbox / sunlamp (descriptive).
- Code: `scripts/exp050_oracle_depth.py`.

## Criterion (fixed now)
- **Proceed to EXP-051–055** only if the synthetic-val STOP/DESCEND headroom at λ = 0, mean(e_5) − mean(e_{l*}), is ≥ 0.25°. That is four times EXP-036's seed std of 0.063°, and enough to matter after an imperfect controller.
- Otherwise Phase 5 stops with a documented negative result. Fixed depth 5 is kept.
- BRANCH headroom is reported, but it is the candidate-selection problem already handled by the Phase-4 verifier (EXP-105 / 045). It does not by itself trigger Phase 5.

## Runs / Results / Decision
Not started.
