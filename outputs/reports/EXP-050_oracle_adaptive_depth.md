# EXP-050: Oracle adaptive-depth targets (reduced Phase 5)

## Status
**Completed 2026-10-15. Criterion PASS** (synthetic-val stop headroom 0.51 ± 0.02° ≥ 0.25°), so EXP-051–055 are eligible. Per DEC-014 they start only when the user says. Score **3/5 Partial**: the headroom is an oracle upper bound (see caveat).

Pre-registered 2026-10-15, before any result. Phase 5 is reduced to this oracle check (DEC-012). EXP-051–055 run only if the criterion below passes.

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

## Runs
`RUN-20261015-132256-seed0` (`scripts/exp050_oracle_depth.py`, seeds 0–2). Training-split targets (seed 0, STOP early / DESCEND / BRANCH) are in `predictions/train_targets_seed0.npz`.

## Results (3-seed mean; per-level prefix-centre error, mean °)
| domain | e0 | e1 | e2 | e3 | e4 | e5 (leaf, = EXP-036) | oracle stop λ=0 | headroom | mean oracle depth | frac. stopping early | λ = 0.5 / 1.0 | BRANCH (beam-8 best leaf) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic train | 75.1 | 39.4 | 19.9 | 10.5 | 5.89 | 3.547 | 3.410 | 0.137 | 4.85 | 0.11 | 3.44 / 3.54 | 2.49 |
| **synthetic val** | 75.4 | 40.0 | 20.6 | 11.5 | 7.27 | 5.541 ± 0.063 | 5.031 | **0.510 ± 0.016** | 4.63 | 0.27 | 5.07 / 5.19 | 3.26 |
| lightbox | 82.7 | 52.6 | 37.5 | 30.5 | 27.8 | 26.89 | 22.68 | 4.21 | 3.82 | 0.53 | 22.7 / 22.8 | 16.9 |
| sunlamp | 84.6 | 58.8 | 45.6 | 39.6 | 37.4 | 36.67 | 30.84 | 5.83 | 3.46 | 0.61 | 30.9 / 31.0 | 22.4 |

## Criterion
STOP/DESCEND headroom on synthetic val = 0.51 ± 0.02° ≥ 0.25° → **PASS**.

## Caveats
- The oracle picks the best of 6 nested prefix centres using GT, which is optimistically biased: picking the minimum of noisy errors. A controller sees no GT, so the realisable gain is smaller.
- The mean error falls monotonically with depth (e5 < e4 < …). Stopping early only helps the images whose deep decisions went wrong. On synthetic val those are 27% of images, with small gains each.
- On real images the stop headroom is large (4–6°), because when the path is wrong a coarser cell centre is "less wrong". It is not a flip fix: > 90° errors stay > 90° at any depth.
- BRANCH (best of the beam-8 leaves) is larger still. That is the Phase-4 verifier problem (EXP-045 / EXP-105), not Phase 5.

## Decision
- Eligible to continue to EXP-051–055 (feature validation, rule-based and MLP controllers, fixed-vs-adaptive). They are **not started**: per DEC-014, they wait for the user's go.
- Recommendation: a controller could at most recover part of 0.5° on synthetic val. On real images the gain is "hedging" towards coarser cells, which leaves the flips unchanged. So this is lower priority than EXP-109 / EXP-108.
