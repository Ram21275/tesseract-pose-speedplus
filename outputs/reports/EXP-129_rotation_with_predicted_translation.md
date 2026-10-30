# EXP-129: Rotation and joint pose on T-LESS and YCB-V without ground-truth translation

## Status
**Pre-registered 2026-10-30, before any run** (DEC-028). Queued in `scripts/run_queue17.sh`, which starts after the `QUEUE16_DONE` marker. This keeps the authorized order (EXP-119 → EXP-128 → EXP-127) unchanged.

## Why
On T-LESS and YCB-V the tree predicts the **allocentric** rotation. Every earlier evaluation (EXP-064, EXP-106, EXP-122) used the **ground-truth translation** at test time in two places:
1. the conversion R_ego = R_ray(t) R_allo, implicitly, because errors were measured between allocentric targets, which equals the egocentric error only under the ground-truth ray;
2. the orbit mapping of candidates between rotated views (`XO.roll_target` / `from_target` with `tvec`).

SPEED+ and SYMSOL-T use egocentric targets, and their roll axis is the crop-centre ray, so they need no translation. EXP-126 now provides predicted translation for T-LESS and YCB-V, so the dependence can be removed and measured.

## Arms (evaluation only; `scripts/exp064_eval.py --pose-t`)
- **Models:** the EXP-126 checkpoints, which hold the frozen rotation tree (unchanged EXP-106 A2 / EXP-122 weights) plus its translation block.
- **`--pose-t gt`:** the earlier protocol, as a reproduction gate. The rotation errors must match EXP-106 A2 (T-LESS) and EXP-122 (YCB-V) within 0.02°, and the new egocentric error must equal the allocentric one.
- **`--pose-t greedy`:** view-0 predicted translation used for the conversion and the orbit mapping.
- **`--pose-t orbit_mean`:** the orbit-averaged predicted translation, used the same way.

## Metrics
- **Rotation:** symmetry-aware **egocentric** error (`ego_mean`, `ego_acc@2/5/10`).
- **Joint pose success:** the fraction with egocentric rotation ≤ a° **and** translation error ≤ b cm, for (a, b) = (5, 5), (10, 5), (10, 10).
- **Coverage:** validation and test, PoE and DINOv3, greedy / orbit_product / orbit_vote32, 3 seeds, object-balanced.

## Criteria (fixed now)
1. **Gate:** the `gt` arm reproduces the earlier rotation numbers. If it does not, the evaluation is invalid.
2. **Dependence (descriptive with a rule):** the change from GT to predicted translation is "negligible" if |Δ ego_mean| ≤ the larger seed std. This is judged for the validation-selected configuration of each dataset:
   - T-LESS: PoE orbit_product, with orbit_mean translation;
   - YCB-V: DINOv3 greedy, with greedy translation.
3. **Paper headline:** the configuration that uses no ground-truth information at test time becomes the primary rotation result for T-LESS and YCB-V, whatever the outcome. GT-translation results are kept as an upper reference, clearly labelled.

## Results
Not run.
