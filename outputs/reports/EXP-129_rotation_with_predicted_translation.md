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

## Amendment 2026-10-30 (before any run)
- **Translation source:** always the **DINOv3** translation block (`--pose-t-variant DINO`). This is the validation-selected translation branch on T-LESS and YCB-V (EXP-126) and the pairing used in the paper's Table A. Without the amendment, the PoE rotation variant would have averaged the DINOv3 and MoGe-2 blocks.
- **Code change:** the default (`same`) keeps every earlier evaluation unchanged.

## Results (completed 2026-11-03)
Runs (`EXP-129_rotation_with_predicted_translation`):

| dataset | `gt` | `greedy` | `orbit_mean` |
|---|---|---|---|
| T-LESS | RUN-20261103-153043 | RUN-20261103-160113 | RUN-20261103-162450 |
| YCB-V | RUN-20261103-164727 | RUN-20261103-165739 | RUN-20261103-170654 |

### Gate: PASS
- The `gt` arm reproduces the earlier numbers exactly: T-LESS A2 PoE orbit_product test 15.187 and validation 8.282; YCB-V DINOv3 greedy test 12.917 and validation 8.969.
- Under the ground-truth ray, ego_mean equals the allocentric error.

### Dependence on ground-truth translation (test; validation in parentheses)
Rotation is the symmetry-aware **egocentric** mean error in degrees. Joint is the share within a° **and** b cm.

| configuration | translation | rotation | ±seed std | acc@10 | joint 5°5cm | joint 10°5cm | joint 10°10cm |
|---|---|---|---|---|---|---|---|
| T-LESS PoE + orbit_product | GT | 15.187 (8.282) | 0.272 | 79.0 | – | – | – |
| | predicted, view 0 | 15.186 (8.288) | 0.281 | 79.0 | 44.5 | 72.2 | 78.0 |
| | **predicted, orbit mean (selected)** | **15.196 (8.284)** | 0.270 | 79.0 | **45.4** | **73.9** | **78.4** |
| T-LESS DINOv3 greedy | predicted, orbit mean | 16.111 | 0.296 | 73.0 | 31.8 | 68.9 | 72.4 |
| YCB-V DINOv3 greedy | GT | 12.917 (8.969) | 6.905 | 85.1 | – | – | – |
| | **predicted, view 0 (selected)** | **12.920 (8.942)** | 6.905 | 85.1 | **55.6** | **82.5** | **85.1** |
| | predicted, orbit mean | 12.932 (8.921) | 6.905 | 85.0 | 53.5 | 80.9 | 85.0 |
| YCB-V PoE + orbit_product | predicted, view 0 | 12.180 | 5.521 | 83.8 | 50.1 | 81.2 | 83.8 |

**Criterion 2 (negligible): PASS on both datasets.**
- T-LESS: |Δ| = 0.009 against a seed std of 0.27.
- YCB-V: |Δ| = 0.003 against a seed std of 6.9.
- The largest change in any arm is 0.07° (YCB-V validation).

**Why it is negligible:** translation errors of a few cm change the viewing ray by well under a degree at these object distances.
- The orbit candidate mapping is equally insensitive.
- Joint success at 10°/10 cm equals rotation acc@10, so translation almost never decides failure at that threshold.

### Scope caveat
- The translation block is attached to the EXP-106 A2 tree (EXP-126 checkpoints), so the T-LESS numbers are for the heads **before distillation**.
- The paper's primary T-LESS rotation result, the EXP-119 distilled students (14.50°, R@10 80.4%), was evaluated with GT translation only. The same conversion applies, but it was not re-measured.
- Table A therefore marks the T-LESS joint cell with ‡.

### Paper
- Table A 6D column (joint 5°5cm / 10°5cm);
- caption and overview paragraph;
- method disclosure (`sec/3_method.tex`), abstract and limitations;
- appendix experiment table row 129.
