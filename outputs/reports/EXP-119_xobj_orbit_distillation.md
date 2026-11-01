# EXP-119: Orbit distillation on T-LESS and SYMSOL-T

## Status
**Pre-registered 2026-10-27, before any run** (DEC-024, user: "do 1,2,3"). Runs after EXP-106.

## Question
EXP-117 showed on SPEED+ that a single-view tree distilled from the orbit system is better both alone and as the base of orbit voting. Does the same hold on T-LESS and SYMSOL-T?

## Base arm (fixed now, before EXP-106 results)
Per dataset, the EXP-106 arm (A0 / A1 / A2) with the lowest validation symmetry-aware **PoE greedy** mean (object-balanced, 3 seeds). The teacher trees, the student's ground-truth term (the arm's label mode) and the comparison baseline all come from that arm.

## Teacher (frozen, offline; `scripts/xobj_teacher.py`)
- The base arm's trees of the same object and seed. Orbit over each training crop's three available views (clean + `exp064_<ds>_aug1/2`, known roll angles). No new extraction.
- Candidates: beam-8 leaves of each view mapped exactly to the clean target frame (24). Weights: softmax of the orbit-product score.
- Variants as in EXP-117: a **PoE** teacher (DINOv3 + MoGe-2) and a **DINO-only** teacher.

## Students (`scripts/train_xobj.py --teacher … --beta 1`)
- Loss: L_GT(v) (the base arm's label mode) + β · level-marginal CE to the teacher candidates, transported into the view's frame. **β = 1** (as EXP-117).
- Three students per object and seed:
  - a DINOv3 student from the DINO teacher (DINOv3-only system);
  - DINOv3 and MoGe-2 students from the PoE teacher (PoE system).
- Same recipe, budget, seeds and checkpoint rule as the base arm. SYMSOL-T MoGe-2: `--sd-floor 1e-2` for cube / cyl / icosa, as in the base.

## Metrics
Symmetry-aware error (as EXP-064 / 106): greedy, orbit_product and orbit_vote32; validation and test; object-balanced, 3 seeds; acc@10 and fraction above 90°.

## Criteria (per dataset; validation)
1. **Distillation helps:** the student's single-view greedy validation mean is below the base arm's greedy mean by more than the larger seed std. DINOv3-only is primary, PoE secondary (as EXP-117).
2. Descriptive: student + orbit vs base + orbit; retention = (base greedy − student greedy) / (base greedy − base orbit).
3. Test reported for every arm and mode.

## Claim boundary
As EXP-117: self-distillation from test-time-augmentation ensembles is known. The question is whether the SPEED+ gain transfers.

## Cost
Teachers: 2 variants × 3 seeds × 12 objects, small. Students: 108 trees, about 6–10 min each, so about 12–16 h. Evaluation: about 2 h.

## Runs / Results / Decision
Not started.

## Results: SYMSOL-T (2026-10-31)
- **Evaluation runs:**
  - PoE students: `RUN-20261031-112902-seed0`;
  - DINOv3 students: `RUN-20261031-120353-seed0`.
- **Base:** EXP-106 A2 (`RUN-20261028-120353`).
- **Format:** object-balanced over 5 shapes, 3 seeds; symmetry-aware mean in degrees ± seed std.

| variant | inference | base val | student val | base test | student test |
|---|---|---|---|---|---|
| DINOv3 | greedy | 2.539 ± 0.009 | **2.520** ± 0.005 | 2.565 | 2.527 |
| DINOv3 | orbit_product | **2.071** ± 0.003 | 2.156 ± 0.003 | **2.067** | 2.167 |
| PoE | greedy | 2.717 ± 0.011 | **2.629** ± 0.010 | 2.714 | 2.621 |
| PoE | orbit_product | **2.100** ± 0.007 | 2.264 ± 0.004 | **2.107** | 2.265 |

- **Criterion 1: PASS for both variants, but small.**
  - DINOv3 (primary): Δ 0.019 > 0.009.
  - PoE: Δ 0.088 > 0.011.
- **Descriptive:**
  - **Retention** = (base greedy − student greedy) / (base greedy − base orbit): DINOv3 4%, PoE 14%.
  - **Unlike on SPEED+, the students are a worse base for orbit aggregation:** student + orbit is 0.085 (DINOv3) and 0.164 (PoE) worse than base + orbit on validation.
  - **Validation selection is unchanged:** SYMSOL-T keeps the base A2 DINOv3 + orbit (2.071 validation, 2.067 test).
- **Evaluator note:** first live run of `exp064_eval.py` after the EXP-129 `--pose-t` edit. The `gt` code path is unchanged in effect (T0 = tvec), SYMSOL-T uses egocentric targets, and the values fall in the range of the base trees.

## Results: T-LESS (2026-11-01)
- **Evaluation runs:**
  - PoE students: `RUN-20261101-065144-seed0`;
  - DINOv3 students: `RUN-20261101-071036-seed0`.
- **Base:** EXP-106 A2 (`RUN-20261028-221625`).
- **Format:** object-balanced over 7 objects, real BOP19 test targets, 3 seeds; symmetry-aware mean in degrees ± seed std.

| variant | inference | base val | student val | base test | student test |
|---|---|---|---|---|---|
| DINOv3 | greedy | 10.893 ± 0.076 | **10.111** ± 0.032 | 16.11 | 15.42 |
| DINOv3 | orbit_product | 8.609 ± 0.015 | **7.970** ± 0.051 | 16.06 | 14.98 |
| PoE | greedy | 10.818 ± 0.110 | **10.076** ± 0.103 | 15.71 | 15.20 |
| PoE | orbit_product | 8.282 ± 0.045 | **7.875** ± 0.096 | 15.19 | **14.50** |

- **Criterion 1: PASS for both variants.**
  - DINOv3 (primary): Δ 0.78 > 0.076.
  - PoE: Δ 0.74 > 0.110.
- **Descriptive:**
  - **Retention** = (base greedy − student greedy) / (base greedy − base orbit): DINOv3 34%, PoE 30%.
  - **Unlike on SYMSOL-T, the students are also a better base for orbit aggregation:** PoE 8.28 → 7.88 on validation, 15.19 → 14.50 on test.
  - **Test recall** (PoE + orbit) at 2/5/10° is 6.5 / 48.1→50.8 / 79.0→80.4, against 5.7 / 48.1 / 79.0 for A2.
  - **Validation selection:** T-LESS now selects distilled PoE + orbit (7.875 validation).
- **Overall:** distillation transfers to SPEED+ and T-LESS but not to SYMSOL-T.
