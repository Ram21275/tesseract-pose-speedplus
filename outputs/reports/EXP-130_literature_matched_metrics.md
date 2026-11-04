# EXP-130: The literature's metrics (ADD(-S), AUC, AR_MSSD, AR_MSPD) on the finalised T-LESS and YCB-V systems

## Status
**Pre-registered 2026-11-04, before any run** (DEC-029). User request: "compare with sota and literature and add tables".

## Why
Published T-LESS and YCB-V results use pose metrics we had not computed. Rotation error in degrees and translation in cm cannot be compared with them, so YCB-V has no literature row in the paper today. This evaluation adds those metrics. No training and no new selection.

## Configurations (fixed: the EXP-129 configurations, 3 seeds, real test set only)
- **YCB-V:** EXP-126 checkpoints, DINOv3 1 view; DINOv3 translation, view 0. Test set: original YCB-V keyframes, our 6 objects (3,478 instances).
- **T-LESS:** EXP-126 checkpoints, A2 PoE + orbit_product; DINOv3 translation, orbit mean. Test set: BOP19 targets, our 7 objects (2,345 instances).
- **Also reported, not primary:** the other variant/inference pairs that the evaluator produces (greedy and orbit_product; PoE and DINOv3).
- **Translation and rotation are both predicted.** Crops are ground truth, which is a stated difference from published rows that use detections.

## Metrics
- **YCB-V (PoseCNN protocol):** ADD and ADD-S on `models/<obj>/points.xyz`.
  - ADD(-S) recall at 0.1 × diameter. All six objects are non-symmetric in that protocol, so ADD(-S) = ADD for them.
  - AUC of ADD and of ADD-S, threshold 0 to 10 cm.
  - Diameter: the maximum pairwise distance of the model points.
- **T-LESS (BOP protocol):** AR_MSSD and AR_MSPD on `models_eval`, using the BOP `models_info.json` symmetries.
  - Continuous symmetries discretised with the BOP toolkit rule (`ceil(pi × diameter / (0.01 × diameter))` = 315 steps).
  - MSSD thresholds 0.05:0.05:0.5 × diameter. MSPD thresholds 5:5:50 × (image width / 640) px.
  - **AR_VSD is not computed** (needs rendering against the depth images), so no overall BOP AR is claimed.
  - Note: objects 18 and 21 have no symmetries in BOP `models_info`. Our recall metric keeps its own symmetry set; AR uses the BOP set only.
- **Aggregation:** per object, then both object-balanced and instance-pooled means, per seed, then mean ± std over 3 seeds.

## Gate
The saved predictions must reproduce the EXP-129 seed-0 to seed-2 per-object `sym_mean` within 0.01°. If not, the evaluation is invalid.

## Reporting (fixed now)
- Descriptive. Every value is reported whatever the outcome.
- Published rows are labelled with their setting (RGB or RGB-D, detections or GT boxes, objects). Only rows with the same objects and metric are aligned, and the object subset is recomputed for a published per-object table where available.
- No "beats" claim across different crops or modalities.

## Results
Metrics run `outputs/experiments/EXP-130_literature_matched_metrics/RUN-20261104-150433-seed0`. **Gate passed:** max |d sym_mean| vs EXP-129 3.6e-15° (YCB-V, 72 rows) and 7.1e-15° (T-LESS, 84 rows). The test instances match the standard keyframes per object (868, 357, 379, 636, 1057, 181).

**YCB-V, DINOv3 1 view, GT crop, predicted R and t, 3 seeds (%):**

| | cracker | mustard | banana | mug | drill | scissors | mean ± seed std | instance-pooled |
|---|---|---|---|---|---|---|---|---|
| AUC ADD | 83.5 | 89.7 | 73.4 | 88.1 | 89.1 | 69.1 | 82.1 ± 1.8 | 84.8 |
| AUC ADD-S | 90.8 | 94.4 | 87.4 | 94.3 | 94.0 | 85.4 | 91.0 ± 0.8 | 92.1 |
| ADD 0.1d | 83.2 | 87.8 | 46.9 | 60.1 | 94.4 | 38.3 | 68.5 ± 2.0 | 76.6 |
| ADD-S 0.1d | 99.5 | 97.9 | 79.9 | 94.0 | 100 | 74.8 | 91.0 ± 1.8 | |

ADD < 2 cm: 70.1 ± 3.0. PoE 1 view is nearly identical (AUC ADD 82.3, ADD 0.1d 68.8).

**T-LESS, A2 PoE + orbit_product before distillation, GT crop, predicted R and t, 3 seeds (%):**

| | AR_MSSD | AR_MSPD |
|---|---|---|
| Instance-pooled (the BOP toolkit's definition) | 43.4 ± 1.0 | 85.4 ± 0.2 |
| Object-balanced | 51.8 ± 0.4 | 84.6 ± 0.2 |

Per object (1, 4, 5, 11, 18, 21, 27): MSSD 30.6, 46.0, 62.3, 50.9, 59.6, 41.0, 72.4; MSPD 84.0, 89.9, 92.0, 90.6, 78.7, 70.8, 86.5.

**Reading:**
- YCB-V: in the range of published fine-tuned networks on the same six objects (AUC ADD: GDR-Net 83.5, ZebraPose 88.3, CheckerPose 88.5), below the best, with a GT-crop advantage. Larger gap at ADD 0.1d (ZebraPose 86.1, CheckerPose 89.4).
- T-LESS: not matched (7 of 30 objects, GT crops). MSPD inside the trained-RGB range, MSSD below it. Depth error is the likely reason, not separately tested.
- Aggregation note found while writing: `eval_bop19_pose.py` in the BOP toolkit uses `scores["recall"]`, pooled over instances, so the instance-pooled row is the one defined like published AR. Pre-registration already required both.
- Limits: AR_VSD not computed, so no overall AR; AUC interpolation differs between papers.

Paper: `tab:ycbv_lit`, `tab:tless_ar`, `tab:tless_ar_perobj`, `tab:foundation_lit`, `tab:spp_lit`; extended `tab:symsol_sota`, `tab:trans_xobj`.

## Amendment 2026-11-04 (before any metric was read)
The first metrics run stopped at the gate because the script compared the predicted-translation predictions with the EXP-129 ground-truth-translation arm (YCB-V, max difference 0.16°). The reference is corrected to the matching EXP-129 arms: `RUN-20261103-165739` (YCB-V, view-0 translation) and `RUN-20261103-162450` (T-LESS, orbit-mean translation). The threshold (0.01°) is unchanged. No metric had been computed for review, because the assertion ran first.
