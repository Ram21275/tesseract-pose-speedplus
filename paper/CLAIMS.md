# Claim → evidence map (2026-10-30, DEC-028)

Every number in the abstract, introduction and conclusion must come from this file. "Val-selected" means the configuration with the lowest validation error, chosen on validation data only (DEC-006).

## C1. Reusable frozen-backbone 6D architecture, small per-object heads (method + measured cost)
- **What is shared:**
  - frozen DINOv3 ViT-L/16, 303.1 M parameters, 1.13 GiB fp32;
  - optional frozen MoGe-2 ViT-L, 330.9 M parameters, 1.23 GiB;
  - one head architecture, one training recipe, one inference algorithm.
  - Source: `outputs/comparisons/cost_inventory.json` (`scripts/cost_inventory.py`).
- **What is trained per object:**
  - rotation tree: 2.39 M (DINOv3 branch) / 2.13 M (MoGe-2 branch) parameters;
  - translation block: 0.33 M parameters;
  - checkpoints 8.4–11.2 MB.
  - Training takes a median of 5–13 min per head on one shared GPU, from cached features. Exceptions: SPEED+ distilled students take 40–84 min; joint SPEED+ 6-DoF takes 12–25 min.
  - Source: `cost_inventory.json` (`training_min`, `checkpoint_MB`).
- **What is computed once per dataset:** feature extraction, 2.2–4.3 GPU-hours per backbone over all views (`cost_inventory.json`, `extraction`). For detected crops, a detector is also trained: 6.3 h on SPEED+ (EXP-124).
- **What a new application must supply:**
  - object identity and a crop (GT box, or a trained detector);
  - pose labels in a fixed object frame;
  - camera intrinsics;
  - the symmetry set if the object is symmetric;
  - for YCB-V, synthetic renders (EXP-122).
- **Not claimed:** zero-shot transfer. Frozen SPEED+ heads fail on other objects (EXP-064 zero-shot). One universal predictor is not claimed either: there is one head per object, and options are chosen on validation.
- **Inference cost** (EXP-112, SPEED+ crops, RTX 6000 Ada shared, bf16, batch 1, end-to-end incl. crop warp):

| Configuration | Time | Throughput |
|---|---|---|
| DINOv3 1-view | 14.7 ms | 68 FPS |
| DINOv3 orbit | 69.6 ms | 14 FPS |
| PoE 1-view | 64.0 ms | 15.6 FPS |
| PoE orbit | 278.5 ms | 3.6 FPS |

  - At batch 1 the tree decoding (7.4 ms) costs about as much as the DINOv3 backbone (6.4 ms).
  - Most PoE time is MoGe-2 (21 ms) plus the CPU consensus mask (20 ms).
  - Head-only cost per precision is EXP-128, which is **pending**.

## C2. Hierarchical rotation formulation
- **Established, architectural:**
  - head size is independent of grid resolution (2.4 M at 131,072 cells, vs a flat head of 33.6 M output parameters);
  - greedy decoding scores 44 nodes, against 131k anchors for a flat head;
  - quantisation floor 2.51° mean (EXP-101/127).
- **Established, empirical (SPEED+):** the tree beats a flat classifier over SPACE-HOP's 131,076-anchor Hopf grid under the same recipe. PoE + orbit: 3.87° vs 7.53° synthetic, better on real domains, with 15× fewer parameters (EXP-113).
  - **Caveat:** that grid's own floor is 6.06° (EXP-127 quantisation), so the comparison does not isolate hierarchy.
- **Pending:** the same-grid flat and balanced-Hopf controls (EXP-127). No claim that hierarchy itself causes the accuracy gain until then.

## C3. Symmetry-aware multi-path supervision and its relation to orbit aggregation (EXP-106)
- **T-LESS** (7 objects, real BOP19 test), PoE + orbit:
  - R@2/5/10 rises from 4.0/36.0/73.5 (A0) to **5.7/48.1/79.0** (A2);
  - mean error from 16.6° to 15.2°;
  - criterion (validation, PoE greedy): 10.82 vs 11.46 (A1). PASS.
- **SYMSOL-T** (5 shapes), DINOv3 orbit test: 2.62° → **2.07°**, better on every shape. Criterion 2.72 vs 3.40: PASS.
- **Relation to orbit aggregation:** canonical labels (A1) help a single view but break orbit voting (SYMSOL-T orbit 9.73°; T-LESS validation orbit 10.71° vs A0 9.25°). Multi-path targets keep orbit voting effective.

## C4. Orbit aggregation and distillation
- **Orbit voting** (exact roll transform; scores hierarchical candidates over 4 views):
  - **helps:** T-LESS validation 10.82 → 8.28 (PoE); SYMSOL-T 2.54 → 2.07 (DINOv3); SPEED+ synthetic 4.85 → 3.87, lightbox/sunlamp 17.3/20.2 → 12.3/12.3 (EXP-109; repairs 62/80% of > 150° flips, breaks 0.4%);
  - **hurts:** YCB-V validation 8.97 → 11.53 (DINOv3; EXP-122).
  - **Cost:** 4.7× (DINOv3) / 4.4× (PoE) end-to-end at batch 1.
- **Translation averaging over orbit views:**
  - helps on T-LESS (validation 2.14 → 1.85 cm) and SYMSOL-T (0.082 → 0.061);
  - hurts on YCB-V (2.73 → 3.44 cm) (EXP-126).
- **Distillation (SPEED+ only):**
  - students improve single-view greedy (val-aug PoE 9.75 → 9.22, DINOv3 9.88 → 9.33; criterion PASS) and the orbit system;
  - retention of the orbit gain is 13–43%;
  - cross-dataset distillation (EXP-119) is **pending**.
- **Scoring rule:** on SPEED+ synthetic validation, the weighted vote beats the pre-registered likelihood product (EXP-114 criterion FAIL; distilled 3.27 vs 3.69).

## C5. Cross-dataset findings
- **Appearance–geometry complementarity is dataset-dependent.** Val-selected variant per dataset:

| Dataset | Selected | PoE | DINOv3 |
|---|---|---|---|
| SPEED+ | PoE | lightbox orbit 12.3° | 17.4° |
| T-LESS | PoE | val orbit 8.28 | 8.61 |
| SYMSOL-T | DINOv3 | 2.10 | 2.07 |
| YCB-V | DINOv3 | 10.52 | 8.97 |

- **Training data:** YCB-V needs synthetic renders. Real-video-only trees fail (test 69–81°); with renders the val-selected test error is 12.9°, 85% within 10° (EXP-122).
- **Translation from the same tokens** (frozen tree + 0.33 M block; EXP-126; real test sets; val-selected):

| Dataset | T@2/5/10 cm | Mean error | Box only |
|---|---|---|---|
| T-LESS | 57.3 / 88.0 / 96.7% | 2.6 cm | 6.3 cm; LiePose 68.0 / 88.4 / 95.2% |
| YCB-V | 73.0 / 96.0 / 99.9% | 1.6 cm | 8.1 cm |
| SYMSOL-T | – | 0.061 units | 0.958; LiePose 0.011–0.016 |

  On SPEED+ (EXP-118), the jointly trained DINOv3 block (orbit mean) gives 0.185/0.183 m on lightbox/sunlamp; the box alone gives 0.55 m.
- **Detected crops (SPEED+, EXP-124):**
  - S1 val-selected (vote32): 24.96°/16.90°, score 0.541/0.341;
  - pre-registered orbit_product: 23.88°/16.42°, 0.523/0.333;
  - versus GT crops 0.229/0.239 (val-selected) / 0.227/0.224.
  - The lightbox detector is the bottleneck (IoU 0.76).

## Headline numbers (val-selected, primary)
- **T-LESS:** PoE + orbit, test R@2/5/10 5.7/48.1/79.0, mean 15.2°, translation T@2/5/10 57.3/88.0/96.7 (orbit mean).
  - Uses **GT translation** for the frame conversion and orbit mapping; EXP-129 is pending.
- **SYMSOL-T:** DINOv3 + orbit, 2.07°, translation 0.061.
- **YCB-V:** DINOv3, 1 view, 12.9°, 85% within 10°, translation 1.6 cm. Uses GT translation for the frame conversion only; EXP-129 is pending.
- **SPEED+:** distilled PoE + vote32 (val-selected):
  - GT crop: 11.35°/11.91°, score 0.229/0.239;
  - detected crop: 24.96°/16.90°, score 0.541/0.341.
  - The pre-registered orbit_product rows are reported as the alternative.

## Pending (blank cells only)
- EXP-119 (cross-dataset distillation);
- EXP-127 (same-grid flat, balanced Hopf, timing and memory);
- EXP-128 (head-only latency and quantisation);
- EXP-129 (no-GT-translation rotation and joint success).
