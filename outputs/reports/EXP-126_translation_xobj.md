# EXP-126: Translation (6D) on T-LESS, SYMSOL-T and YCB-V

## Status
**Pre-registered 2026-10-28, before any run** (DEC-025 amendment; user: "translation only ran on speed+ which needs to be evaluated on other datasets as well").

## Question
EXP-118 evaluated the SPACE-HOP translation block on SPEED+ only. Does the same block, on the frozen trees selected for each other dataset, predict translation well beyond what the ground-truth box alone gives? How does it compare with LiePose's translation numbers?

## Design
- **Arm F (frozen tree + translation block), the only arm.** The rotation trees and their results stay exactly as reported, and only the block (2-layer attention + MLP head, as EXP-118) is trained.
  - SPEED+ showed joint training does not hurt rotation (EXP-118), but retraining every tree jointly would change the reported rotation results and double the cost.
- **Base trees:**
  - **T-LESS and SYMSOL-T:** the EXP-106 arm selected by the EXP-119 base-arm rule (lowest validation PoE greedy mean).
  - **YCB-V:** the EXP-122 trees (real + synthetic). Their translation block trains on the same rows.
  - 2 branches × seeds 0–2 per object.
- **Target:** SPACE-HOP u = (U_x, U_y, U_z), kept identical, with each dataset's own full image size (W, H): T-LESS 720 × 540, YCB-V 640 × 480, SYMSOL-T 224 × 224 (α = 1.6). Box = the GT square crop.
  - u is standardized per dimension with train statistics for the MSE loss (units differ: mm, m, scene units); predictions are de-standardized before inversion.
- **Views:** clean + the two augmented copies. The translation of a rolled view is t' = R(d, −θ) t (exact, as rotation). At test time: view 0 and the orbit mean (each view's t rolled back and averaged).
- **Training and selection:** 100 epochs, batch 256, AdamW 1e-3 / wd 0.05, one-cycle. Checkpoint = lowest validation mean E_T (view 0). Validation sets as EXP-064 (T-LESS PBR scenes 45–49; SYMSOL-T held-out train; YCB-V held-out real sequences).
- **Variants:** DINOv3 block, and PoE = the mean of the two branches' u (as EXP-118).

## Metrics (test reported for all; validation for selection)
- E_T mean / median in dataset units; normalized e_t = E_T / ‖t‖.
- T-LESS and YCB-V: translation recall T@2/5/10 cm (LiePose's T-LESS metric). The BOP symmetry transforms of our 7 T-LESS objects have zero translation, so te_sym = E_T.
- SYMSOL-T: mean Euclidean distance (LiePose Tab. 3 units).
- Baseline **box only**: U_x = U_y = 0, U_z = train mean, no image.

## Criteria (per dataset; validation)
1. **The block beats box-only:** DINOv3 validation mean E_T (orbit mean) below box-only by more than the larger seed std. DINOv3 is primary because it was selected on SPEED+ synthetic validation in EXP-118; PoE is secondary.
2. Descriptive:
   - orbit mean vs view 0;
   - T-LESS T@2/5/10 vs LiePose's per-object Tab. 8 averaged over our 7 objects (T@2/5/10 = 68.0 / 88.4 / 95.2);
   - SYMSOL-T vs LiePose Tab. 3 (0.011–0.016).
   Settings differ (frozen vs fine-tuned, visible mask), as for rotation.

## Cost
108 blocks on frozen trees, about 3–6 min each, so about 6–9 h; evaluations about 3 h. Runs in queue15 after EXP-124, before queue14 resumes EXP-119.

## Runs / Results / Decision
Not started.

## Results: YCB-V (evaluation `RUN-20261030-044920-seed0`, 2026-10-30)
Object-balanced over 6 objects, 3 seeds. E_T in cm; T@k in %.

| split | variant | mode | E_T mean | E_T median | T@2 | T@5 | T@10 |
|---|---|---|---|---|---|---|---|
| validation | DINOv3 | box only | 10.70 | 10.07 | 10.9 | 48.4 | 79.3 |
| validation | DINOv3 | view 0 | **2.73** ± 0.13 | 2.65 | 64.8 | 88.0 | 92.4 |
| validation | DINOv3 | orbit mean | 3.44 ± 0.33 | 3.24 | 60.7 | 84.6 | 90.6 |
| validation | PoE | view 0 | 3.00 ± 0.10 | 2.54 | 64.1 | 85.6 | 93.4 |
| test | DINOv3 | box only | 8.14 | 6.79 | 16.0 | 44.7 | 76.7 |
| test | DINOv3 | view 0 | **1.57** ± 0.11 | 1.27 | 73.0 | 96.0 | 99.9 |
| test | DINOv3 | orbit mean | 1.83 ± 0.17 | 1.55 | 68.8 | 92.9 | 99.6 |
| test | PoE | view 0 | 1.75 ± 0.02 | 1.43 | 70.6 | 93.9 | 99.4 |

- **Criterion 1: PASS.** DINOv3 validation orbit mean is 3.44 ± 0.33, against 10.70 for box only.
- **Descriptive:** orbit averaging is worse than view 0, as for YCB-V rotation (EXP-122). Validation selects DINOv3 view 0.
