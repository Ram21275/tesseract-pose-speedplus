# EXP-064: Broader-object validation of the selected Tesseract rotation method (T-LESS, SYMSOL-T, YCB-Video)

## Status
**Running** (T-LESS completed 2026-10-19; SYMSOL-T and YCB-V in progress).
**Planned; data located (2026-10-13), waiting on the Phase-6 prerequisites.** The user chose T-LESS, SYMSOL-T and YCB-Video (DEC-009).
- *Correction:* the 2026-10-10 note said neither BOP YCB-V nor BOP T-LESS was on the local disks. That search missed `/media/kavinder/hdd2`, where all three datasets are (details and deviations in DEC-009: YCB-Video is the original PoseCNN release, not BOP; the local T-LESS test set is the BOP19 subset; SYMSOL-T was rendered locally). Prerequisites, all in Phase 6: the rotation architecture is selected (Phases 3–4), the adaptive-depth decision is made (Phase 5), and EXP-060–063 are done. ID and placement: DEC-007.

## Research question
Does the selected Tesseract rotation method keep its benefits (fusion gain, accuracy–compute advantage) on objects other than the spacecraft?

## Scope
- **This is** cross-object validation of the method **with pose-head retraining**. It is not zero-shot transfer of the spacecraft-trained predictor.
- **Out of scope:** true held-out-object generalization without retraining. That is separate future work and needs explicit CAD/template or reference-frame conditioning.

## Protocol to pre-register in full before any result (once the data is present)
- **Datasets:** T-LESS (BOP; textureless, symmetric), SYMSOL-T (synthetic symmetric shapes with translation; exact symmetry-equivalent rotations annotated) and YCB-Video (original release; varied everyday objects). Record the exact versions, the download sources and the camera/model conventions.
- **Object subset:**
  - declared **before** any evaluation, with written criteria covering appearance (textured vs textureless), geometry, and symmetry class (none / discrete / continuous);
  - initially a manageable number per dataset, fixed thereafter.
- **Training setup:**
  - same frozen encoders, selected architecture and recipe;
  - **fresh per-object heads**, kept fixed for all arms;
  - trained on each benchmark's permitted training split;
  - RGB with GT/controlled crops; no automatic localization;
  - camera, rotation, object-frame and unit conventions verified by reprojection, as in EXP-000.
- **Splits:** validation split by scene/sequence (never adjacent frames), independent of test. Normalization statistics and hyperparameters come from train/val only.
- **Comparisons** (matched features, training budget, refinement capacity and compute):
  1. DINO-only vs the selected fusion;
  2. Tesseract vs a properly matched Hopf baseline, retrained under the same budget (not the old subset-trained EXP-101 Hopf);
  3. discrete vs tangent refinement vs the selected flow method, if retained;
  4. fixed vs adaptive depth, if retained.
- **Metrics:**
  - per-object results and object-balanced aggregates;
  - geodesic error for asymmetric objects;
  - for symmetric objects, raw **and** documented symmetry-equivalent error, kept clearly separate;
  - BOP symmetry-aware metrics only where compatible with rotation-only evaluation, never presented as a 6DoF benchmark entry;
  - severe failures (> 90°, > 150°), coverage if applicable, runtime, memory.
- **Decision question:** do the fusion benefit and Tesseract's accuracy–compute advantage persist across objects?

## Runs / Results / Decision
Running since 2026-10-18 21:10 (`scripts/run_exp064.sh` via queue11; user: EXP-064 before EXP-117). Results are added per dataset as each block finishes. Full T-LESS results are below; SYMSOL-T and YCB-V follow.

### Runs
- Convention gate: `RUN-20261018-211031` (all three datasets verified, label error 0.000°).
- T-LESS: 42 trainings (7 objects × 2 branches × seeds 0–2; `stage: train`, 2026-10-18 23:xx – 2026-10-19 06:32), evaluation `RUN-20261019-063238` (`tables/object_balanced.md`, `tables/per_object.md`, `metrics/per_object_seed.csv`).
- Compact-feature gate: `RUN-20261019-065522` (strict), functional follow-up `RUN-20261019-065802`. See "Deviation" below.

### Deviation: compact-feature gate ran late and FAILED as written
- The pre-registered "gate before use" (512 SPEED+ crops, compact vs full caches, max abs diff ≤ 1e-2 after standardization, mask agreement ≥ 0.99) was **not run before use**, neither for EXP-109 (whose augmented and orbit views come from the same compact extractor) nor for EXP-064. It was run on 2026-10-19, **after** the T-LESS results.
- **Strict result: FAIL.** `scripts/exp064_compact_gate.py` (standardized by the gate rows' own sd):
  - DINOv3 `tokens8`: max 0.0108 (11 of 33.5 M elements > 1e-2; p99 0.002; corr 0.99999996);
  - MoGe-2 `normals16~and`: max 0.51 (1,196 of 524 k elements > 1e-2, in 97 of 512 images; p99 0.004; corr 0.999998). 34 of those 97 images have a consensus-mask difference; the rest are small MoGe-2 output differences;
  - consensus-mask pixel agreement 0.99998 (passes).
  - Standardized instead with the models' synthetic-train statistics (`exp064_compact_functional.py`, 20 k-row subsample): DINOv3 max 0.011 (48 elements > 1e-2), MoGe-2 max 0.55 (1,197). Still FAIL.
- **Functional follow-up** (reading fixed in the script before running: mean angle < 0.1° and ≤ 1% changed leaves ⇒ no impact). EXP-109 arm-A trees (seed 0) on the same 512 crops, compact vs full features, same statistics:
  | predictor | mean angle | p99 | max | changed greedy leaves |
  |---|---|---|---|---|
  | DINOv3 greedy | 0.010° | 0.000° | 5.10° | 1 / 512 |
  | MoGe-2 greedy | 0.031° | 0.000° | 6.91° | 3 / 512 |
  | PoE greedy | 0.000° | 0.000° | 0.000° | 0 / 512 |
  **No impact** by the pre-set reading. The gate FAIL is kept on record. EXP-064 is internally consistent anyway (all its features come from the compact extractor). For EXP-109 / 113–116, mixing full-cache view 0 with compact augmented / orbit views changes at most a few single-branch predictions per thousand and no PoE prediction on this sample.

### T-LESS results (7 objects, object-balanced, symmetry-aware error, 3-seed mean ± seed std, degrees)
Validation is `train_pbr` scenes 45–49 (synthetic, PBR); test is the BOP19 targets (real Primesense images).

| variant | inference | val mean | val median | val acc@10 | test mean | test median | test acc@10 | test > 90° |
|---|---|---|---|---|---|---|---|---|
| DINOv3 | greedy | 12.67 ± 0.06 | 8.07 | 0.660 | 18.59 ± 0.18 | 8.43 | 0.620 | 0.052 |
| DINOv3 | orbit product | 9.94 ± 0.06 | 6.45 | 0.766 | 17.88 ± 0.35 | 7.43 | 0.674 | 0.053 |
| DINOv3 | orbit vote32 | 10.10 ± 0.12 | 6.15 | 0.769 | 17.90 ± 0.47 | 7.10 | 0.687 | 0.054 |
| PoE | greedy | 12.31 ± 0.13 | 7.37 | 0.702 | 17.92 ± 0.16 | 7.38 | 0.681 | 0.055 |
| PoE | **orbit product** | **9.25 ± 0.07** | 5.80 | **0.808** | **16.57 ± 0.25** | 6.33 | 0.735 | 0.051 |
| PoE | orbit vote32 | 9.65 ± 0.11 | **5.61** | 0.805 | 16.65 ± 0.34 | **6.08** | **0.736** | 0.053 |

Per object, PoE + orbit product (val / test): obj 1 5.36 / 23.82; obj 4 5.42 / 14.50; obj 5 7.04 / 6.22; obj 11 9.49 / 11.90; obj 18 16.81 / 14.38; obj 21 13.33 / 34.78; obj 27 7.31 / 10.36.

**Criterion (fusion helps on T-LESS): PASS.**
- PoE val mean is below DINOv3-only by more than the larger seed std: greedy 12.31 vs 12.67 (Δ 0.37 > 0.13). With the locked orbit inference: 9.25 vs 9.94 (Δ 0.69 > 0.07). PoE is better in every seed.
- **Test, reported as is:** PoE is better in every mode (orbit product 16.57 vs 17.88).

**Descriptive:**
- **Orbit voting transfers.** It lowers val by about 3° for both variants (PoE 12.31 → 9.25) and test by about 1.3° (PoE 17.92 → 16.57). The orbit product and the weighted vote are within noise of each other, as on SPEED+ (EXP-114).
- **Synthetic-to-real gap.** Test is much worse than val (16.6 vs 9.3) and dominated by tails:
  - test median 6.3° and acc@10 0.74, but > 90° fraction 0.05 overall;
  - obj 21 (no symmetry metadata): test 34.8°, 18% of predictions > 90°;
  - obj 1 (continuous symmetry): test 23.8°, 8.6% > 90°.
  Unlike SPEED+, orbit voting removes few of the test-domain gross errors (> 90° fraction 0.055 → 0.051).
- **The MoGe-2 branch alone is weak on T-LESS:** own-val greedy 19–63° vs DINOv3 7–22°. Yet PoE with it still helps every seed.
- Raw (non-symmetry-aware) means are 38–47°, which confirms that the symmetry-aware metric is required for these objects.
- **External numbers are unmatched** (GT crops, rotation only, 7 objects, frozen features).


## Full protocol (pre-registered 2026-10-15, before any cross-dataset feature extraction or model result)
Schedule: the end of Phase 6, after the recipe lock (DEC-011/012). The "locked recipe" means the decision entry that closes Phase 6 SPEED+ work. Below it is written for the current frozen predictor (EXP-036 tree, both branches, equal-weight per-level PoE, greedy). Whatever the lock adds is applied unchanged.

### Data (manifests built and checked 2026-10-15; no model was run on them)
| dataset | objects (declared now) | train / val / test | input and target | checks |
|---|---|---|---|---|
| T-LESS (BOP) | 1, 4 (continuous sym.), 5, 11 (one 180°), 18, 21 (no metadata sym.), 27 (three discrete): same set as EXP-106 | `train_pbr` scenes 0–44 + `train_primesense`, visibility ≥ 0.3, ≤ 12,000 per object / `train_pbr` scenes 45–49 / BOP19 targets (2,345) | RGB GT square crop of the projected model box; **allocentric** rotation | reprojection round trip 3e-13 px; projected CAD vertices vs BOP `bbox_obj` IoU median 0.977 (min 0.93) |
| SYMSOL-T | tet, cube, icosa, cone, cyl | `train-45k` minus 2,000 per shape / 2,000 per shape held out (rng 0) / `test-5k` (5,000 per shape) | **LiePose protocol**: non-colour render, full 224² frame, egocentric quaternion | symmetry groups (12 / 24 / 60 / 360 / 720) recovered and verified constant |
| YCB-Video (original release) | 003_cracker_box, 006_mustard_bottle, 011_banana, 025_mug, 035_power_drill, 037_scissors. Criteria: box / bottle / elongated / near-symmetric (the mug's handle is the closest analogue of SPEED+ flips) / thin. Declared-symmetric YCB objects are excluded: this release has no symmetry metadata | every 3rd frame of `train.txt` minus one held-out sequence per object (rng 0), ≥ 1,000 visible px, ≤ 12,000 per object / held-out sequences / keyframes (2,949 frames) | RGB GT square crop of projected `points.xyz`; allocentric rotation | reprojection round trip exact; points box vs label box IoU checked |

### Features
- Frozen DINOv3-L and MoGe-2 through `scripts/extract_compact.py` (stores only `tokens8`, the DINO PCA foreground, and MoGe-2 normals and mask), cached on hdd2.
- **Gate before use:** on 512 SPEED+ crops, the compact `tokens8` and `normals16~and` features must match the full SPEED+ caches. Tolerance: max abs difference ≤ 1e-2 after standardization, and consensus-mask agreement ≥ 0.99. Otherwise extraction is fixed first.

### Models (heads retrained; nothing tuned per dataset)
- **Per-object** heads (per shape for SYMSOL-T), as the original protocol says: the EXP-036 recipe exactly (Transformer tree, depth 5, geodesic soft targets on the raw annotated rotation, 100 epochs, batch 256, checkpoint by own-val greedy mean). One tree per branch. Normalization statistics from that object's train split.
- System = equal-weight per-level PoE, greedy (plus any stage the recipe lock adds).
- Seeds 0–2. If time runs short, SYMSOL-T and YCB-V drop to seed 0 first (DEC-012 cut order); T-LESS keeps 3 seeds.

### Comparisons
1. **DINO-only vs PoE fusion:** both branches are trained anyway.
2. Tesseract vs a matched flat Hopf classifier (EXP-101 head, same features and budget): stage 2, second in the cut order.
3. Flow verifier per object: stage 3, third in the cut order.

### Metrics
- SYMSOL-T and T-LESS: **symmetry-aware** rotation error (min over the equivalence set; T-LESS continuous symmetries in 36 steps). Raw error reported separately.
- YCB-V: raw error.
- Per object and object-balanced: mean, median, acc@2/5/10 (LiePose) and @15/30, fraction > 90°.
- Runtime from EXP-107.

### Criterion (decision question)
Per dataset, **fusion helps** if the object-balanced PoE val mean (symmetry-aware where defined) is below DINO-only by more than the larger seed std. The test split is reported for every arm, whatever the direction.

### External context
LiePose (SYMSOL-T, T-LESS), SpyroPose, Implicit-PDF and PoseCNN rotation numbers are listed as **unmatched**, unless split, crop, symmetry metric and object set all match. SYMSOL-T uses LiePose's own split and image, so it is the closest to matched (differences: frozen features, and our rotation-only target vs their SE(3)).

## Amendment 2026-10-16 (DEC-018, before any run): locked recipe
This experiment uses the **EXP-109 methodology** (augmentation-trained trees, with the in-plane rotation label transform derived for the dataset's camera / crop convention; orbit PoE voting at test time). It is reported in **both** variants, PoE (DINOv3 + MoGe-2) and DINOv3-only, each with and without orbit voting. The in-plane label transform for each dataset is derived and verified by projecting its 3-D model points, before any training, as in EXP-109 (`scripts/exp109_convention_geometry.py`). Other protocol details are unchanged.

## Amendment 2026-10-18 (before any EXP-064 feature extraction, training or evaluation)
1. **Label convention, derived and verified** (`scripts/exp064_convention.py`, `RUN-20261018-160444`):
   - **SYMSOL-T:** full 224² frame, principal point at the centre, no distortion, so the in-plane rotation about the image centre is an exact camera roll. Label error 0.000°.
   - **T-LESS / YCB-V:** the first check with the SPEED+-style affine crop rotation **failed** the pre-set rule (`RUN-20261018-160329`: label error p95 1.43° / 2.40°, max 2.5° / 6.7°; objects far off-axis). The augmented and orbit views therefore use the **exact camera-roll homography** K·Rot(d, −θ)·K⁻¹ about the crop-centre ray (`inplane.crop_rolled_homography`, extractor `--roll-homography`). Labels follow exactly: egocentric R′ = Rot(d, −θ)R, t′ = Rot(d, −θ)t, then the allocentric target recomputed from t′ (`tfpose.xobj`). Verified label error 0.000° and pixel error 0.000; mapping round trip ≤ 0.004°.
2. **SYMSOL-T training cap:** 20,000 per shape (the first 20k by the manifest's seeded rank), instead of 43k, for the GPU budget. Validation (2k per shape) and test-5k are unchanged.
3. **Recipe (DEC-018):**
   - per-object (per-shape) trees, 2 branches × seeds 0–2, EXP-036 architecture and budget;
   - clean + 2 augmented copies (photometric + in-plane roll, seeds 1 / 2);
   - normalization statistics from the object's clean train rows;
   - **checkpoint by the symmetry-aware validation greedy mean** (the raw mean is ill-defined for symmetric objects).
4. **Inference:** both variants (PoE, DINOv3-only); greedy, orbit product and orbit weighted vote (4 views). For allocentric datasets, candidates are mapped between views using the GT translation, consistent with the GT-crop protocol.
5. **Comparisons kept:** DINOv3-only vs PoE (criterion unchanged), and greedy vs orbit (descriptive). Flat-Hopf and flow stages are **not run** here (cut order, DEC-012). The val-aug proxy is not used.
6. Code: `scripts/run_exp064.sh`, `train_xobj.py`, `exp064_eval.py`, `src/tfpose/xobj.py`, `featsets.load_rows`. Caches on `/media/kavinder/hdd2/RAM/compact` (about 170 GB).
