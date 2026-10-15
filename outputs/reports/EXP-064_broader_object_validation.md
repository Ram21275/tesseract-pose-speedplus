# EXP-064: Broader-object validation of the selected Tesseract rotation method (T-LESS, SYMSOL-T, YCB-Video)

## Status
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
Not started (data located; waiting on Phase-6 prerequisites).

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
