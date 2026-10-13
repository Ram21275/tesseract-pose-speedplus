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
