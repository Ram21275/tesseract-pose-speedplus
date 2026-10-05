# EXP-000: Immutable SPEED+ loader and coordinate conventions

## Status
Completed

## Research question
Does the loader read the original SPEED+ files correctly, and which quaternion/rotation convention maps the Tango body frame into the camera frame?

## Hypothesis
`q_vbs2tango_true` is scalar-first, and either R(q) or R(q)ᵀ maps body points into the camera frame. Only one of the two will land the projected Tango keypoints on the spacecraft.

## Component under test
Other (data loader / conventions)

## Track
Training-free (no learning)

## Fixed setup
- Dataset and exact split: SPEED+ (`/media/kavinder/hdd2/RFO/speed_pose_moe/speedplusv2`, read-only); synthetic/train.json, synthetic/validation.json, lightbox/test.json, sunlamp/test.json
- DINOv3 variant/layer: n/a
- VGGT variant/output/layer: n/a
- Predictor: n/a
- Tesseract depth and codebook size: n/a
- Input resolution and preprocessing: full 1920×1200 grayscale images; projection uses `cv2.projectPoints` with `camera.json` K and the 5 distortion coefficients
- Seeds: 0 (image sampling)
- Compute device: CPU
- Git commit: `9586198` (clean code tree); first run at `4f2118d` (scripts were not yet committed)
- Flow base distribution, solver, NFE and endpoint samples, if applicable: n/a

## Changed variable
Rotation convention: `R` (p_cam = R(q)·p + r) versus `RT` (p_cam = R(q)ᵀ·p + r).

## Method
1. Load all four split JSONs and check counts, missing images, quaternion norms and translation range. Assert that no image path is duplicated across splits.
2. Sample 200 random images per domain. Project the 11 Tango keypoints (`tangoPoints.mat`) with GT pose under both conventions, including lens distortion.
3. Score each convention three ways:
   - the fraction of keypoints inside the image;
   - the fraction of keypoints on bright (spacecraft) pixels;
   - for synthetic images, the IoU of the keypoint bounding box against an independently derived bbox table (`train.csv` from an earlier project, used only as a cross-check).
4. Save overlays for visual inspection.
5. Run numerical tests: identity, axis rotations versus `cv2.Rodrigues`, 180° rotations, and the matrix↔quaternion round trip. These are also in `tests/test_geometry.py`.

## Commands
```
python scripts/exp000_loader_conventions.py --n-per-domain 200 --seed 0
```

## Runs
| Run ID | Seed | Status | Runtime | Artifact directory |
|---|---:|---|---:|---|
| RUN-20261005-173109-seed0 | 0 | Completed (code uncommitted at run time) | 13.3 s | outputs/experiments/EXP-000_loader_conventions/RUN-20261005-173109-seed0 |
| RUN-20261005-173947-seed0 | 0 | Completed (clean commit; metrics bit-identical to the first run) | 8.4 s | outputs/experiments/EXP-000_loader_conventions/RUN-20261005-173947-seed0 |

## Quantitative results
Split integrity (`metrics/splits.csv`):

| domain | split | n | missing images | ‖q‖ min | ‖q‖ max | t_z range (m) |
|---|---|---:|---:|---:|---:|---|
| synthetic | train | 47,966 | 0 | 0.999999 | 1.000001 | 2.24–10.00 |
| synthetic | validation | 11,994 | 0 | 0.999999 | 1.000001 | 2.23–10.00 |
| lightbox | test | 6,740 | 0 | 1.000000 | 1.000000 | 2.47–9.55 |
| sunlamp | test | 2,791 | 0 | 1.000000 | 1.000000 | 2.48–9.53 |

No image path is duplicated across splits.

Convention test, 200 images per domain (`metrics/convention.csv`):

| domain | convention | keypoints inside image | keypoints on bright pixels | bbox IoU vs cross-check (mean / min) |
|---|---|---:|---:|---|
| synthetic | **R** | 0.959 | **0.482** | **0.970** / 0.546 |
| synthetic | RT | 0.955 | 0.261 | 0.519 / 0.194 |
| lightbox | **R** | 0.981 | **0.271** | – |
| lightbox | RT | 0.985 | 0.179 | – |
| sunlamp | **R** | 0.978 | **0.560** | – |
| sunlamp | RT | 0.969 | 0.331 | – |

Numerical tests (`metrics/metrics.json`):
- identity: OK;
- quaternion→matrix→quaternion maximum error: 2.96e-6°;
- maximum deviation from `cv2.Rodrigues`: 1.6e-16.

## Domain-wise results
Convention R is preferred in all three domains (table above).

## Qualitative results
Overlays are in `figures/overlay_*.jpg`: green is convention R, red is RT, with the body axes drawn. Green keypoints sit on the bus corners and antenna tips in synthetic, lightbox and sunlamp images; red keypoints fall off the body.

## Resource results
- Parameters: n/a
- Trainable parameters: 0
- Peak memory: CPU only (see `status.json`)
- Mean/median latency: n/a
- Tesseract nodes visited: n/a
- Flow sampling latency and NFE, if applicable: n/a
- Parent-child mass consistency error, if applicable: n/a
- Credible-region coverage, if applicable: n/a

## Comparison with control
Convention RT is the control. R wins on every measure that discriminates (brightness, bbox IoU). Both conventions keep most points inside the image, so that measure does not discriminate.

## Interpretation
Observed conventions:
- Quaternions are scalar-first `[w, x, y, z]` (Hamilton).
- Object-to-camera transform: `p_cam = R(q)·p_body + r_Vo2To_vbs`.
- Camera frame is OpenCV: x right, y down, z forward.
- Images are distorted, so distortion must be applied when projecting.

All labels are unit-norm to within 1e-6. These conventions are recorded in `src/tfpose/data.py` (`ROTATION_CONVENTION = "R"`).

The minimum bbox IoU of 0.55 under R is not a convention problem. The cross-check table was built by another project with an unknown recipe, which may clip to the image or use undistorted projection.

## Failure analysis
None.

## Decision
Keep. Convention R is fixed for all later experiments.

## Next experiment
EXP-001: ground-truth to Tesseract path.

## Artifact index
- Metrics JSON: `outputs/experiments/EXP-000_loader_conventions/RUN-20261005-173947-seed0/metrics/{splits,convention,metrics}.json`
- CSV: same directory, `*.csv`
- TSV: same directory, `*.tsv`
- Figures: `.../RUN-20261005-173947-seed0/figures/overlay_*.jpg`
- Predictions: n/a
- Checkpoint, if any: n/a
- Logs: `status.json`, `environment.txt`, `command.txt`
