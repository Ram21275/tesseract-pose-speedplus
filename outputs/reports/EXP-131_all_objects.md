# EXP-131: The shared architecture on all objects of T-LESS (30) and YCB-V (21), with full BOP AR and log-likelihood

## Status
**Pre-registered 2026-11-17, before any feature extraction, training or evaluation on the new object sets** (DEC-030).
User request: "I want to run full runs to mention in the supplement for tless and ycb use hdd3 RAM directory to store caches", then "queue them after that" (after the other project's SymSol training job) and "do not rerun on symsol just queue the tless and ycb".

Code gates run before this pre-registration was committed (metric code only, on the existing EXP-130 predictions; no new model):
the YCB-V symmetry-frame gate (below) and the regression of `scripts/exp131_ar.py` against EXP-130.

## Why
Every published T-LESS / YCB-V number averages over all objects (30 / 21) and reports BOP AR. Our paper results use 7 / 6 objects
and AR_MSSD / AR_MSPD only (no VSD), so they are not matched. This experiment runs the unchanged architecture and recipe on every
object and reports the full AR (with VSD) and the log-likelihood of the predicted rotation density, for the supplement.

## Question
With nothing re-tuned, what do the locked recipe and the configurations the paper selected (EXP-129/130) reach on all objects?
Descriptive; there is no pass criterion. Every value is reported whatever it is.

## Data (built by CPU scripts, committed with this pre-registration where small)
| dataset key | objects | train / validation / test | source |
|---|---|---|---|
| `tlessall` | all 30 | `train_pbr` scenes 0–44 + `train_primesense`, visib ≥ 0.3, ≤ 12,000 per object (rng 0) / `train_pbr` scenes 45–49 / every instance of the target objects in the BOP19 test images (6,733; BOP19 lists 6,423 targets; same rule as EXP-106) | `scripts/exp106_build_manifest.py --objects 1..30 --out tless_exp131` (the EXP-106 builder, unchanged rules) |
| `ycbvall` | all 21 | every 3rd frame of `train.txt` minus one held-out sequence per object (rng 0, sequence level), ≥ 1,000 visible px, ≤ 12,000 per object / held-out sequences / all keyframes (2,949 frames) | `scripts/exp064_ycbv_manifest.py --objects all --out ycbv_exp131` (EXP-064 rules) |
| `ycbvsynall` | all 21 | `data_syn` renders composited onto real `ycbvall` train frames, ≥ 1,000 px, ≤ 12,000 per object (train only) | `scripts/exp122_ycbv_syn.py --objects all` (EXP-122 rules), composites on hdd3 |

- **Not in git.** The manifests are about 0.5 GB (two files exceed GitHub's 100 MB limit). The builders are deterministic, so they are rebuilt with the commands above plus `scripts/exp131_splits.py tlessall ycbvall`. Committed: the builders' row-key hashes (`*.sha256.txt`, the cache identity), the file-byte sha256 prefixes of the split files (`exp131_splits.sha256.txt`), the check files and the symmetry files. Built sizes: `tlessall` 360,000 / 60,913 / 6,733; `ycbvall` 147,830 / 20,884 / 14,025; `ycbvsynall` 252,000 (12,000 × 21; 78,685 composited frames, 33,195 backgrounds). Builder determinism check: rebuilding the EXP-064 YCB-V manifest with the edited builder (default arguments) gives the committed `ycbv_exp064.csv` value for value (max abs difference 0; only float text formatting differs).
- **Fresh manifests for all objects.** The per-object caps and the YCB-V held-out sequences are drawn from one RNG in object order, so the rows of the original 7 / 6 objects differ from EXP-064. All objects are therefore retrained. The original 7 / 6 objects inside this run should reproduce the paper's numbers within seed noise; this is reported as a check, not a criterion.
- **Symmetries.** T-LESS: BOP `models_cad/models_info.json`, as EXP-106. YCB-V: BOP `models_eval/models_info.json`, converted to the original YCB-Video model frame by `scripts/exp131_ycbv_symmetries.py`. The BOP models are the original models shifted to their bounding-box centre. **Gate (passed):** the rigid ICP residual rotation is ≤ 0.13° on all 21 objects, and the shift-only fit is below the mesh's own sampling floor. Every converted symmetry maps the original mesh onto itself within 0.8 mm. Eight objects are symmetric: master chef can, bowl, wood block, large marker, both clamps, foam brick.
- **Labels.** Multi-path targets (A2, EXP-106) for every object of both datasets. With an identity-only symmetry set they are exactly the raw-label targets (same quaternion, encoding and soft targets), so the 13 non-symmetric YCB-V objects keep the EXP-122 recipe.
- **Storage.** New caches go to `/media/kavinder/hdd3/RAM/compact` (`extract_compact.py --cache-root`, symlinked into the shared cache) and composites to `/media/kavinder/hdd3/RAM/ycbv_syn_composite_all`. The existing hdd2 caches and the EXP-064 manifests are not touched.

## Models (the locked recipe, DEC-018; nothing tuned)
- Frozen DINOv3-L (`tokens8`) and MoGe-2 (`normals16~and`); clean view and 2 augmented training copies (photometric + exact camera-roll homography, seeds 1 and 2); orbit views 90/180/270° of validation and test.
- Per object, per branch: Transformer tree, depth 5, 100 epochs, batch 256, checkpoint by own-validation symmetry-aware greedy mean (`scripts/train_xobj.py --label multi`); YCB-V with `--extra-train ycbvsynall` (EXP-122).
- Translation block on each frozen tree (`--trans frozen`, EXP-126).
- Seeds 0, 1, 2. Run order is seed-major (all objects at seed 0 first). If time runs out, the DEC-012 cut order applies: seeds 1–2 are dropped for the remaining objects, and the report says so.
- **NaN rule (as EXP-064):** a tree whose training log shows a NaN validation value is retrained once with `--sd-floor 1e-2`.

## Configurations evaluated (fixed now: those of EXP-129 / EXP-130)
- **T-LESS:** A2 PoE + orbit_product; translation from the DINOv3 block, orbit mean (`--pose-t orbit_mean --pose-t-variant DINO`).
- **YCB-V:** DINOv3, 1 view (greedy); translation from the DINOv3 block, view 0 (`--pose-t greedy --pose-t-variant DINO`).
- GT crops (stated difference from published rows, which use detections). The other variant / mode pairs the evaluator produces are reported as secondary.

## Metrics (fixed now)
1. Symmetry-aware rotation error, R@2/5/10 (T-LESS), translation error, as in the paper's tables (`scripts/exp064_eval.py`).
2. **Full BOP AR** = mean(AR_VSD, AR_MSSD, AR_MSPD), instance-pooled (BOP toolkit) and object-balanced, `scripts/exp131_ar.py`:
   - MSSD / MSPD exactly as EXP-130 (BOP symmetries, 315 continuous steps).
   - VSD: BOP19 definition (δ = 15 mm, τ = 0.05–0.5 × diameter, θ = 0.05–0.5, distance images, visib_mode bop19), with depth rendered by an area-weighted **point-splat z-buffer** of the CAD mesh, because no OpenGL renderer is installed. **Gate:** GT poses rendered and scored against the real depth images must give AR_VSD ≥ 99 (mean over every 25th instance). The median GT-render vs measured-depth difference is reported, both over all overlapping pixels and over unoccluded pixels only (measured depth not more than δ in front of the render). The second is the frame check: occluders inflate the first.
   - YCB-V is reported on all keyframes and on the **BOP19 test-target subset** (4,123 instances; scene / image ids equal the original sequence / frame, verified).
3. **Log-likelihood** of the tree density at the GT, `scripts/exp131_ll.py`: p(R) = P(leaf) / vol(leaf) in the quaternion metric (uniform = −2.29), averaged over the symmetric GT copies (IPDF protocol), single view, selected variant.

## Reporting (fixed now)
- Supplement table: all-object AR (VSD / MSSD / MSPD / AR), the BOP19 subset for YCB-V, LL, rotation recall; ± seed std.
- Published rows are labelled with their setting (RGB or RGB-D, detections or GT crops, objects). No "beats" claim across different crops or modalities.
- The paper's 7 / 6-object numbers stay as they are. This experiment adds rows; it does not replace them.

## Code gates run before the queue (no new model)
- **AR regression on the EXP-130 predictions:** T-LESS MSSD 43.39 / MSPD 85.44 (EXP-130: 43.4 / 85.4), VSD 39.98, AR 56.27; YCB-V MSSD 83.98 / MSPD 82.34, VSD 73.39, AR 79.90. VSD gate 100 on both, median GT depth agreement 3.7 / 2.1 mm (runs `EXP-131_all_objects_ar/RUN-20261117-165540-seed0`, `RUN-20261117-170859-seed0`). The gate itself (GT scored against its own render) is nearly 100 by construction; the depth agreement is the informative part.
- **YCB-V frame check (GT render vs measured depth, 25 test instances each, unoccluded pixels):** bowl 1.2 mm, master chef can 2.6 mm, cracker box 2.3 mm, foam brick 1.9 mm; render in front of the measured surface by > 15 mm on 0.7–5.1 % of pixels.
- **Smoke test** (`scripts/exp131_smoke.sh`, 48/16/16 rows of T-LESS obj 27 and YCB-V bowl, 2 epochs, GPU1): every stage ran end to end (extraction to hdd3, A2 trees, translation blocks, eval with saved predictions, AR, LL). One fix came out of it: `extract_compact.py` capped the `and_mask` HDF5 chunk at the row count (no effect on values).

## Cost estimate (before running)
GPU (RTX 6000 Ada, after the other project's job ends): extraction about 25 h (T-LESS) and 24 h (YCB-V); training 180 + 126 trees plus translation blocks, about 23 h and 16 h; evaluation about 5 h and 4 h. CPU: manifests, composites, AR. Caches about 470 GB on hdd3.

## Commands
`scripts/run_queue21.sh` (tmux window `tesseract:exp131`). It waits for the other project's training process to exit, then runs T-LESS, then YCB-V. Finished runs are skipped on resume (`scripts/run_done.py`).

## Results
_Not run yet._
