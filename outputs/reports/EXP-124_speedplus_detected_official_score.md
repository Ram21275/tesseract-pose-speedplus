# EXP-124: SPEED+ with a detector and the official score

## Status
**Pre-registered 2026-10-28, before the detector was trained or any detected-crop result existed** (DEC-025, user: "then get paper ready results, which are comparable to sota" → "Core matched set").

## Question
All our SPEED+ numbers so far use ground-truth (GT) crops and report mean errors. Published methods use detected boxes or full images, and the official SPEED+ score. What do our systems score on the **full** lightbox (6,740) and sunlamp (2,791) test sets with a detector trained on synthetic images only, and how much of the GT-crop accuracy survives?

## Detector (`scripts/exp124_yolo_data.py`, `exp124_train_det.py`)
- Ultralytics YOLO11s (COCO-pretrained). One class: the tight box of the 11 projected Tango keypoints, which is the box the GT crops are built from.
- Training: synthetic/train only. Selection: synthetic/validation fitness only. Lightbox and sunlamp are test-only (DEC-006).
- Fixed settings:
  - images stored at 960 × 600 (half resolution); imgsz 960;
  - 20 epochs, batch 32, seed 0;
  - flips lr / ud 0.5; no rotation or shear;
  - mosaic, closed for the last 5 epochs;
  - brightness jitter hsv_v 0.6; the Ultralytics default albumentations.
- Keypoint file: the hdd2 copy of `tangoPoints.mat` (the SPEED+ toolkit). It reproduces every manifest crop to < 5e-6 px; the original hdd copy is unmounted since the reboot.

## Crops from detections (`scripts/exp124_detect.py`)
- Top-1 box (conf ≥ 0.001), scaled to full resolution.
- Square crop with the GT-crop rule: centre of the box, side = max(w, h) × 1.3, minimum 32 px.
- No detection → whole-image crop, counted.
- Features (DINOv3-L `tokens8`, MoGe-2 `normals16~and`, consensus masks) are extracted for the detected crops and their 90 / 180 / 270° in-plane views, with the same extractor and roll rule as `exp109_test3_r*`.

## Systems (fixed now; none is re-selected on test data)
- **S1 (primary):**
  - rotation: EXP-117 PoE students (DINOv3 + MoGe-2, PoE teacher), orbit_product over 4 views;
  - translation: EXP-118 joint (J) DINOv3 translation block, orbit mean.
  - The DINOv3 block is the one synthetic validation selects (0.059 vs 0.093 m).
- **S2:** EXP-118 J PoE (trees and translation blocks of one model family).
- **S3 (light):** EXP-117 DINOv3 student (DINO teacher) + EXP-118 J DINOv3 translation.
- Also reported: greedy and orbit_vote32 rotation; single-view translation. Seeds 0–2, mean ± std.

## Metric (SPEED+ paper Eqs. 1–4; SPNv2 Eq. 9)
- e_q = 2 arccos|⟨q̂, q⟩| (rad); e_t = ‖t̂ − t‖ / ‖t‖; score = mean over the domain's images of (e_q + e_t).
- Thresholds: e_q < 0.169° → 0 and e_t < 2.173e-3 → 0, applied per component as in SPNv2 / Kelvins (primary).
- The SPEED+ Eq. 4 joint rule is also reported as a check (the sources disagree on the rule; at our error levels the difference is negligible).
- Also reported: E_R and E_T mean and median, the fraction of E_R > 150°, and detection IoU / centre error / missed detections per domain.
- GT-crop runs of the same systems with the same code serve as the reference. The GT-crop S1 rotation must reproduce EXP-117 (11.21 / 11.09°) and the S1 translation must reproduce EXP-118 J DINOv3 (0.185 / 0.183 m); otherwise the evaluator is wrong and the run is not reported.

## Criteria
1. **Benchmark result (descriptive):** S1 detected-crop score per domain. It is placed against published numbers with each method's setting stated (detector / full image, domain adaptation, test-time refinement, real images in training). There is no pass threshold; it is a measurement.
2. **Detection gap:** S1 detected − GT score per domain, reported.
3. All systems and modes are reported whatever the direction.

## Published context (verified 2026-10-28 from primary sources; lightbox / sunlamp)
| Method | E_R (deg) | E_t (m) | Score | Setting |
|---|---|---|---|---|
| SPNv2 φ=3 BN, offline | 6.48 / 11.07 | 0.175 / 0.225 | 0.142 / 0.230 | full image, no adaptation (AdvSR 2024, Table 5) |
| SPNv2 φ=6 GN + ODR | 5.58 / 9.79 | 0.150 / 0.161 | 0.122 / 0.197 | + test-time entropy minimisation on unlabelled test images |
| SPEC2021 TangoUnchained / VPU / lava1302 | – | – | 0.0735 / 0.0900; 0.1014 / 0.0612; 0.1647 / 0.0589 | keypoints + PnP; top teams used real or test-domain images (Kelvins leaderboard) |
| FastPose-ViT ViT-B-384 | 12.26 / 18.28 | 0.270 / 0.400 | n.r. | LW-DETR detector, synthetic only (WACV 2026, Tables 3–4) |
| PAID-ViT | 12.29 / 11.76 | 0.135 / 0.151 | 0.237 / 0.230 | crop-based (arXiv 2606.11619, Table I) |
| Church & Nikolian, DINOv3-H+ heatmaps + PnP, ensemble + TTA | 1.17 / 1.56 | – | 0.0273 / 0.0383 | **GT crops**, synthetic only (arXiv 2609.26561, Table II) |
| SPACE-HOP-224 | 9.02 / 26.65 as printed | 0.302 / 0.282 | n.r. | its baseline rows have lightbox and sunlamp swapped relative to the sources, so its own domain labels need checking |

## Runs / Results / Decision
Not started.

## Incident 2026-10-28/29 (infrastructure, no protocol change)
- The first detector training (`RUN-20261028-235010`) was aborted at 23:58 by memguard: "process RSS 44.3 GB > limit 34.0 GB". The scope's real memory peak was 11 GB.
- Cause: memguard summed the RSS of the 6 forked DataLoader workers, which share most pages with the parent.
- The abort killed only the main process. The orphaned workers kept queue15's output pipe open, so queue15 stalled for about 14.7 h with no retry.
- Fix in `src/tfpose/memguard.py`, applying to all jobs:
  - children are counted by USS;
  - on abort, memguard kills the job's child processes.
- The kernel cap (systemd scope `MemoryMax`) and the system available-memory floor are unchanged.
- The orphans were killed by PID at 14:44 on 2026-10-29. The queue retried with the identical pre-registered settings.

## Results (completed 2026-10-29)
- **Detector:** `RUN-20261029-145416`, YOLO11s, 20 epochs; synthetic validation mAP50-95 0.993, mAP50 0.995.
- **Detection on the test sets:** `RUN-20261029-211323`. A miss falls back to the full image and is still scored.

| domain | n | misses | IoU mean | IoU median | IoU < 0.5 | centre error (px) |
|---|---|---|---|---|---|---|
| synthetic val | 11,994 | 0 | 0.986 | 0.990 | 0.0% | 13.2 |
| lightbox | 6,740 | 54 | 0.764 | 0.844 | 15.7% | 66.8 |
| sunlamp | 2,791 | 0 | 0.884 | 0.915 | 2.0% | 36.4 |

- **Scores:** `RUN-20261029-232454-seed0` (`--crops det`), 3 seeds. S1 is the primary system: distilled PoE students with orbit_product for rotation, the J DINOv3 translation block with orbit_mean. Values are synthetic val / lightbox / sunlamp.

| system | crops | E_R (deg) | E_T (m) | score |
|---|---|---|---|---|
| S1 | GT (`RUN-20261028-231916`) | 3.69 / 11.21 / 11.09 | 0.058 / 0.185 / 0.183 | 0.075 / 0.227 / 0.224 |
| S1 | **detected** | 3.88 / **23.88 / 16.42** | 0.067 / **0.662 / 0.270** | 0.081 / **0.523 / 0.333** |
| S2 (J PoE, orbit) | detected | 4.04 / 24.74 / 17.10 | 0.101 / 0.718 / 0.288 | 0.090 / 0.547 / 0.348 |
| S3 (DINOv3 student + J DINOv3 translation) | detected | 3.97 / 24.68 / 17.27 | 0.067 / 0.662 / 0.270 | 0.082 / 0.537 / 0.347 |

- **Criterion 1 (descriptive):** S1 detected-crop score 0.523 (lightbox) / 0.333 (sunlamp). Matched setting: synthetic-only training, detector, no adaptation. SPNv2 offline (full image) scores 0.142 / 0.230, so we are far behind it in this setting.
- **Criterion 2: detection gap** (S1 detected − GT score) is **+0.296** on lightbox and **+0.109** on sunlamp.
  - Rotation is 2.1× worse on lightbox and 1.5× worse on sunlamp. Translation is 3.6× worse on lightbox.
  - Translation is box-relative (SPACE-HOP u), so box errors pass directly into E_T.
  - The lightbox detector is the bottleneck: 16% of its boxes have IoU < 0.5.
- **Descriptive:**
  - orbit_vote32 has the lowest synthetic-val E_R for S1 (3.50 vs 3.88). It would be the synthetic-validation choice, with lightbox / sunlamp 24.96 / 16.90 and scores 0.541 / 0.341.
  - Severe-flip rates (> 150°) for S1 orbit_product: lightbox 5.3%, sunlamp 3.1%.
- **Decision:** the paper reports the detected-crop row as measured and states that a real-domain detector, or box-independent translation, is needed for deployment.
