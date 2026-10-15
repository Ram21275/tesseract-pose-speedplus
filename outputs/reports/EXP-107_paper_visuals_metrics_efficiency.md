# EXP-107: Paper visualizations, comparison-ready metrics, and efficiency (steps, samples, FPS)

## Status
**Pre-registered 2026-10-13, before any figure or number** (DEC-012, user request). Analysis only: no model is trained and nothing is selected here. Runs after EXP-105. CPU parts read saved outputs; GPU parts (NFE sweep, timing, trajectories) go through the guarded queue, one job at a time.

## Purpose
Show what the Tesseract tree + PoE and the SO(3) flow do, in the visual language of LiePose / Implicit-PDF. Report metrics in the forms other papers use, so external numbers can be placed next to ours (labelled matched or unmatched). Measure speed against steps and samples.

## Inputs (frozen)
EXP-036 trees (PoE), EXP-042 fused flow heads, and the saved fused samples (EXP-042 `RUN-20261010-075303-seed0`, EXP-043 `RUN-20261013-004254-seed0`, 3 seeds, M = 32). SPEED+ synthetic val, lightbox, sunlamp (DEC-006). The LiePose reference code (`~/liepose-diffusion/src/liepose/utils/vis.py`) is read for style only. Our plotting is reimplemented in matplotlib (no JAX dependency).

## Figures (example images chosen by a fixed rule, not by eye)
Example rule: per domain, the image at the 10th, 50th and 90th percentile of PoE greedy error, plus the lightbox image with the largest flow probability mass within 30° of the 180° flip of GT (seed 0, image index ties broken by the lowest index).
- **F1 SO(3) distribution plot (LiePose / Implicit-PDF style):**
  - Mollweide projection of the rotation's first column, with the in-plane (tilt) angle as hsv colour and a tilt colour wheel.
  - The GT is drawn as an open ring.
  - Flow samples are dots. For plotting only, 256 samples are drawn (not the evaluation M = 32).
  - The fused PoE tree is shown as the top-64 leaf centres, with marker area ∝ leaf probability, side by side with the flow.
- **F2 Sampling trajectories:** the same projection with 16 sample paths over τ = 0 → 1 (NFE 10), plus an mp4 (ffmpeg, as LiePose `encode_video`) for supplementary material.
- **F3 Coarse-to-fine tree:** fused PoE mass per level (root and levels 1–5) on the projection, showing how the probability concentrates down the hierarchy, flip modes included.
- **F4 Speed–accuracy curves:**
  - Flow mean error and acc@5/10 against NFE ∈ {1, 2, 5, 10, 20, 50} (M = 32) and against M ∈ {1, 4, 8, 16, 32, 64} (NFE = 10).
  - Latency on the same axes, with the tree (one pass) marked as a point.
- **F5 Error histograms** per domain, log count, 0–180° (as SPACE-HOP Fig. 4 / LiePose), tree vs flow.
- **F6 Calibration:** nominal vs empirical coverage of the flow's credible regions (0.5–0.95), per domain.

## Metrics table (comparison-ready; all on GT crops, rotation only)
- **E_R** = 2·arccos(|⟨q_pred, q_gt⟩|) in degrees (identical to SPACE-HOP / the ESA challenge rotation error): mean, median, p95.
- acc@2/5/10/15/30° (LiePose uses 2/5/10; I-PDF uses 15/30), and the fraction > 90° and > 150°.
- **Spread (Implicit-PDF):** mean over flow samples of the geodesic distance to GT.
- **Log-likelihood of GT:**
  - tree: log(leaf probability / leaf-cell volume), with cell volume from the Haar measure of the cell, estimated by Monte Carlo with 10⁶ uniform rotations, seeded;
  - flow: the exact CNF log-density at GT (the EXP-105 `flow_logp` code).
  - Both are reported in nats against the uniform density on SO(3).
- Coverage at 0.9, and best-of-M (oracle, labelled).
- External rows (SPACE-HOP, SPNv2, VPU, FastPose-ViT; LiePose on SYMSOL/T-LESS later in EXP-064) are copied from the papers. Each is labelled unmatched unless the crop, prediction target and test split all match.

## Efficiency (EXP-063 shares these numbers)
- Per component, with `torch.cuda.synchronize()` around each, after a warm-up:
  - DINOv3-L and MoGe-2 feature extraction;
  - each tree encoder + greedy decoding;
  - PoE;
  - flow per NFE step;
  - KDE.
- Batch 1 (latency, FPS = 1/latency) and batch 64 (throughput, images/s). Peak GPU memory and parameter counts (frozen / trainable).
- The GPU is shared with other users' jobs, so each timing is the median of 5 repeats with the concurrent GPU load recorded. Numbers are stated as upper bounds on an unloaded GPU. The hardware is named. No Jetson measurement (not available), so SPACE-HOP's Orin latency is unmatched.

## Criteria
None: this is a descriptive experiment. Everything is reported whatever the outcome. Example images follow the rule above.

## Amendment 2026-10-15 (before any run)
- The F4 speed–accuracy sweep uses **seed 0 only**, all three domains (`scripts/run_exp107_sweep.sh`, `exp04x_fused_eval.py --fused-only --feat-cpu --queries 1024`). The 3-seed point at NFE 10 / M 32 is EXP-042's. KDE mode at M = 1 is the single sample itself.

## Runs / Results
Not started.
