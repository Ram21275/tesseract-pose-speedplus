# EXP-030: Phase-3 MLP hierarchical predictor baseline on the full SPEED+ splits (frozen DEC-003 design)

## Status
Running. Pre-registered on 2026-10-09, before any result.

## Research question
With the frozen design, does training on the full synthetic training split (47,966 images, about 8× the subset) improve each branch and the PoE fusion? Does the fusion gain hold at full data? This becomes the Phase-3 control for EXP-031–035.

The frozen design (DEC-003) is a training-free PoE of a DINOv3-L `grid4` probe and a MoGe-2 `normals16~and` probe.

## Component under test
Predictor (training data scale) and the frozen fusion

## Track
Trainable predictor + training-free fusion

## Pre-registered protocol
- **Manifest `full`:** all of synthetic/train (47,966), synthetic/validation (11,994), lightbox (6,740) and sunlamp (2,791). GT crops as in subset v1 (15% margin).
- **Features:** DINOv3-L/16 (`grid4`), MoGe-2 ViT-L normal checkpoint, and the EXP-015 consensus masks (AND rule, unchanged), all recomputed on `full`.
- **Probes:** the EXP-010/016 hierarchical MLP, unchanged hyperparameters (hidden 512, 100 epochs, AdamW 1e-3 / 0.05, batch 256, hard labels, L5). 3 seeds; selection on synthetic-val mean.
  - Only change: features are stored on the GPU as fp16 (`--gpu-dtype fp16`) because of shared-GPU memory. The model computes in fp32.
- **Fusion:** equal-weight PoE, greedy (EXP-025 script, `--subset full`).
- **Metrics:** mean / median geodesic error, acc@5/10/20, per domain. Lightbox and sunlamp are descriptive.
- **Comparison:** against the subset-v1 runs (EXP-010/016/025), same seeds.
- **Safety:** every step runs alone under `scripts/guarded.sh` (24–32 GB caps) in tmux.

## Results
(to be filled)
