# Experiment index

All Phase-1 numbers come from the **subset v1** sanity runs. The subset has 6,000 synthetic-train, 1,500 synthetic-val, 1,000 lightbox and 1,000 sunlamp images, all with GT crops.
- **Train:** synthetic/train only.
- **Model selection:** synthetic/validation only.
- **Scored only:** lightbox and sunlamp, which are never used for selection or tuning.

| ID | Question | Track | Status | Best result | Decision | Report |
|---|---|---|---|---|---|---|
| EXP-000 | SPEED+ loader and pose convention | Training-free | Completed | Convention R (p_cam = R(q)p + r), bbox IoU 0.97 vs 0.52 for Rᵀ | Keep | [report](reports/EXP-000_loader_conventions.md) |
| EXP-001 | GT quaternion → Tesseract path | Training-free | Completed | All invariants hold on 69,491 labels | Keep | [report](reports/EXP-001_gt_tesseract_paths.md) |
| EXP-002 | Precision by depth | Training-free | Completed | L5: 2.51° mean, 3.97° p95, 6.20° exact max; volume ratio 14.4× | Keep; L_max = 5 | [report](reports/EXP-002_depth_precision.md) |
| EXP-003 | Tesseract vs Hopf vs Super-Fibonacci | Training-free | Completed | Less precise per hypothesis (needs 1.2–1.6× the points of Super-Fibonacci for equal p95), cheapest hierarchy (44 nodes at L5) | Keep Tesseract; Hopf as matched baseline | [report](reports/EXP-003_grid_comparison.md) |
| EXP-010 | DINOv3-only representation + MLP predictor | Trainable predictor | Completed | ViT-L grid4: synth-val median 17.2°, lightbox 62.2°, sunlamp 70.3° (3 seeds) | Keep as primary branch | [report](reports/EXP-010_dinov3_baseline.md) |
| EXP-011 | VGGT token / geometry controls | Trainable predictor | Completed | l11 tokens / grid4: synth-val median 25–27°, lightbox 84–105°; weaker than DINOv3 everywhere, 17× slower | Keep conditionally for EXP-012 | [report](reports/EXP-011_vggt_controls.md) |
| EXP-012 | DINOv3–VGGT complementarity and frequency audit | Analysis | Completed (one earlier attempt crashed the machine; fixed) | Synthetic val: oracle-of-two acc@10 +7.7 pts, agreement makes DINO correct 2.6× more often (real-domain gains small, descriptive only) | VGGT kept conditionally for Phase 2 (decided on synthetic val only) | [report](reports/EXP-012_complementarity.md) |
| EXP-013 | MoGe-2 single-image geometry controls (VGGT replacement candidate, user-approved) | Trainable predictor | Completed | grid4: synth-val mean 39.6° / median 21.2° (VGGT best 48.7° / 25.4°); normals16 best acc@10 of any geometric feature (0.179); foreground mask fails on real images | Keep as geometric branch candidate | [report](reports/EXP-013_moge2_controls.md) |
| EXP-014 | DINOv3–MoGe-2 complementarity (repeat of EXP-012) | Analysis | Completed | Synth-val: oracle-of-two acc@10 0.289 (VGGT 0.278); agreement makes DINO correct 3.2× more often on 36.7% of images (VGGT 2.6× on 29.8%) | Replace VGGT with MoGe-2 for Phase 2 (decided on synthetic val) | [report](reports/EXP-014_moge2_complementarity.md) |

Branch and complementarity summaries: `comparisons/branch_summary.csv`, `comparisons/complementarity_summary.csv`.
