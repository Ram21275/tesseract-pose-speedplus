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
| EXP-012 | DINOv3–VGGT complementarity and frequency audit | Analysis | Completed (one earlier attempt crashed the machine; fixed) | Oracle-of-two acc@10 +7.7 pts synthetic, +0.9–2.5 lightbox; agreement makes DINO correct 2.6–6× more often | VGGT kept conditionally for Phase 2 | [report](reports/EXP-012_complementarity.md) |
