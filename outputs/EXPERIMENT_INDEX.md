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
| EXP-013 | MoGe-2 single-image geometry controls (VGGT replacement candidate, user-approved) | Trainable predictor | Completed | grid4: synth-val mean 39.6° / median 21.2° (VGGT best 48.7° / 25.4°); normals16 best acc@10 of any geometric feature (0.179); foreground mask fails on real images | Adopted: MoGe-2 replaces VGGT (DEC-001) | [report](reports/EXP-013_moge2_controls.md) |
| EXP-014 | DINOv3–MoGe-2 complementarity (repeat of EXP-012) | Analysis | Completed | Synth-val: oracle-of-two acc@10 0.289 ± 0.020 vs VGGT 0.278 ± 0.022 (equal within noise); a non-geometric control (DINOv3-B) gives 0.308, so no geometry-specific complementarity shown | MoGe-2 adopted (DEC-001); whether to keep a geometric branch at all is re-tested at full data / Phase 2 | [report](reports/EXP-014_moge2_complementarity.md) |
| EXP-015 | Two-way DINOv3 ↔ MoGe-2 foreground consensus (user-requested) | Training-free | Completed | AND mask cuts MoGe-2 background leakage 0.52→0.12 (lightbox), 0.57→0.12 (sunlamp); MoGe-2 trims DINO's halo (small on real, large on synthetic) | Two-way by pre-registered rule, asymmetric; AND mask kept | [report](reports/EXP-015_mask_consensus.md) |
| EXP-016 | Does the consensus mask help pose, both directions? (user-requested) | Training-free fusion + trainable probe | Completed | MoGe-2 normals with consensus mask: mean 84.0→77.1° lightbox, 100.1→87.1° sunlamp; DINO masked pooling: no gain | One-way at pose level (DINO → MoGe-2) | [report](reports/EXP-016_consensus_pose.md) |
| EXP-017 | CASS option 1: spectral foreground from fused graph | Training-free | Completed | Fused-graph Fiedler mask worse than AND (leakage +0.10/+0.16, recall −0.10); splits faces, not figure/ground | Reject | [report](reports/EXP-017_spectral_foreground.md) |
| EXP-018 | CASS option 2: spectral attention injection, both directions | Training-free | Running | – | – | [report](reports/EXP-018_cass_attention_injection.md) |
| EXP-019 | CASS option 3: spectral part pooling for pose | Training-free features + probe | Planned (queued) | – | – | [report](reports/EXP-019_spectral_part_pooling.md) |
| EXP-020 | CASS option 4: spectral head-matching complementarity | Analysis | Planned (queued) | – | – | [report](reports/EXP-020_spectral_head_complementarity.md) |

Branch and complementarity summaries: `comparisons/branch_summary.csv`, `comparisons/complementarity_summary.csv`.
Decision log: `DECISIONS.md` (DEC-001: MoGe-2 replaces VGGT, 2026-10-08).
