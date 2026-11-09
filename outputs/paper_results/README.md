# Per-image inference results behind the paper

Each row is one test image under one system, with columns:
- ground truth: `q_gt_*`, `t_gt_*`;
- predicted rotation, 1 view and orbit: `q_greedy_*`, `q_orbit_product_*` (SPEED+ also `q_orbit_vote32_*`);
- predicted translation: `t_pred_*`;
- symmetry-aware rotation error in degrees: `err_rot_deg_*`;
- translation error: `err_t` (T-LESS mm, YCB-V m, SYMSOL-T scene units, SPEED+ m).

Quaternions are wxyz and egocentric. Written by `scripts/export_paper_predictions.py`; `index.csv` lists the source runs.

| File | System | Seeds | Check against the paper |
|---|---|---|---|
| `tless_per_image.csv.gz` | EXP-130: A2 trees before distillation, PoE and DINOv3; predicted translation (orbit mean) | 0–2 | PoE orbit, object-balanced test: 15.19° (EXP-129: 15.196) |
| `ycbv_per_image.csv.gz` | EXP-130: synthetic-trained trees, DINOv3 and PoE; translation view 0 | 0–2 | DINOv3 1 view: 12.92° (EXP-129: 12.920) |
| `symsolt_per_image.csv.gz` | FIG-qualitative: A2 DINOv3 tree; translation orbit mean | 0 | seed 0 only |
| `speedplus_gt_per_image.csv.gz`, `speedplus_det_per_image.csv.gz` | S1: distilled PoE students, orbit (product and vote) + joint DINOv3 translation; GT and detected crops | 0 | seed 0: 11.25° / 11.57° on GT crops (the paper reports the 3-seed mean, 11.21° / 11.09°) |

The full prediction files, which also hold all 32 orbit candidates and their scores, are `.npz` files under `outputs/experiments/*/predictions/`. They are git-ignored because of their size (about 320 MB).
