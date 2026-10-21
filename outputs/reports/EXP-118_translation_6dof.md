# EXP-118: Full 6-DoF: a translation block with the SPACE-HOP formulation on the Tesseract token encoder

## Status
**Completed 2026-10-20.** Pre-registered 2026-10-19, before any code or run. Queued after the current queue (EXP-064, then EXP-117), per user: "after all these queue the translation test with the space-hop formulation keep it same then our whole architecture encorporates the translation block as well which recieves the dino or moge tokens as an input then we need to run the evaluations for that as well, it will make the paper for 6dof as we haven't tested for translation till now."

## Question
Can the locked architecture (DEC-018: frozen DINOv3-L `tokens8` / MoGe-2 `normals16~and`, one Tesseract tree per branch, per-level PoE, orbit inference) be extended to full 6-DoF with a translation block on the same tokens? Does the block add translation information beyond what the crop box already gives, without hurting rotation?

## Translation formulation (SPACE-HOP, Dutta et al., CVPRW 2026, §3.3; kept identical)
Box (b_x, b_y, b_w, b_h) = our crop (crop_cx, crop_cy, crop_side, crop_side); full image W_full = 1920, H_full = 1200; camera K (f_x, f_y, c_x, c_y); α = 1.6.
- s_x = W_full / (α·b_w), s_y = H_full / b_h; **U_z = ½(1/s_x + 1/s_y)·Z** (eqs 7–8)
- **U_x = (f_x·X/Z + c_x − b_x) / b_w**, **U_y = (f_y·Y/Z + c_y − b_y) / b_h** (eqs 9–13)
- Target u = (U_x, U_y, U_z). Loss: L_trans = mean ‖u_pred − u_gt‖² (eq 18). Inverse mapping: Z = U_z / ½(1/s_x + 1/s_y), X = (U_x·b_w + b_x − c_x)·Z / f_x, Y likewise. Pinhole, no distortion term (as in SPACE-HOP).
- Total loss: **L = L_tree + L_trans** (unweighted sum, as SPACE-HOP sums its translation and coarse terms).

## Architecture
- **Translation block per branch:** reads the tree's encoded visual tokens (`HierTransformer.encode` output, same tokens the tree decoder attends to). One learned query token cross-attends to them (1 layer), then MLP d → d → 3 predicts u.
- **Fusion:** DINOv3-only uses its block. PoE variant: the mean of the two branches' u (rotation stays per-level PoE).
- **In-plane roll augmentation (EXP-109 recipe):** the rolled crop keeps its centre and side, the camera rotates by Rot(d, −θ) about the undistorted crop ray d, so t′ = Rot(d, −θ)·t ("pre−", the convention verified in EXP-109) and u′ is recomputed from t′ and the same box. Photometric augmentation leaves t unchanged.

## Arms (SPEED+, EXP-109 A recipe otherwise: same views, budget, optimizer; both branches; seeds 0–2)
- **J, joint (primary):** tree + translation block trained together from scratch with L = L_tree + L_trans.
- **F, frozen tree:** the translation block trained on the frozen EXP-109 arm-A encoder (rotation unchanged by construction). Isolates whether joint training is needed.
- **B, box-only baseline (no image):** U_x = U_y = 0 (object centre at the box centre), U_z = the training-set mean of U_z, mapped through the same inverse. Measures how much translation the ground-truth crop alone gives.

## Evaluation
- Domains: synthetic val (selection, DEC-006), lightbox and sunlamp (test only).
- Inference: greedy and orbit. Translation orbit: each view's t back-rolled to view 0 (t₀ = Rot(d, θ)·t_k), then averaged over the 4 views.
- Metrics:
  - official ESA / SPACE-HOP: **E_T = ‖T_pred − T_gt‖₂ (m)**, E_R = 2·arccos|⟨q̂, q⟩| (°); mean and median;
  - descriptive: normalized e_t = E_T / ‖T_gt‖, and the ESA pose score e_q (rad) + e_t, with the challenge precision floors (e_t < 2.173 mm/m and e_q < 0.169° set to 0).

## Criteria
1. **The block adds information beyond the box:** J (and F) synthetic-val mean E_T is below B by more than the larger seed std (PoE and DINOv3-only).
2. **Joint training does not hurt rotation:** J's synthetic-val greedy and orbit rotation means are within the larger seed std of EXP-109 arm A, or better.
3. Real domains: descriptive (E_T, E_R, ESA score).

## Claim boundaries
- **GT crops:** evaluation uses ground-truth crops, as all our rotation results do, and the box encodes most of the translation (centre ≈ projected centre; size ∝ 1/Z). Arm B makes this explicit. A detector-based 6-DoF claim needs a detection stage (or at least box jitter); not part of this experiment.
- **External numbers are unmatched:**
  - SPACE-HOP: E_T 0.096 m synth, 0.302 m lightbox, 0.282 m sunlamp;
  - SPNv2 and other references.
  The box source and crop differ, so they are listed as context only.

## Cost
No new features. 2 trained arms (J, F) × 2 branches × 3 seeds = 12 trainings (J about 20–25 min each, F shorter), plus evaluation: about 5–7 GPU-hours. Code: `scripts/train_trans.py`, `scripts/exp118_eval.py`, `src/tfpose/trans.py`.

## Runs
- Trainings, 2026-10-20 00:20–03:15, `outputs/experiments/EXP-118_translation_6dof/`: J and F × DINOv3 / MoGe-2 × seeds 0–2 (12 runs, `RUN-20261020-002053` … `031037`).
- `RUN-20261019-174116` (J DINOv3 seed 0): stopped after about 5 min (queue reorder for the EXP-064 YCB-V re-run); not used.
- Evaluation: `RUN-20261020-031510` (`scripts/exp118_eval.py`; `tables/metrics.md`, `metrics/per_seed.csv`).

## Results (3-seed means, SPEED+ GT crops; E_T in m, E_R in degrees; seed std of E_T ≤ 0.005 m)
Translation with orbit averaging over 4 views:
| arm | variant | rotation inference | E_T synth | E_T lightbox | E_T sunlamp | E_R synth | E_R lightbox | E_R sunlamp | ESA score synth / LB / SL |
|---|---|---|---|---|---|---|---|---|---|
| B box only | – | – | 0.561 | 0.547 | 0.553 | – | – | – | – |
| F frozen tree | DINOv3 | orbit | 0.068 | 0.205 | 0.205 | 4.21 | 17.40 | 15.27 | 0.086 / 0.339 / 0.301 |
| F frozen tree | PoE | orbit | 0.089 | 0.184 | 0.206 | 3.87 | 12.30 | 12.25 | 0.084 / 0.247 / 0.250 |
| **J joint** | DINOv3 | orbit | **0.059** | 0.185 | **0.183** | 4.19 | 17.31 | 14.87 | 0.084 / 0.333 / 0.290 |
| **J joint** | PoE | orbit | 0.093 | **0.175** | 0.194 | **3.86** | **12.11** | **11.73** | 0.084 / **0.241** / **0.238** |

Greedy (view-0) translation is 10–25% worse than the orbit average: J DINOv3 0.078 / 0.209 / 0.212; J PoE 0.111 / 0.196 / 0.221. Per branch, the MoGe-2 block alone is much worse (J, view 0: 0.18–0.21 m synth), so the PoE translation (the mean of the two branches' u) is worse than DINOv3-only on synthetic.

## Criteria
1. **The block adds information beyond the box: PASS.** J and F beat the box-only baseline by far on synthetic val, e.g. J DINOv3 0.078 (greedy) / 0.059 (orbit) vs 0.561, about 7–9×, against seed std ≤ 0.004.
2. **Joint training does not hurt rotation: PASS.** J vs EXP-109 arm A (= F rotation) on synthetic val: PoE greedy 4.78 vs 4.85, orbit 3.86 vs 3.87; DINOv3 greedy 4.78 vs 4.84. Within or better than the seed std.
3. **Real domains (descriptive):** E_T 0.18–0.19 m (lightbox and sunlamp, J, orbit). Joint training also slightly improves real rotation (PoE + orbit 12.11 / 11.73 vs 12.30 / 12.25; about one seed std).

## Decision
- The architecture extends to **full 6-DoF** with a SPACE-HOP-formulated translation block on the tree's tokens, trained jointly at no rotation cost.
- **Recommended 6-DoF configuration:** J; rotation PoE + orbit; translation DINOv3 block + orbit averaging. This reports E_T 0.059 / 0.185 / 0.183 m and E_R 3.86 / 12.11 / 11.73°. Mixing branches is a reporting choice. The pre-registered PoE translation (mean of both u) is 0.093 / 0.175 / 0.194.
- **External context (unmatched: our ground-truth crops; SPACE-HOP's box source not stated):** SPACE-HOP E_T 0.096 / 0.302 / 0.282 m, E_R 3.995 / 9.015 / 26.646°; SPNv2 0.161 / 0.150 m on lightbox / sunlamp.
- **Caveat:** the box itself carries most translation information (box only 0.56 m). A detector-based claim needs a detection stage.

