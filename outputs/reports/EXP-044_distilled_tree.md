# EXP-044: Distilled probability-consistent tree (Phase 4)

## Status
**Completed 2026-10-13. Verdict (pre-registered rule): distillation does not help; β = 0 selected, seeds 1–2 not run.** Score **2/5 Weak**: the distillation took (criterion 2 met) and each branch improved, but the fused PoE synthetic-val mean did not.

Pre-registered 2026-10-13, before any result. Training on synthetic/train only; selection on synthetic/validation only; lightbox and sunlamp descriptive (DEC-006).

## Research question
Does training each branch's hierarchical tree to also match the flow-induced Tesseract masses of the EXP-042 posterior (EXP-043) improve the fused (PoE) greedy prediction, at the tree's inference cost (no ODE at test time)?

## Component under test
Trainable predictor: the EXP-036 Transformer tree, with an extra distillation loss. Teacher: the frozen EXP-042 fused flow.

## Loss
For each training image with GT path loss \(\mathcal L_{GT}\) (EXP-036 soft targets, unchanged):

\[
\mathcal L = \mathcal L_{GT} + \beta\,\frac{1}{L+1}\sum_{l=0}^{L}\operatorname{KL}\big(\operatorname{stopgrad}[\mu_l]\,\Vert\,p_\phi^{(l)}\big)
\]

- \(\mu_l\) = flow-induced mass at level l (0 = root chart … 5 = leaf) from the M = 32 fused EXP-042 endpoint samples of that image (as in EXP-043).
- \(p_\phi^{(l)}(c)\) = the tree's marginal of the level-l cell c = root probability × child conditionals along c's path (teacher-forced). Only cells with \(\mu_l>0\) are needed, so no sum over \(8^l\) cells.
- Implemented as the KL up to the teacher's (constant) entropy: \(-\frac{1}{L+1}\sum_l \log p_\phi^{(l)}(\text{prefix}_l(R_m))\), averaged over **K = 8 of the 32 samples drawn uniformly per image per step** (an unbiased Monte-Carlo estimate of the 32-sample cross-entropy). Code: `scripts/train_probe_p3.py --kd-samples --kd-beta --kd-k`.
- The 1/(L+1) normalization is fixed here, before results; it only rescales β.

## Fixed setup
- **Teacher targets:** the EXP-042 fused flow (DINOv3 + MoGe-2 heads, velocity average, Haar base, NFE 10, M = 32) re-sampled on **synthetic/train only** (`exp04x_fused_eval.py --groups synthetic_train --save-samples --save-all-seeds --feat-cpu --queries 1024`). Teacher seed s → student seed s. Validation and test images are never used to build targets.
- **Students:** both branch trees (DINOv3 `tokens8`, MoGe-2 `normals16~and`) are retrained from scratch with the EXP-036 recipe (Transformer decoder, depth 5, soft targets, 100 epochs, batch 256, bf16 autocast, checkpoint by synthetic-val greedy mean), each with the **same fused** teacher masses. The system is the equal-weight per-level PoE of the two trees, decoded greedily (DEC-003). This distils the fused posterior into each branch, which is not the same as distilling it into the PoE; stated as a design limitation.
- **Memory:** features in host memory (`--feat-device cpu`), every job through `scripts/guarded.sh` in tmux `ram`, one at a time.

## Arms and selection
- **β = 0:** the EXP-036 runs (3 seeds; identical code path, since the distillation term is skipped at β = 0). A fresh β = 0 seed-0 rerun of both branches in this experiment checks the code path; it should match EXP-036 within seed noise.
- **β ∈ {0.1, 0.3, 1.0}:** seed 0, both branches.
- **Selection:** the β with the lowest seed-0 PoE greedy synthetic-val mean error (the β = 0 rerun included). If β = 0 wins, stop: the verdict is "distillation does not help", and no further seeds are run. Otherwise run the selected β for seeds 1 and 2.

## Reported metrics
- PoE greedy: mean / median / p95 error, acc@5/10/20, fraction > 90° and > 150°, top-8 beam leaf recall; per domain; 3-seed mean ± std for β = 0 and the selected β, seed 0 for the whole grid.
- Per level (as EXP-043): tree mass on the GT cell, greedy top-1, and the distillation fit on synthetic val = cross-entropy of the (val) flow samples under the PoE tree's level marginals, β = 0 vs selected β (uses the EXP-043 val samples; evaluation only).
- **Teacher diagnostic:** flow mass on the GT cell per level, synthetic train vs synthetic val. The flow heads were fit on train (train/val gap seen in EXP-040), so near-one-hot train masses would make the distillation term close to a copy of the GT loss. Reported to read a null result.
- Latency per image (tree inference only) vs the EXP-042 flow (about 17 ms).

## Pre-registered criteria
1. **Helps (primary):** the selected β's 3-seed PoE greedy synthetic-val mean is below the β = 0 (EXP-036) mean by more than the larger seed std.
2. **Distillation took (secondary):** the distillation fit on synthetic val improves vs β = 0.
3. Lightbox / sunlamp: descriptive only; reported regardless of direction (EXP-043 found the tree better than the flow at every level there, so distillation may hurt real-domain routing).

## Runs
All in `outputs/experiments/EXP-044_distilled_tree/`, chain `scripts/run_exp044.sh` (log `outputs/experiments/exp044_chain.log`), seed 0.
- Teacher samples (synthetic train, 47,966 images × 32, three teacher seeds): `RUN-20261013-110159-seed0`. Fused flow point estimate on train: KDE mode 3.85°, leaf 3.55°.
- Teacher diagnostic: `RUN-20261013-114107-seed0`.
- Students, (DINOv3, MoGe-2): β = 0 `114528`, `120330`; β = 0.1 `122159`, `125739`; β = 0.3 `140950`, `144526`; β = 1.0 `155905`, `163806`.
- Selection eval (PoE greedy, per level, distillation fit): `RUN-20261013-175631-seed0` (`scripts/exp044_eval.py --stage select`).

## Results

### Teacher diagnostic (synthetic train)
| level | flow mass on GT | tree mass on GT | flow top-1 | tree greedy top-1 |
|---|---|---|---|---|
| 0 | 0.910 | 0.955 | 0.970 | 0.981 |
| 2 | 0.693 | 0.669 | 0.847 | 0.808 |
| 3 | 0.549 | 0.535 | 0.730 | 0.704 |
| 5 | 0.197 | 0.351 | 0.323 | 0.547 |

The train masses are not one-hot, so the distillation term is not a copy of the GT loss. The flow routes better than the tree at L2–3 and worse at the leaf.

### Single branches (best checkpoint, seed 0, mean error °)
| β | DINOv3 synth / lightbox / sunlamp | MoGe-2 synth / lightbox / sunlamp |
|---|---|---|
| 0 | 5.45 / 39.12 / 42.75 | 16.81 / 43.41 / 60.28 |
| 0.1 | 5.37 / 38.80 / 41.32 | 16.43 / 42.25 / 61.09 |
| 0.3 | 5.21 / 37.22 / 42.61 | 15.65 / 41.66 / 59.70 |
| 1.0 | **5.07** / 38.15 / **40.54** | **14.48** / **39.86** / 58.13 |

The β = 0 rerun matches EXP-036 (DINOv3 5.45, MoGe-2 16.81 identical), so the code path is checked. Each branch alone improves steadily with β on synthetic val (DINOv3 −0.38°, MoGe-2 −2.33° at β = 1).

### Fused PoE greedy (seed 0; selection metric = synthetic-val mean)
| arm | synth mean / median / p95 | synth acc@5 / acc@10 | synth >90° | top-8 leaf recall (synth) | lightbox mean / median / acc@10 | sunlamp mean / median / acc@10 |
|---|---|---|---|---|---|---|
| EXP-036 seed 0 (re-evaluated) | 5.567 / 4.66 / 10.25 | 0.558 / 0.945 | 0.0020 | 0.693 | 27.62 / 9.36 / 0.537 | 37.53 / 13.37 / 0.359 |
| **β = 0 (rerun, selected)** | **5.589** / 4.66 / 10.31 | 0.558 / 0.944 | 0.0021 | 0.694 | 27.63 / 9.32 / 0.537 | 37.75 / 13.32 / 0.363 |
| β = 0.1 | 5.688 / 4.59 / 10.28 | 0.572 / 0.945 | 0.0035 | 0.711 | 27.18 / 8.79 / 0.569 | 38.15 / 12.93 / 0.384 |
| β = 0.3 | 5.672 / 4.44 / 9.80 | 0.600 / 0.953 | 0.0044 | 0.731 | 26.58 / 8.55 / 0.583 | 38.16 / 12.70 / 0.396 |
| β = 1.0 | 5.622 / **4.33 / 9.60** | **0.620 / 0.955** | 0.0049 | **0.768** | **26.17 / 8.19 / 0.596** | 37.67 / **12.01 / 0.421** |

Tree latency is unchanged (about 0.3–0.4 ms per image at batch 1024, vs about 15–17 ms for the flow).

### Distillation fit on synthetic val (cross-entropy of the EXP-043 val flow samples under the PoE level marginals, mean over levels; lower = closer to the flow)
β = 0: 4.375 · β = 0.1: 3.613 · β = 0.3: 3.231 · β = 1.0: **2.863** (leaf level 11.19 → 6.63). PoE tree mass on the GT cell also rises at every level (leaf 0.158 → 0.194; L2 0.651 → 0.728).

## Criteria
1. **Helps (primary): FAIL.** β = 0 has the lowest seed-0 PoE synthetic-val mean (5.589°; β = 1.0 is 5.622°, +0.03°). By the pre-registered rule the experiment stops at seed 0. All four arms lie within 0.10°, which is less than 2 seed stds of EXP-036 (± 0.063°), so the grid does not separate the arms on this metric in either direction.
2. **Distillation took (secondary): PASS.** The val fit improves steadily with β (4.38 → 2.86).
3. **Real domains (descriptive):** lightbox mean, median and acc@10 improve steadily with β (27.63 → 26.17°, median 9.32 → 8.19°, acc@10 0.537 → 0.596). Sunlamp median and acc@10 improve (13.32 → 12.01°, 0.363 → 0.421), but its mean is flat (37.75 → 37.67°, worse at β = 0.1 / 0.3).

## Interpretation (descriptive)
- The distilled trees are better almost everywhere *except* the selection metric. Synthetic-val median, p95, acc@5/10, top-8 recall and per-level GT mass all improve with β. The mean does not, because the fraction of > 90° errors grows (0.21% → 0.49%, 25 → 59 of 11,994 images). That is consistent with the teacher posterior putting mass on the 180° flip mode (EXP-042/104): copying it makes the trees spread mass onto the flip, and for a few images the PoE then picks it. Mean error on a set with almost no flips is dominated by those few flips.
- At seed 0, β = 1.0 is the best arm by every metric except the synthetic-val mean and > 90° fraction. It is the natural student to pair with a flip verifier (EXP-105): its top-8 recall (0.768 vs 0.694 synth, 0.369 vs 0.283 lightbox) gives the verifier more correct candidates. This is noted for EXP-105's planning only. It does not change this experiment's verdict, and it is not a post-hoc selection.
- One seed per β: the arm-to-arm differences are not tested for significance.

## Decision
Not adopted as the Phase-4 predictor: EXP-036 stays frozen. Distillation recipe and β = 1.0 seed-0 checkpoints kept (`RUN-20261013-155905`, `RUN-20261013-163806`) as an optional higher-recall candidate generator for EXP-105, which would be a separate pre-registered arm.
