# EXP-044: Distilled probability-consistent tree (Phase 4)

## Status
**Pre-registered 2026-10-13, before any result.** Training on synthetic/train only; selection on synthetic/validation only; lightbox and sunlamp descriptive (DEC-006).

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

## Runs / Results / Decision
Not started.
