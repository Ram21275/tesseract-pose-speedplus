# EXP-016: Does the DINOv3 ∧ MoGe-2 consensus mask help pose, in both directions?

## Status
Completed. Pre-registered on 2026-10-08, before any result.


## Keep score
**4 / 5: Support.** One-way at the pose level, but large: consensus-masked MoGe-2 normals gain 6.9° (lightbox) and 13.0° (sunlamp) mean error and have the best real-domain acc@20 of any feature. Strong candidate for Phase-2 fusion.

_Keep score rubric (0–5): 5 strong support: pre-registered goal met with a large effect, adopt. 4 support: goal met, carry forward. 3 partial: mixed or one-sided evidence, keep for further testing. 2 weak: goal not met but an informative side result. 1 negative: fails, reject, kept only as a recorded result. 0 inconclusive or not run._

## Research question
Does the training-free consensus mask from EXP-015 improve rotation prediction for **both** branches?
- **DINO helps MoGe-2:** MoGe-2 `normals16` built with the consensus mask instead of MoGe-2's own mask.
- **MoGe-2 helps DINO:** DINOv3-L `grid4` built with mask-weighted 4×4 pooling using the consensus mask, instead of plain average pooling.

User-requested; outside the planned EXP list.

## Component under test
Fusion (training-free mask fusion) feeding the trainable predictor

## Track
The mask fusion is training-free. The predictor is the trainable Phase-1 MLP, unchanged.

## Pre-registered protocol
- **Mask:** the EXP-015 AND mask (74×74). No other variant.
- **MoGe-2 arm:** `normals16` recomputed with the consensus mask (normals × mask, plus the mask), pooled to 16×16. Same dimension (1024) as `normals16`.
- **DINO arm:** `grid4` with each 4×4 cell's 4×4 tokens averaged with weights from the consensus mask area-pooled to 16×16.
  - If a cell's weight sum is < 1e-6, use the unweighted cell mean.
  - Same dimension (16,384) as `grid4`.
- **Predictor and budget:** identical to EXP-010/013: 3 seeds, 100 epochs, selection on synthetic-val mean error.
- **Controls:** the existing unmasked runs (EXP-013 `normals16`, EXP-010 `vitl16:grid4`), same seeds.
- **Primary metric:** mean geodesic error per domain, greedy decoding.
- **A direction "helps" if both hold:**
  - lightbox **and** sunlamp mean error both drop by more than the larger seed std of the two arms;
  - synthetic val does not get worse by more than 1 seed std (guard).
- **"Two-way at pose level"** only if both arms help.

## Runs
6 runs, all completed: `outputs/experiments/EXP-016_consensus_pose/run_list.csv`. Each ran alone under `guarded.sh 16`. The controls are the same-seed runs from EXP-013 (`normals16`) and EXP-010 (`vitl16:grid4`).

## Results
Greedy decoding, mean ± std over 3 seeds. Sources: `EXP-016_consensus_pose/summary_by_features.csv`, `EXP-013_moge2_controls/summary_by_features.csv`, `EXP-010_dinov3_baseline/summary_by_features.csv`.

| features | synthetic_val mean | lightbox mean | sunlamp mean | synth / lightbox / sunlamp median | synth / lightbox / sunlamp acc@20 |
|---|---|---|---|---|---|
| MoGe-2 `normals16` (own mask) | 56.4 ± 0.8 | 84.0 ± 2.1 | 100.1 ± 0.2 | 24.8 / 84.6 / 105.0 | 0.442 / 0.202 / 0.092 |
| **MoGe-2 `normals16~and`** (consensus mask) | **47.4 ± 1.0** | **77.1 ± 1.1** | **87.1 ± 1.4** | 18.9 / 68.7 / 88.3 | 0.521 / **0.241** / **0.160** |
| DINOv3-L `grid4` (plain pooling) | **27.7 ± 0.8** | **76.1 ± 1.0** | 79.1 ± 0.6 | 17.2 / 62.2 / 70.3 | 0.585 / 0.145 / 0.109 |
| DINOv3-L `grid4~and` (mask-weighted pooling) | 31.1 ± 0.4 | 77.0 ± 0.6 | 78.2 ± 0.8 | 19.6 / 64.2 / 68.0 | 0.514 / 0.127 / 0.098 |

## Pre-registered verdict
- **DINO helps MoGe-2 (pose): YES.**
  - Mean error drops by 6.9° on lightbox and 13.0° on sunlamp. Both exceed the larger seed std (2.1° / 1.4°).
  - Synthetic val also improves, by 9.0°, so the guard passes.
- **MoGe-2 helps DINO (pose): NO.**
  - Lightbox gets slightly worse (+0.9°). Sunlamp improves by 0.9°, which is only marginally above the 0.8° std.
  - Synthetic val gets worse by 3.4°, so the guard fails.
- **Two-way at the pose level: NO. The pose-level consensus is one-way, DINO → MoGe-2.**

## Interpretation
- **Fixing MoGe-2's foreground with DINO is the single most useful training-free fusion found so far.**
  - The consensus-masked normals reach the best real-domain acc@20 of any feature tested: 0.241 lightbox and 0.160 sunlamp, against DINOv3 grid4's 0.145 and 0.109. This is descriptive only, since these are test domains.
  - Their mean error still trails DINOv3 slightly on lightbox and clearly on sunlamp.
  - The pattern is many more near-correct predictions together with more large failures.
- **DINOv3's pooled tokens don't benefit from masking.** DINOv3's patch tokens already carry context; zeroing background-weighted cells removes information the probe was using, such as silhouette edges and the background contrast near the boundary.
- **The mask-level two-way effect (EXP-015) doesn't carry over to DINO's pose features.**

## Decision
Keep `normals16~and` as a strong geometric-branch candidate for Phase-2 fusion with DINOv3. Reject mask-weighted DINO pooling.
