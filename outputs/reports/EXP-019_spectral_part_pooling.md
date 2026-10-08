# EXP-019: Spectral part pooling of DINOv3 features for pose (CASS option 3)

## Status
Completed. Pre-registered on 2026-10-08, before any result. User-requested; option 3 of 4.


## Keep score
**1 / 5: Negative.** Spectral part pooling is far worse than the size-matched 2×2 grid on every domain.

_Keep score rubric (0–5): 5 strong support: pre-registered goal met with a large effect, adopt. 4 support: goal met, carry forward. 3 partial: mixed or one-sided evidence, keep for further testing. 2 weak: goal not met but an informative side result. 1 negative: fails, reject, kept only as a recorded result. 0 inconclusive or not run._

## Research question
Rotation depends on how the spacecraft's parts are arranged. Does pooling DINOv3-L tokens into **spectral parts** describe that arrangement better than a fixed grid, especially on real images? The parts are soft regions from eigenvectors of the EXP-017 fused graph.

## Track
The feature construction is training-free. The predictor is the common Phase-1 MLP.

## Pre-registered protocol
- **Graph:** the EXP-017 primary fused graph (A_dino ⊙ A_geo, 16×16).
- **Parts:** eigenvectors 2–5 of the symmetric normalized Laplacian (k = 4). Soft weights w_k = v_k² / Σ v_k², which is sign-invariant.
- **Feature, `parts4`:** for each part, the weighted mean DINOv3-L token (1024), the weighted centroid (x, y) and the weighted second moments (xx, xy, yy). Total 4 × 1029 = 4,116 dims.
- **Matched control:** `grid2` (2×2 average pooling, 4 × 1024 = 4,096 dims). `grid4` (EXP-010) is also reported.
- **Predictor, seeds, selection:** as in EXP-010 (3 seeds, synthetic-val selection).
- **Claim:** `parts4` helps if lightbox **and** sunlamp mean error both beat `grid2` by more than the larger seed std, with synthetic val no worse than `grid2` by more than 1 std.

## Runs
6 runs (`parts4`, `grid2` × 3 seeds), all completed under `guarded.sh 16`. See `outputs/experiments/EXP-019_spectral_part_pooling/run_list.csv`.

## Results
Mean ± std over 3 seeds, greedy decoding. Source: `summary_by_features.csv`; `grid4` from EXP-010.

| features | dim | synthetic_val mean / median | lightbox mean / median | sunlamp mean / median | synth / lightbox / sunlamp acc@20 |
|---|---:|---|---|---|---|
| `parts4` (spectral parts) | 4,116 | 53.0 ± 0.1 / 31.8 | 96.9 ± 1.4 / 94.9 | 104.2 ± 1.0 / 106.2 | 0.299 / 0.039 / 0.019 |
| `grid2` (matched control) | 4,096 | **28.5 ± 1.0 / 16.2** | **79.9 ± 1.4 / 68.7** | **85.8 ± 3.7 / 81.4** | 0.612 / 0.140 / 0.099 |
| `grid4` (EXP-010, reference) | 16,384 | 27.7 ± 0.8 / 17.2 | 76.1 ± 1.0 / 62.2 | 79.1 ± 0.6 / 70.3 | 0.585 / 0.145 / 0.109 |

## Pre-registered verdict
**FAIL.** `parts4` is worse than `grid2` by 24.5° (synthetic), 17.0° (lightbox) and 18.4° (sunlamp) mean error.

## Interpretation
- **Laplacian eigenvectors aren't consistent across images.** Their order changes when eigenvalues are close, and the parts they pick out depend on the view. So "part 1" in one image isn't the same physical part in the next, and the probe can't learn a stable mapping. A fixed spatial grid has a consistent layout and wins easily.
- **Side result:** `grid2` is as good as `grid4` on synthetic val (28.5° vs 27.7°) but worse on real data, so finer spatial layout helps transfer.

## Decision
Reject spectral part pooling.
