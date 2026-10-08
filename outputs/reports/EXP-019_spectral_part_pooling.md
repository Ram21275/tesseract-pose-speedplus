# EXP-019: Spectral part pooling of DINOv3 features for pose (CASS option 3)

## Status
Running. Pre-registered on 2026-10-08, before any result. User-requested; option 3 of 4.

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

## Results
(to be filled)
