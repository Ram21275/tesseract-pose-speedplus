# EXP-017: Spectral-graph foreground from fused DINOv3 + MoGe-2 affinities (CASS option 1)

## Status
Running. Pre-registered on 2026-10-08, before any result. User-requested; option 1 of 4 inspired by CASS (Kim et al., CVPR 2025, arXiv 2411.17150).

## Research question
EXP-015 used a simple principal-component DINO foreground. Does a spectral partition of a **fused** patch graph give cleaner training-free spacecraft masks? The fused graph keeps an edge only where DINOv3 appearance **and** MoGe-2 geometry agree.

## Track
Training-free. Fixed constants chosen a priori; no labels used except for evaluation.

## Pre-registered protocol
- **Grid:** 16×16 patches per GT crop (256 nodes).
- **A_dino:** cosine similarity of DINOv3-L last-layer patch tokens, clipped at ≥ 0.
- **A_geo:** built from MoGe-2's 74×74 maps, area-pooled to 16×16. The normals are renormalized and depth is taken as log-depth.
  - A_geo = exp(−(1 − nᵢ·nⱼ)/0.2) · exp(−|log dᵢ − log dⱼ|/0.1).
  - The MoGe-2 mask is **not** used.
- **Graphs:**
  - **primary:** fused product A = A_dino ⊙ A_geo;
  - descriptive: A_dino alone, and the average ½(A_dino + A_geo).
- **Partition:**
  - Take the symmetric normalized Laplacian and its Fiedler vector (second-smallest eigenvector).
  - Sign: the border ring is background, as in EXP-015.
  - Bilinear upsample to 74×74, then a per-image Otsu threshold.
- **Evaluation:** EXP-015 amendment A1 metrics.
  - Primary: leakage outside the 11-keypoint envelope. Retention: recall against `hull8`. Secondary: IoU and boundary-F.
  - Per domain, with paired bootstrap 95% CIs against MoGe-2, the EXP-015 PCA-DINO mask and the EXP-015 AND mask.
- **Claim:** "Spectral fusion improves on the EXP-015 consensus" if both hold on lightbox **and** sunlamp:
  - primary leakage < AND leakage (CI excludes 0);
  - recall ≥ AND recall − 0.05.
- **Guard:** on synthetic val, recall ≥ MoGe-2 recall − 0.05.

## Results
(to be filled)
