# EXP-017: Spectral-graph foreground from fused DINOv3 + MoGe-2 affinities (CASS option 1)

## Status
Completed. Pre-registered on 2026-10-08, before any result. User-requested; option 1 of 4 inspired by CASS (Kim et al., CVPR 2025, arXiv 2411.17150).

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

## Runs
| Run ID | Status | Notes |
|---|---|---|
| RUN-20261008-234916-seed0 | Completed | 129 s, CPU only, `guarded.sh 16`. Wrote fused-graph eigenvectors 2–5 to `outputs/shared_cache/spectral_parts__subset_v1/` (EXP-019). |

## Results
Source: `RUN-20261008-234916-seed0/metrics/{metrics,paired_differences}.csv`. Baselines are the identical EXP-015 masks.

| domain | method | **leakage** ↓ | **recall** ↑ | IoU* | boundary-F* | tip coverage |
|---|---|---:|---:|---:|---:|---:|
| synthetic_val | AND (EXP-015) | **0.070** | 0.952 | **0.775** | **0.775** | 0.669 |
| synthetic_val | spec_dino | 0.078 | 0.972 | 0.663 | 0.530 | 0.651 |
| synthetic_val | **spec_fused (primary)** | 0.149 | 0.784 | 0.572 | 0.594 | 0.337 |
| synthetic_val | spec_avg | 0.164 | 0.929 | 0.638 | 0.630 | 0.464 |
| lightbox | AND (EXP-015) | 0.119 | 0.977 | 0.599 | 0.393 | 0.771 |
| lightbox | spec_dino | **0.108** | 0.982 | 0.595 | 0.359 | 0.751 |
| lightbox | **spec_fused (primary)** | 0.217 | 0.875 | 0.591 | 0.648 | 0.380 |
| lightbox | spec_avg | 0.220 | 0.940 | 0.609 | 0.672 | 0.457 |
| sunlamp | AND (EXP-015) | 0.116 | 0.994 | 0.593 | 0.329 | 0.632 |
| sunlamp | spec_dino | **0.112** | 0.995 | 0.590 | 0.304 | 0.645 |
| sunlamp | **spec_fused (primary)** | 0.276 | 0.893 | 0.548 | 0.617 | 0.391 |
| sunlamp | spec_avg | 0.272 | 0.945 | 0.558 | 0.613 | 0.442 |

\* Against approximate `hull8` (EXP-015 amendment A1).

Paired differences, primary versus AND (95% CI):
- leakage: +0.098 [+0.083, +0.112] on lightbox and +0.160 [+0.146, +0.174] on sunlamp;
- recall: −0.102 on lightbox and −0.101 on sunlamp.

## Pre-registered verdict
**FAIL.** The fused spectral foreground is worse than the EXP-015 AND consensus: higher leakage and lower recall on both real domains. It also fails the synthetic-val recall guard (0.784 vs MoGe-2's 0.989).

## Interpretation
- **The geometric affinity is a part cue, not a figure/ground cue.** It links patches whose surface normals and depths agree, so the spectral split separates spacecraft **faces** from each other instead of spacecraft from background. Recall falls and boundaries get sharper (boundary-F 0.62–0.67 vs 0.33–0.39). That is the behaviour wanted for part pooling (EXP-019), not for foreground.
- **Descriptive (not pre-registered as primary):** a spectral split of the DINOv3 graph alone beats the EXP-015 PCA-DINO mask on every domain. Leakage changes by −0.037 (synthetic), −0.031 (lightbox) and −0.009 (sunlamp), all with CIs excluding 0, at equal recall. A CASS-style spectral DINO foreground is therefore a slightly better DINO input to the AND consensus. It would be a new, separately registered variant.

## Decision
Reject fused-graph spectral foreground. The EXP-015 AND consensus stays the foreground of record. Carry the fused-graph eigenvectors to EXP-019 as part masks, as pre-registered.
