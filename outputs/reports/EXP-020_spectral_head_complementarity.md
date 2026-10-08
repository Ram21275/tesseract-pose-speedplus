# EXP-020: Spectral head-matching as a complementarity measure (CASS option 4)

## Status
Running. Pre-registered on 2026-10-08, before any result. User-requested; option 4 of 4. Descriptive analysis; no claim thresholds.

## Research question
CASS pairs attention heads by the Wasserstein distance between their key-graph eigenvalue spectra. Does that label-free spectral distance between DINOv3-L and MoGe-2 measure their complementarity, and does it predict per image when they disagree or fail?

## Track
Training-free analysis.

## Pre-registered protocol
- **Data:** per-image, per-head top-20 eigenvalue spectra of the last-block key graphs, saved during the EXP-018 extraction pass. DINOv3-L has 16 heads at 16×16; MoGe-2 has 16 heads at 37×37.
- **Per image:** the 16×16 Wasserstein cost matrix (CASS normalization), the Hungarian matching, and the mean matched distance D.
- **Reported per domain:**
  - the distribution of D;
  - the Spearman correlation of D with the DINO–MoGe-2 prediction disagreement, and with DINOv3-L and MoGe-2 errors (EXP-010/013 seed-0 probes);
  - the AUROC of D for "DINOv3-L error > 20°".
- **Selection:** none. Real-domain numbers are descriptive.

## Results
(to be filled)
