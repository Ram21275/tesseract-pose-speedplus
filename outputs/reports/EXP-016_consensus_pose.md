# EXP-016: Does the DINOv3 ∧ MoGe-2 consensus mask help pose, in both directions?

## Status
Running. Pre-registered on 2026-10-08, before any result.

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

## Results
(to be filled)
