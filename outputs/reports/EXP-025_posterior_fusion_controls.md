# EXP-025: Phase-2 basic fusion controls: single branches vs training-free posterior fusion

## Status
Running. Pre-registered on 2026-10-09, before any result. This is the plan's Phase-2 EXP-020 ("basic controls"), renumbered under DEC-002.

## Research question
Does a training-free combination of the two branches' Tesseract posteriors beat the strongest single branch?
- Branch A: DINOv3-L `grid4`.
- Branch B: the consensus MoGe-2 `normals16~and`.

## Component under test
Fusion (score level)

## Track
Training-free. No new parameters; the existing probes are reused.

## Pre-registered protocol
- **Models:** the EXP-010 `dinov3_vitl16:grid4` and EXP-016 `moge2_vitl:normals16~and` checkpoints, seeds 0/1/2, paired by seed. Each uses its own train-split feature standardization, recomputed exactly as in training.
- **Fusion of the per-level conditional distributions** (root over 4 charts, then 8 children given the parent), applied inside the beam search:
  - **PoE:** log p = log p_A + log p_B (fixed product of experts, equal weights).
  - **AVG:** log p = log(½ p_A + ½ p_B) (equal posterior averaging, applied per level to the conditionals).
  - **Controls:** A alone, B alone.
- **Decoding:** greedy and beam 4, L5, no GT path.
- **Primary metric:** mean geodesic error per domain.
- **Claim:** fusion "beats the strongest branch" if its synthetic-val mean error is below A-alone by more than A's seed std (0.8°). Lightbox and sunlamp are descriptive.
- **Phase-2 gate:**
  - if neither fusion rule passes, DINOv3-L `grid4` alone is frozen as the representation for Phase 3;
  - the conditional EXP-026–029 (learned gates, cross-attention) are skipped, because training-free fusion failed and complementarity was weak (EXP-014: no geometry-specific complementarity).

## Results
(to be filled)
