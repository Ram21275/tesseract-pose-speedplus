# EXP-026: Training-free reliability signals for posterior fusion, tested one at a time

## Status
Running. Pre-registered on 2026-10-09, before any result. This is the plan's Phase-2 EXP-021, renumbered under DEC-002.

## Research question
Does weighting each branch by a fixed, training-free reliability signal improve on the equal-weight PoE of EXP-025?

## Component under test
Fusion (reliability weighting)

## Track
Training-free.

## Pre-registered protocol
- **Branches, checkpoints, seeds and standardization:** as EXP-025 (A = DINOv3-L `grid4`, B = consensus `normals16~and`, seeds 0/1/2 paired). Greedy decoding at L5.
- **Signals, each tested on its own** (`src/tfpose/fusion.py`):
  1. **Entropy:** per image and level, w_i ∝ 1/(H_i + 1e-3), with Σw = 2 (equal weights = PoE).
  2. **Margin:** w_i ∝ (p_top1 − p_top2) + 1e-3, with Σw = 2.
  3. **Agreement gate:** per image, if the greedy single-branch predictions of A and B lie within 20° → PoE, otherwise A alone. The 20° threshold is fixed a priori and matches EXP-012/014/020.
- **Plan signals not run, with reasons:**
  - VGGT confidence: VGGT is retired (DEC-001).
  - Spectral-energy reliability: no signal on synthetic val in EXP-020.
  - Query/template spectral coherence: there are no templates in this pipeline.
- **Claim:** a signal "helps" if its synthetic-val mean error is below PoE's by more than PoE's seed std (0.46°). Lightbox and sunlamp are descriptive.
- **EXP-027:** if two or more signals help, test their combination. Otherwise equal-weight PoE stays the fusion of record.

## Results
(to be filled)
