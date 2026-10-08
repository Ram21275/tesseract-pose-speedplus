# EXP-034: Greedy vs beam decoding at full data (Phase 3)

## Status
Running. Pre-registered on 2026-10-09, before any result.

## Research question
Does beam search (widths 2, 4, 8) improve over greedy decoding at full data, for the single branches and for the PoE fusion?

## Component under test
Decoder

## Track
Inference only (no training)

## Pre-registered protocol
- **Models:** the EXP-030 checkpoints (3 seeds, both branches). PoE as in DEC-003.
- **Decoding:** beams 1, 2, 4, 8. Reported: mean / median error, nodes scored per image, top-k leaf recall.
- **Claim:** beam k helps if its PoE synthetic-val mean is below greedy's by more than the seed std.
- The beam width chosen here is applied to the frozen Phase-3 configuration.

## Results
(to be filled)
