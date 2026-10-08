# EXP-101: Tesseract vs SPACE-HOP Hopf discretization: grid quality, prediction and timing in the frozen-backbone pipeline

## Status
Running. Pre-registered on 2026-10-09, before any result. User-requested side experiment (DEC-002 numbering).

## Research question
Inside **our** pipeline (frozen DINOv3-L features, same probe trunk, same data), how does the Tesseract discretization with hierarchical prediction compare with SPACE-HOP's Hopf anchor grid with flat classification? The comparison covers (A) discretization quality, (B) rotation accuracy and (C) prediction time.

**Scope.** SPACE-HOP's own published results (fine-tuned JEPA ViT-B, full data, continuous offsets) are **not** compared. Only the discretization and decoding are swapped, as the user directed.

## Component under test
Rotation grid, decoder and prediction head

## Track
Trainable predictor (the backbone is frozen and identical across arms)

## Pre-registered protocol
**SPACE-HOP grid.**
- Reproduced from Team-M3OW/SPACE-HOP `src/hopf_grid.py`: Fibonacci S² directions, a frame per direction, then in-plane rolls. It matches the original code to ≤ 1.6e-4 (float32 rounding).
- Paper config: 256 × 12 = **3,072** anchors.
- Count-matched to the Tesseract with 12 rolls: 171 × 12 = 2,052 (≈ L3 2,048), 1,365 × 12 = 16,380 (≈ L4 16,384), 10,923 × 12 = 131,076 (≈ L5 131,072).

**A. Discretization** (no learning; 1M Haar rotations, seed 0, plus all 69,491 SPEED+ labels):
- nearest-anchor error mean / median / p95 / max (the flat-decoder error);
- Tesseract own-cell error (the tree-decoder error) and nearest-centre error;
- nearest-neighbour spacing CV;
- the structure each grid offers: Tesseract greedy nodes 4 + 8L, against K for a flat head.

**B. Prediction** (subset v1, DINOv3-L `grid4`, 3 seeds, 100 epochs, the EXP-010 trunk and optimizer, selection on synthetic-val mean error):
- **Tesseract:** the EXP-010 L5 hierarchical probes, same seeds. Errors are reported for decoding at L3, L4 and L5 from the same model.
- **SPACE-HOP-style flat head:** the same trunk plus one linear layer over K anchors.
  - Cross-entropy on the closest anchor (SPACE-HOP `get_closest_anchor`); prediction = argmax anchor.
  - No continuous offset in either arm, so only discrete prediction is compared.
  - K ∈ {3,072 (paper), 16,380, 131,076}.
- **Metrics:** mean / median geodesic error, acc@10/20, per domain. Lightbox and sunlamp are descriptive.

**C. Timing** (head only; the frozen backbone is identical for both arms and is timed separately for context):
- Tesseract `beam_search_fast` (fully on device; verified identical to the reference decoder), greedy and beam 4, depth L = 2–6.
- The flat head at matched K = 4 · 8^L, L = 2–6 (up to 1,048,576 anchors) where memory allows.
- Batch 1 and batch 256, on: GPU0 (RTX 6000 Ada, **shared with other users, so noisy**), GPU1 (RTX A400, idle) and CPU (1 thread).
- 50 warm-up and 200 timed iterations, CUDA-synchronized. Reported: median, p95, parameter count and head FLOPs per image.
- The DINOv3-L backbone forward is timed with the same protocol.

**Claims (pre-registered):**
- "Tesseract head is faster" at a given K if its median latency is lower on GPU1 **and** CPU at batch 1.
- "Tesseract is as accurate" at matched hypothesis count if its synthetic-val mean error is within the seed std of the flat head (or lower).
- The answer is expected to depend on K, so the crossover K is reported, not a single verdict.

## Protocol amendment B2 (2026-10-09, user request, before any B2 result)
Part B is repeated on the consensus feature that worked in EXP-016: **`moge2_vitl:normals16~and`** (MoGe-2 normals with the EXP-015 DINO∧MoGe-2 consensus mask).
- **Flat SPACE-HOP-style head:** K ∈ {3,072, 16,380, 131,076}, 3 seeds, the same trunk and budget.
- **Tesseract arm:** the existing EXP-016 runs (same feature and seeds), decoded at L3 and L5.
- The protocol and claims are otherwise unchanged. The B2 runs start only after the part-C timing finishes, so GPU contention from training cannot bias the timings.

## Results
(to be filled)
