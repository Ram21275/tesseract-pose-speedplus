# EXP-112: Locked recipe (DEC-018) — PoE vs DINOv3-only, with and without orbit voting: accuracy, end-to-end timing, memory

## Status
**Pre-registered 2026-10-16, before any run** (DEC-018). Evaluation only: no training. It reuses the EXP-109 arm-A trees (seeds 0–2) and the already-extracted orbit-view features. Queued after EXP-111.

## Systems (all from the EXP-109 arm-A trees)
| system | branches | inference |
|---|---|---|
| A-PoE greedy | DINOv3 + MoGe-2 | per-level PoE, greedy (= EXP-109 "A PoE greedy") |
| **A-PoE + orbit** | DINOv3 + MoGe-2 | orbit PoE voting, 4 views (= EXP-109 "C on A") |
| A-DINO greedy | DINOv3 | single tree, greedy |
| **A-DINO + orbit** | DINOv3 | orbit voting with the DINOv3 tree only (new) |

## Metrics
- Synthetic val, synthetic val-aug, lightbox, sunlamp:
  - mean / median / p95, acc@5/10/20, fraction > 90° / > 150°;
  - flip transitions vs EXP-036 PoE greedy and vs the same variant without orbit (denominators).
  - 3 seeds.
- **End-to-end timing** on real lightbox crops (image already decoded), batch 1 and batch 64, median of 5 after warm-up, cuda-synchronized:
  - crop warp(s) + backbone(s) + tree encode / beam + orbit scoring, per system;
  - the per-component breakdown;
  - peak GPU memory;
  - concurrent GPU load recorded (shared GPU, so upper bounds).

## Criteria
Descriptive comparison (the recipe is locked by DEC-018). Reported as is:
- the cost of adding MoGe-2 (Δ ms, Δ°) and the cost of orbit voting, for each variant;
- synthetic val is the reference domain; real domains descriptive.

## Runs / Results / Decision
Not started.
