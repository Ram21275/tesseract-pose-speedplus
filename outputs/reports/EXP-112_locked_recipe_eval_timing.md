# EXP-112: Locked recipe (DEC-018) — PoE vs DINOv3-only, with and without orbit voting: accuracy, end-to-end timing, memory

## Status
**Completed 2026-10-16.** All four locked-recipe variants evaluated, with end-to-end timing. PoE + orbit 3.87 / 12.30 / 12.28 at 279 ms (batch 1); DINOv3-only + orbit 4.21 / 17.40 / 15.27 at 70 ms; DINOv3-only greedy 4.84 / 22.56 / 22.02 at 15 ms.

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

## Runs
Accuracy `RUN-20261016-225940-seed0` (3 seeds); timing `RUN-20261016-231146-seed0` (lightbox crops, shared GPU).

## Accuracy (3-seed mean ± std, °)
| system | synth val | synth val-aug | lightbox | sunlamp | > 150° LB / SL |
|---|---|---|---|---|---|
| EXP-036 PoE greedy | 5.54 ± 0.07 | 30.84 | 26.91 | 36.66 | 0.059 / 0.074 |
| A-DINO greedy | 4.84 ± 0.05 | 9.88 | 22.56 | 22.02 | 0.060 / 0.032 |
| **A-DINO + orbit** | 4.21 ± 0.04 | 6.25 | 17.40 | 15.27 | 0.048 / 0.020 |
| A-PoE greedy | 4.86 ± 0.03 | 9.75 | 17.35 | 20.16 | 0.042 / 0.033 |
| **A-PoE + orbit** | **3.87 ± 0.03** | **5.82** | **12.30** | **12.28** | 0.030 / 0.017 |

## Timing (ms per image; batch 1 → FPS; batch 64 → img/s)
| system | batch 1 | FPS | batch 64 | img/s | components (batch 1) | peak GPU |
|---|---|---|---|---|---|---|
| A-DINO greedy | 14.7 | 68 | 2.9 | 347 | DINOv3 6.4, tree 7.4 | 2.5 GB |
| A-DINO + orbit | 69.6 | 14 | 12.1 | 82 | DINOv3 26.3, trees 31.9, scoring 8.2 | 2.5 GB |
| A-PoE greedy | 64.0 | 16 | 50.0 | 20 | DINOv3 7.0, MoGe-2 21.0, mask / post 20.3 (CPU), trees 14.1 | 2.9 GB (28.8 GB at 64) |
| A-PoE + orbit | 278.5 | 3.6 | 206.8 | 4.8 | MoGe-2 84.2, mask / post 80.8, DINOv3 28.1, trees 61.2, scoring 15.3 | 2.9 GB |

## Reading
- **MoGe-2 overhead:** about 50 ms per view at batch 1 (network 21 ms plus the EXP-015 consensus mask on the CPU, 20 ms). It costs 4× the time with orbit voting.
- In exchange it gives 0.3° on synthetic, 5.1° on lightbox and 3.0° on sunlamp with orbit voting.
- **Orbit overhead:** about 4.7× (DINOv3-only) and 4.4× (PoE) at batch 1.
- The CPU mask step is a clear optimization target (it could be moved to the GPU).
