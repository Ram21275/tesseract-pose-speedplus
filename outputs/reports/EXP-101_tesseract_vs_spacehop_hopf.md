# EXP-101: Tesseract vs SPACE-HOP Hopf discretization: grid quality, prediction and timing in the frozen-backbone pipeline

## Status
Completed. Pre-registered on 2026-10-09, before any result (amendment B2 was added before its runs). User-requested side experiment (DEC-002 numbering).


## Keep score
**3 / 5: Partial.**
- **For the Tesseract tree:** it scales: it is faster with constant parameters from about 131k hypotheses, and it is far more accurate than flat classification on DINOv3 features.
- **For SPACE-HOP's Hopf:** the grid discretizes better per anchor; at SPACE-HOP's size (3,072) the flat head is faster; and on the consensus feature the flat head ties the tree on mean error while beating it on acc@20.
- **Overall:** keep the tree for fine resolution, and keep a flat Hopf head as a strong baseline.

_Keep score rubric (0–5): 5 strong support: pre-registered goal met with a large effect, adopt. 4 support: goal met, carry forward. 3 partial: mixed or one-sided evidence, keep for further testing. 2 weak: goal not met but an informative side result. 1 negative: fails, reject, kept only as a recorded result. 0 inconclusive or not run._

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

## Runs
| Run ID | Part | Status |
|---|---|---|
| RUN-20261009-002419-seed0 | A discretization | Completed (+ descriptive balanced-Hopf file `metrics/discretization_balanced_descriptive.csv`) |
| RUN-20261009-002520-seed99 | B smoke test (5 epochs) | Completed; excluded from all tables |
| RUN-20261009-002558-seed0 | C smoke test (2 depths, 5 iters) | Completed; excluded |
| 9 runs RUN-20261009-002735 … 003008 | B: flat heads on DINOv3-L `grid4` | Completed |
| RUN-20261009-003047-seed0 | C timing, full sweep | Completed (no OOM; 90 head rows + backbone) |
| 9 runs after 003047 | B2: flat heads on consensus `normals16~and` | Completed |

All flat runs are aggregated in `outputs/experiments/EXP-101_tesseract_vs_spacehop_hopf/flat_runs_all.csv`. Each job ran alone under `scripts/guarded.sh`.

## Results
### A. Discretization (no learning; 1M Haar rotations). Source: `RUN-20261009-002419-seed0/metrics/discretization.csv`
| grid | n | decoded mean ° | p95 ° | max ° | decoder |
|---|---:|---:|---:|---:|---|
| Tesseract L3 | 2,048 | 10.01 | 15.83 | 23.81 | tree, 28 nodes |
| SPACE-HOP Hopf 171×12 | 2,052 | **9.67** | **14.60** | **18.12** | flat over 2,052 |
| SPACE-HOP Hopf 256×12 (paper) | 3,072 | 8.79 | 14.00 | 16.94 | flat over 3,072 |
| Tesseract L4 | 16,384 | **5.01** | **7.92** | **11.83** | tree, 36 nodes |
| SPACE-HOP Hopf 1365×12 | 16,380 | 6.76 | 13.09 | 15.40 | flat |
| Tesseract L5 | 131,072 | **2.51** | **3.97** | 5.90 | tree, 44 nodes |
| SPACE-HOP Hopf 10923×12 | 131,076 | 6.06 | 12.91 | 15.05 | flat |
| *Balanced Hopf 108×19 (descriptive, not pre-registered)* | 2,052 | *9.22* | *12.96* | *16.40* | flat |
| *Balanced Hopf 443×37 (descriptive)* | 16,391 | *4.61* | *6.47* | *8.12* | flat |
| *Balanced Hopf 1771×74 (descriptive)* | 131,054 | *2.30* | *3.22* | *3.91* | flat |

### B and B2. Prediction with identical frozen features and trunk
Mean rotation error in degrees ± std over 3 seeds, with median and acc@20 in brackets. Tesseract numbers are the EXP-010 / EXP-016 runs (same seeds), decoded at L3 or L5 from the same model. Lightbox and sunlamp are descriptive.

| features | domain | Tesseract L3 (2,048) | Flat Hopf 3,072 (SPACE-HOP config) | Tesseract L5 (131,072) | Flat Hopf 16,380 | Flat Hopf 131,076 |
|---|---|---|---|---|---|---|
| DINOv3-L `grid4` | synthetic_val | 29.2 ± 0.8 | 50.7 ± 2.0 (med 22.1, a20 0.446) | 27.7 ± 0.8 (med 17.2, a20 0.585) | 65.8 ± 1.4 (med 34.2, a20 0.317) | 69.9 ± 0.3 (med 39.5, a20 0.294) |
| DINOv3-L `grid4` | lightbox | 77.1 ± 0.7 | 93.2 ± 2.5 (med 96.1, a20 0.134) | 76.1 ± 1.0 (med 62.2, a20 0.145) | 106.7 ± 2.5 (med 112.2, a20 0.055) | 109.9 ± 1.4 (med 117.3, a20 0.050) |
| DINOv3-L `grid4` | sunlamp | 80.1 ± 0.4 | 100.6 ± 1.1 (med 105.2, a20 0.090) | 79.1 ± 0.6 (med 70.3, a20 0.109) | 109.2 ± 0.9 (med 113.2, a20 0.036) | 111.7 ± 0.8 (med 115.9, a20 0.033) |
| **Consensus: MoGe-2 `normals16~and`** | synthetic_val | 48.8 ± 1.1 | 49.5 ± 0.6 (med 16.3, a20 0.604) | 47.4 ± 1.0 (med 18.9, a20 0.521) | 50.3 ± 0.6 (med 16.4, a20 0.582) | 52.0 ± 0.9 (med 16.6, a20 0.571) |
| **Consensus: MoGe-2 `normals16~and`** | lightbox | 78.2 ± 1.2 | 77.9 ± 0.3 (med 69.0, a20 0.337) | 77.1 ± 1.1 (med 68.7, a20 0.241) | 84.8 ± 1.9 (med 85.7, a20 0.276) | 87.4 ± 2.1 (med 89.3, a20 0.247) |
| **Consensus: MoGe-2 `normals16~and`** | sunlamp | 88.0 ± 1.7 | 87.0 ± 0.9 (med 89.8, a20 0.249) | 87.1 ± 1.4 (med 88.3, a20 0.160) | 91.9 ± 1.1 (med 97.4, a20 0.209) | 95.9 ± 0.6 (med 101.4, a20 0.183) |

### C. Prediction-head latency (median ms per batch). Source: `RUN-20261009-003047-seed0/metrics/timing.csv` (also `outputs/comparisons/head_timing_tesseract_vs_flat_hopf.csv`)
The last column is flat time divided by Tesseract-greedy time; **> 1 means the Tesseract is faster.** GPU0 (RTX 6000 Ada) is shared with other users' jobs and noisy; the RTX A400 was idle; the CPU ran 1 thread.

| device | batch | depth (hypotheses K) | Tesseract greedy ms | Tesseract beam-4 ms | Flat argmax ms | flat / Tesseract-greedy |
|---|---:|---|---:|---:|---:|---:|
| RTX A400 | 1 | L2 (256) | 0.639 | 0.674 | 0.418 | 0.65× |
| RTX A400 | 1 | L3 (2,048) | 0.742 | 0.825 | 0.506 | 0.68× |
| RTX A400 | 1 | L4 (16,384) | 0.982 | 1.087 | 0.775 | 0.79× |
| RTX A400 | 1 | L5 (131,072) | 1.215 | 1.347 | 3.320 | 2.73× |
| RTX A400 | 1 | L6 (1,048,576) | 1.451 | 1.610 | 23.555 | 16.23× |
| RTX A400 | 256 | L2 (256) | 3.916 | 4.904 | 3.426 | 0.87× |
| RTX A400 | 256 | L3 (2,048) | 4.205 | 5.708 | 3.796 | 0.90× |
| RTX A400 | 256 | L4 (16,384) | 4.508 | 6.478 | 6.181 | 1.37× |
| RTX A400 | 256 | L5 (131,072) | 4.789 | 7.282 | 26.312 | 5.49× |
| RTX A400 | 256 | L6 (1,048,576) | 5.109 | 8.122 | 185.517 | 36.31× |
| CPU (1 thread) | 1 | L2 (256) | 1.370 | 1.414 | 0.983 | 0.72× |
| CPU (1 thread) | 1 | L3 (2,048) | 1.500 | 1.589 | 1.152 | 0.77× |
| CPU (1 thread) | 1 | L4 (16,384) | 1.637 | 1.753 | 2.260 | 1.38× |
| CPU (1 thread) | 1 | L5 (131,072) | 1.746 | 1.922 | 12.160 | 6.96× |
| CPU (1 thread) | 1 | L6 (1,048,576) | 1.863 | 2.097 | 91.781 | 49.26× |
| CPU (1 thread) | 256 | L2 (256) | 34.578 | 42.522 | 31.601 | 0.91× |
| CPU (1 thread) | 256 | L3 (2,048) | 35.745 | 47.697 | 35.433 | 0.99× |
| CPU (1 thread) | 256 | L4 (16,384) | 37.367 | 53.014 | 64.087 | 1.72× |
| CPU (1 thread) | 256 | L5 (131,072) | 38.620 | 58.062 | 345.780 | 8.95× |
| CPU (1 thread) | 256 | L6 (1,048,576) | 40.617 | 63.807 | 2583.859 | 63.61× |
| RTX 6000 Ada | 1 | L2 (256) | 0.499 | 0.531 | 0.054 | 0.11× |
| RTX 6000 Ada | 1 | L3 (2,048) | 0.748 | 0.795 | 0.058 | 0.08× |
| RTX 6000 Ada | 1 | L4 (16,384) | 0.991 | 1.052 | 0.060 | 0.06× |
| RTX 6000 Ada | 1 | L5 (131,072) | 1.233 | 1.309 | 0.398 | 0.32× |
| RTX 6000 Ada | 1 | L6 (1,048,576) | 1.472 | 1.544 | 2.534 | 1.72× |
| RTX 6000 Ada | 256 | L2 (256) | 0.538 | 0.572 | 0.231 | 0.43× |
| RTX 6000 Ada | 256 | L3 (2,048) | 0.795 | 0.859 | 0.238 | 0.30× |
| RTX 6000 Ada | 256 | L4 (16,384) | 1.055 | 1.136 | 0.442 | 0.42× |
| RTX 6000 Ada | 256 | L5 (131,072) | 1.299 | 1.403 | 1.950 | 1.50× |
| RTX 6000 Ada | 256 | L6 (1,048,576) | 1.561 | 1.685 | 14.667 | 9.39× |

- **Parameters:** the Tesseract head stays at ≈ 9.03M at every depth (the trunk dominates). The flat head grows: 8.8M (K = 256), 9.7M (2,048), 17.1M (16,384), **75.9M (131,072)**, **546.6M (1,048,576)**.
- **Backbone for context** (identical in both arms): DINOv3-L/16 at 256 px takes 5.9 ms at batch 1 and 75.7 ms at batch 32 on the RTX 6000 Ada (shared); on the RTX A400 it takes 64.7 ms and 1,124 ms.

## Pre-registered verdict
- **Speed:** the Tesseract head is faster on **both** the A400 GPU and the CPU at batch 1 **from L5 (131k hypotheses) upward.**
  - L5: 1.22 vs 3.32 ms (GPU), 1.75 vs 12.16 ms (CPU). L6: 1.45 vs 23.56 ms, 1.86 vs 91.78 ms.
  - Not at L2–L4.
  - On the shared RTX 6000 at batch 1 the crossover is only at L6.
  - **At SPACE-HOP's own size (3,072) the flat head is faster on every device** (a sub-millisecond difference).
- **Accuracy (pre-registered on mean error):** the Tesseract is as accurate as or more accurate than the flat head at matched hypothesis count, on both features.
  - DINOv3: L3 29.2° vs flat-3,072 50.7°; L5 27.7° vs flat-131k 69.9°.
  - Consensus: L3 48.8 ± 1.1° vs flat-3,072 49.5 ± 0.6° (within std); L5 47.4° vs flat-131k 52.0°.

## Interpretation
- **Discretization alone favours Hopf.** A balanced Hopf grid is more uniform and more precise per anchor than the Tesseract at every size (consistent with EXP-003). SPACE-HOP's fixed 12 rolls cap its precision near 6° at large K.
- **On DINOv3 features, prediction strongly favours the tree.** The flat head must learn one class per anchor. With 6,000 training images most fine classes are nearly empty, so accuracy collapses as K grows. The tree's coarse levels learn from every image.
- **On the consensus feature the picture is closer, and partly reversed:**
  - At 3,072 anchors the flat head ties the tree on mean error everywhere.
  - It beats the tree on acc@20: synthetic 0.604 vs 0.521, lightbox 0.337 vs 0.241, sunlamp 0.249 vs 0.160. It also has a lower synthetic median (16.3° vs 18.9°).
  - The low-dimensional geometric normals feature seems to suit direct anchor classification. Larger flat grids still lose on real images.
  - This gap may change with the full 48k training set. Untested.
- **Speed favours the tree only at fine resolution.** Tree cost grows with depth (O(L)) and its size is constant; flat cost and size grow with K (O(K)). The tree's advantage appears only when ≳ 10⁵ hypotheses are needed, which is the sub-3° discrete regime. At coarse resolution one flat matrix multiply is faster.
- **The backbone sets the pipeline speed at small K.** At K = 3,072 both heads add < 1 ms to a 5.9 ms (RTX 6000) or 65 ms (A400) backbone. At L6 the flat head costs 2.5 ms on the RTX 6000 (≈ 40% of the backbone) and 92 ms on CPU, against < 2 ms for the tree.

## Decision
- Keep the Tesseract tree as the fine-resolution head (L5–L6).
- Keep a flat Hopf head at about 3,000 anchors as a **strong baseline**: on the consensus feature it gives more near-correct predictions.
- **Wording constraints:**
  - say "the Tesseract head is faster" only for ≥ 131k hypotheses;
  - do not claim better discretization than Hopf;
  - do not claim better accuracy than a 3k flat Hopf head on the consensus feature.
- Next: repeat B at full data, and add SPACE-HOP-style continuous offsets to both arms (EXP-035/040).

