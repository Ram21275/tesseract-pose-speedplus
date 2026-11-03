# EXP-127: Hierarchical prediction vs a flat classifier over the same 131,072 Tesseract leaves, and vs a balanced Hopf grid

## Status
**Pre-registered 2026-10-29, before any code was run on the GPU and before any EXP-127 result** (DEC-026; user request:
"Queue a controlled rotation experiment … to isolate the benefit of hierarchical prediction"). Queued in
`scripts/run_queue16.sh`, after the already-authorized queues (queue15: EXP-124, EXP-126; queue14: EXP-119).

## Question
EXP-113 compared the tree against a flat classifier over a *different* grid (SPACE-HOP Hopf, 10,923 directions × 12 rolls), so
hierarchy and discretization were confounded. EXP-127 separates them:
- **Same grid:** is the gain from the hierarchical (autoregressive, root-to-leaf) prediction itself, at the identical 131,072 leaf
  centres?
- **Balanced Hopf:** how does the full system compare with a flat classifier over an alternative, more uniform discretization of
  comparable size (1,771 Fibonacci directions × 74 rolls = 131,054 anchors; EXP-101 quantization 2.30° mean vs 2.51° for the
  Tesseract leaves)?

## Finalized recipe (what "current" means here)
- **The locked recipe is DEC-018 (EXP-109 arm A)**, and DEC-023 (2026-10-21) keeps it:
  - frozen DINOv3-L `tokens8` (64×1024), plus MoGe-2 `normals16~and` (256×4) in the PoE variant;
  - GT square crops;
  - one network per branch, trained with the EXP-109 arm-A augmentation (clean + `exp109_train_aug1/2`, one per sample per step,
    labels rotated by the `pre-` convention);
  - 100 epochs, batch 256, AdamW (lr 1e-3, wd 0.05), OneCycle (pct_start 0.05), bf16, dropout 0.1, no gradient clipping;
  - checkpoint = lowest clean synthetic-val greedy/argmax mean (evaluated every 5 epochs);
  - PoE = equal-weight product across branches;
  - orbit voting = 4 views (0/90/180/270°), top-8 per view (32 candidates), orbit-product scoring.
- **Orbit distillation (EXP-117) is not part of the locked recipe.** DEC-023: "It is not part of the recipe used by later
  experiments unless the user says so." The request includes distillation only "if it belongs to the finalized recipe", so it is
  **excluded** from all three arms. The tree reference is therefore the EXP-109 arm-A system (PoE + orbit 3.87° synthetic), not
  the distilled 3.69° headline. A distillation extension (each family taught by its own orbit over the 3 training views,
  candidate weights transported into the student's view, level-marginal CE for the tree vs anchor-level soft CE for the flat
  heads) would need its own amendment before running.
- **What differs from EXP-113.** The recipe is unchanged since EXP-113; what is new:
  - the same-grid flat arm;
  - the balanced Hopf grid instead of SPACE-HOP's 12-roll grid;
  - quantization on identical rotations;
  - acc@5/20 and per-image outputs;
  - identical-condition timing and memory for all arms.

## Arms (per branch: DINOv3 and MoGe-2; seeds 0, 1, 2)
| Arm | Rotation representation | Prediction | Supervision | Checkpoints |
|---|---|---|---|---|
| **T** tree (reference) | Tesseract L5: 4 charts × 8⁵ = 131,072 cells (leaf centres) | autoregressive root → 5 levels (4-way + 5 × 8-way) | per-level geodesic soft targets over the children (EXP-036, τ per level, leaf τ = 2.51°) | **reused:** EXP-109 arm-A runs (`train_probe_p3.py`; they *are* the locked configuration) |
| **FL** flat, same grid | the same 131,072 L5 leaf centres (`tesseract.codebook(5)`, leaf-id order) | one 131,072-way softmax | geodesic soft targets over the 32 nearest centres, τ = 2.51° (the tree's leaf temperature) | **new** (`train_flat_tf.py --grid tesseract_L5`) |
| **FH** flat, balanced Hopf | 1,771 Fibonacci directions × 74 rolls = 131,054 (`spacehop_hopf_grid(1771, 74)`; reproduces EXP-101: 2.30° mean on Haar) | one 131,054-way softmax | same as FL | **new** (`train_flat_tf.py --grid hopf1771x74`) |

**Necessary architectural and supervision differences (documented, not tuned):**
- **Same trunk.** All arms share the same input layer (LN + Linear → 256), 2-D positional encoding, 1 encoder layer and
  2-layer decoder.
  - The tree feeds the decoder [ROOT] + 5 path tokens and has a 4-way root head and an 8-way child head.
  - The flat arms run the decoder on the [ROOT] query only and add a linear K-way head (256·K + K ≈ 33.6 M parameters).
  - Parameter counts are reported, not matched; this capacity difference is inherent to flat classification.
- **Targets.** The tree's targets are per-level distributions over 8 children. The flat targets are a single distribution over
  the 32 nearest anchors with the tree's leaf temperature.
- **Hard anchor of a rotation** (used when a flat model scores an orbit candidate):
  - FL uses the candidate's Tesseract L5 cell (encode → leaf id), the same mapping the tree's teacher-forced scoring uses. The
    two grids then differ only in how the leaf probability is produced. `leaf_id(encode(codebook(5))) == arange(131072)` was
    verified.
  - FH uses the nearest anchor.
- **Candidates.** Tree candidates per view are its beam-8 leaves (approximate top-8 of the leaf distribution). Flat candidates
  are the exact top-8 anchors.
  - Both give 32 candidates over 4 views.
  - Both are scored by the orbit product Σ_k log p_k(map_k(c)): teacher-forced leaf log-probability for the tree, anchor
    log-probability for the flat arms. For the flat arms this needs no extra forward pass.
- **PoE.**
  - Tree: per-level product of the two branches' distributions.
  - Flat: per-anchor product of the two branches' anchor distributions.

## Evaluation (identical for all arms; `scripts/orbit_eval.py`, one run for all three families)
- **Variants:** DINOv3-only and PoE.
- **Inference:**
  - greedy (single view);
  - orbit_product (primary orbit rule, the locked recipe's);
  - orbit_vote32 (descriptive only).
- **Domains:**
  - synthetic val (`exp109_test3` synthetic rows);
  - synthetic val-aug (EXP-109 seed-1000 views);
  - lightbox and sunlamp (`exp109_test3`, scored only, never used for selection or tuning).
- **Metrics:** mean, median, acc@5/10/20, fraction > 90°, severe flips (> 150°), flip transitions vs the same arm's greedy.
  - 3-seed mean ± seed std.
  - Per-image errors and predicted quaternions go to `predictions/per_image.csv.gz`.
- **Quantization floor** (`scripts/exp127_quant.py`): nearest-anchor error on the *identical* synthetic-val and val-aug
  rotations, for:
  - Tesseract encode → decode (the tree leaf and FL's hard anchor);
  - Tesseract nearest centre;
  - balanced Hopf nearest;
  - SPACE-HOP 10923×12 nearest (descriptive).
- **Cost** (`scripts/exp127_timing.py`, one process, seed-0 checkpoints):
  - trainable parameters per branch;
  - prediction-head latency (per-view decode / top-8, plus orbit scoring);
  - full-pipeline latency (crop warp → DINOv3 (→ MoGe-2 + consensus) → head), measured at batch 1 and 64 with 2 warm-ups and
    the median of 5;
  - inference peak GPU memory;
  - training peak GPU memory and ms per step (20 AdamW steps at batch 256, each arm's own loss).
  - All 12 systems are interleaved round by round on the same 64 lightbox crops. GPU load is logged; the GPU is shared, so times
    are upper bounds.
  - The flat trainings also record their own peak training memory.
- **Selection:** checkpoints by clean synthetic val (as above); inference settings fixed now (no tuning). If any choice had to be
  made, it would use synthetic val only.

## Criteria (fixed now)
A difference "passes" if the 3-seed synthetic-val means differ by more than the larger seed std (EXP-113 rule).
1. **Hierarchy (T vs FL, same grid), primary.** Primary configuration: PoE + orbit_product on synthetic val. Secondary: PoE
   greedy, DINOv3-only greedy and orbit, and synthetic val-aug.
   - T better: hierarchical prediction itself improves accuracy at an identical discretization.
   - Tie or FL better: the claim "the tree's advantage comes from hierarchical prediction" is **withdrawn**. The EXP-113 gain
     is then attributed to the grid, and the paper's flat-baseline sentence is rewritten accordingly.
2. **Discretization (FL vs FH, both flat)**, by the same rule: the effect of grid geometry on a flat classifier, descriptive for
   the claim.
3. **System (T vs FH), PoE + orbit.**
   - T better: the tree is supported against an alternative discretization of comparable coverage.
   - FH ties or wins: the paper states that a flat balanced-Hopf classifier matches or beats the tree in accuracy, and the tree's
     case rests on cost (parameters, memory, latency) only if the cost numbers support it.
   - The balanced grid's lower quantization floor (2.30° vs 2.51°) is reported next to this.
4. **Real domains:** lightbox and sunlamp are descriptive, never tuned or used for selection.
5. **Cost:** descriptive.

## Runs
- **Trainings (new):** 2 grids × 2 branches × 3 seeds = 12, in `EXP-127_hierarchy_vs_flat_same_grid/`. A 2-epoch smoke run
  (`SMOKE_exp127`) comes first and is excluded from all tables.
- **Reused:** the EXP-109 arm-A tree checkpoints, seeds 0–2, DINOv3 `tokens8` and MoGe-2 `normals16~and`.
  - sha256 checksums go to `source_checkpoints.txt` in the experiment directory, with a copy of `exp109_convention.txt`.
  - Gate: the evaluator's tree rows must reproduce EXP-113 / EXP-117 (PoE + orbit 3.87, PoE greedy 4.85 synthetic val). A
    mismatch invalidates the evaluation run.
- **Cost:** about 12 × 8–17 min of training, plus evaluation and timing, about 4–5 GPU-hours in total.

## Amendment 2026-10-29 (before any trained-arm run)
- **Inference memory per system** = resident weights the system uses (DINOv3, plus MoGe-2 for PoE, plus its own head models) +
  transient peak above the allocated baseline. All models share one timing process, so the raw `max_memory_allocated` would
  not be per-system. Training memory is unchanged: one model at a time.
- **Smoke stage extended:**
  - 2-epoch flat trainings for both branches;
  - an evaluator smoke run (tree + smoke flat, seed 0, greedy and orbit_product, per-image outputs);
  - **tree reproduction gate** (`scripts/exp127_gate.py`): the seed-0 tree rows must match EXP-113's seed-0 rows within
    0.02°, or the queue stops before the 12 trainings.

## Results
### Quantization floor on identical rotations
Run `RUN-20261029-003312-seed0` (CPU, no learning; 11,994 rotations per set). Values are in degrees.

| grid | synthetic val mean / median / p95 / max | val-aug mean / median / p95 / max |
|---|---|---|
| Tesseract L5 encode → decode (tree leaf; FL hard anchor) | 2.507 / 2.486 / 3.950 / 5.776 | 2.504 / 2.480 / 3.963 / 5.855 |
| Tesseract L5 nearest centre | 2.476 / 2.449 / 3.914 / 5.776 | 2.474 / 2.449 / 3.907 / 5.820 |
| balanced Hopf 1771×74 nearest | **2.291 / 2.374 / 3.214 / 3.840** | **2.305 / 2.384 / 3.235 / 3.811** |
| SPACE-HOP Hopf 10923×12 nearest (EXP-113, descriptive) | 6.057 / 5.436 / 12.947 / 14.973 | 6.032 / 5.394 / 12.899 / 15.004 |

- The balanced Hopf grid has the lower floor: 0.22° lower in mean and markedly lower at the tail (max 3.8° vs 5.8°).
- Encoded cell and nearest centre differ by only 0.03° on average.

### Trainings, evaluation and cost
Queued (queue16).

## Results: evaluation (2026-11-03, `RUN-20261103-143538-seed0`)
- **Interruptions:** run after the GPU loss and the resume; the 12 flat trainings finished at 14:35.
- **Gate:**
  - smoke stage: GATE PASS (the evaluator reproduces EXP-113 seed 0);
  - the tree rows reproduce EXP-113 exactly: PoE + orbit 3.87, PoE greedy 4.85.
- **Format:** 3-seed mean ± seed std, degrees.

| head | inference | synth val | val-aug | lightbox | sunlamp |
|---|---|---|---|---|---|
| tree (2.4 M) | PoE greedy | **4.85 ± 0.03** | **9.75 ± 0.06** | **17.33** | **20.15** |
| flat, same 131,072 leaves | PoE greedy | 8.19 ± 0.15 | 16.88 ± 0.09 | 23.83 | 32.87 |
| flat, balanced Hopf 1771×74 | PoE greedy | 8.14 ± 0.09 | 17.14 ± 0.18 | 24.05 | 31.21 |
| tree | PoE orbit_product | **3.87 ± 0.03** | **5.82 ± 0.10** | **12.30** | **12.25** |
| flat, same leaves | PoE orbit_product | 5.34 ± 0.02 | 8.61 ± 0.12 | 13.04 | 16.73 |
| flat, balanced Hopf | PoE orbit_product | 5.28 ± 0.02 | 8.58 ± 0.14 | 12.88 | 16.29 |
| tree | DINO greedy / orbit | 4.84 / 4.21 | 9.88 / 6.25 | 22.55 / 17.40 | 22.00 / 15.27 |
| flat, same leaves | DINO greedy / orbit | 7.88 / 5.48 | 15.34 / 8.30 | 31.14 / 17.75 | 38.01 / 19.69 |
| flat, balanced Hopf | DINO greedy / orbit | 7.82 / 5.44 | 15.04 / 8.20 | 30.70 / 17.53 | 35.87 / 18.43 |

- **Lightbox, PoE + orbit:**

  | head | median | acc@5 | acc@10 | acc@20 | > 150° |
  |---|---|---|---|---|---|
  | tree | 5.26 | 0.465 | 0.836 | 0.940 | 3.0% |
  | flat, same leaves | 6.35 | 0.333 | 0.799 | 0.938 | 2.5% |
  | flat, balanced Hopf | 6.25 | 0.339 | 0.804 | 0.938 | 2.3% |

### Criteria
1. **Hierarchy (tree vs same-grid flat), PoE + orbit, synthetic val: PASS.** 3.87 ± 0.03 vs 5.34 ± 0.02. All secondary configurations agree: PoE greedy 4.85 vs 8.19, DINOv3 greedy and orbit, and val-aug. **The claim "the tree's advantage comes from hierarchical prediction" is supported.**
2. **Discretisation (same-grid vs balanced Hopf, both flat):** balanced Hopf is better by 0.06 (5.28 vs 5.34; larger std 0.02). That passes the rule but is small, despite the lower quantisation floor (2.29 vs 2.51).
3. **System (tree vs balanced Hopf), PoE + orbit: PASS.** 3.87 vs 5.28.
4. **Real domains (descriptive):**
   - The tree is better in mean on both domains in every configuration.
   - With orbit voting on lightbox, the flat heads have slightly fewer > 150° flips (2.3–2.5% vs 3.0%), but a higher median and lower acc@5.

### Interpretation
- At an identical grid, the hierarchical decoder beats a flat softmax by about 1.5° with orbit aggregation and about 3.3° from a single view, using roughly 14× fewer parameters.
- The EXP-113 gap (7.53 vs 3.87) was mostly hierarchy, not grid: the SPACE-HOP grid's coarse rolls account for 7.53 − 5.28 ≈ 2.25°, hierarchy for 5.28 − 3.87 ≈ 1.4°, and the remainder.
- Timing and memory are pending (next block).

## Results: cost (`RUN-20261103-151108-seed0`; one interleaved run, shared RTX 6000 Ada under load; upper bounds)

| head (DINOv3 branch) | params | head ms b1 greedy / orbit | full pipeline ms b1 greedy / orbit | head ms/img b64 greedy / orbit | train step ms (b256) | train peak GB | inference memory GB |
|---|---|---|---|---|---|---|---|
| tree | **2.39 M** | 34.3 / 184.4 | 58.5 / 270.5 | 0.53 / 3.58 | **18.4** | **3.6** | 1.18 |
| flat, same leaves | 36.08 M | **9.7** / 71.7 | **33.5** / 150.6 | **0.13** / 1.16 | 37.7 | 4.5 | 1.36 |
| flat, balanced Hopf | 36.07 M | 9.7 / **62.0** | 38.9 / 158.3 | 0.13 / 1.35 | 37.3 | 4.5 | 1.36 |

- **PoE variant:**
  - head at batch 1, greedy: tree 63.6 ms vs flat 15 ms;
  - full pipeline at batch 1, greedy: 236 ms vs 177–189 ms. MoGe-2 and the CPU mask dominate it.
  - Inference memory is 2.8 vs 3.1 GB, dominated by the backbones.
- **Verdict (descriptive):**
  - On a GPU the flat heads are faster (one matmul vs six sequential decoder steps): about 3.5× for the head and about 1.7× end-to-end at batch 1.
  - The tree is 15× smaller and trains about 2× faster per step with about 20% less memory.
  - The tree's case is accuracy, size and training cost, not GPU latency. CPU timing was not repeated here; EXP-101's older-architecture CPU result favoured the tree.
- **Paper:** "Cost of the output layer" paragraph (`sec/5_ablations.tex`).
