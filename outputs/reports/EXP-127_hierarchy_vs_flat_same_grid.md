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

## Results
Not run.
