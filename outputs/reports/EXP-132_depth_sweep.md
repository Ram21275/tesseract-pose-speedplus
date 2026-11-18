# EXP-132: Tesseract depth sweep, L = 5, 6, 7, on T-LESS (30), YCB-V (21) and SYMSOL-T (5)

## Status
**Pre-registered 2026-11-18, before any depth-6 or depth-7 tree was trained** (DEC-032). Queued after EXP-131.
User request: "after this finishes queue experiment for testing till depth 7 on both datasets like we know 5 after that 6 and 7 do this also for symsol-T."

## Why
Every tree so far has depth 5 (`L = 5` was hard-coded). The L5 grid has a 2.51° mean / 6.04° worst-case cell error, which is close to
the in-mode error of the trees (EXP-131 discussion: fitted σ ≈ 2.8° on T-LESS). Depth 6 / 7 halve / quarter the cell (1.25° / 0.63°).
The error model predicts most of the gain at L6 and little at L7. No depth above 5 has ever been trained or evaluated for accuracy
(EXP-002: grid only; EXP-101: timing only). This experiment measures it, for the supplement.

## Question
With everything else fixed, how do rotation error and log-likelihood change from L5 to L6 to L7?
Descriptive; no pass criterion. Every value is reported whatever it is.

## Changed variable
Tree depth L ∈ {5, 6, 7} (`train_xobj.py --depth`). The soft-target temperature of level l is the mean own-cell error of level l,
as for levels 1–5: L6 1.25°, L7 0.63° (measured exactly as EXP-002, 2M Haar rotations, seed 0; levels 1–5 reproduce the EXP-002 table).

## Fixed (identical at every depth)
- DINOv3 branch (`tokens8`) only, A2 multi-path labels, the locked recipe otherwise (DEC-018: Transformer tree, 100 epochs, batch 256,
  clean view + 2 augmented copies, checkpoint by own-validation symmetry-aware greedy mean). YCB-V with `--extra-train ycbvsynall`.
- Data: T-LESS / YCB-V = the EXP-131 all-object manifests and caches; SYMSOL-T = the EXP-064 manifest and caches (the paper's).
- Seeds 0, 1, 2; seed-major (seed 0 at L6 and L7 for every object first). NaN rule as EXP-064.
- **Depth-5 trees are not retrained:** T-LESS / YCB-V use the EXP-131 trees, SYMSOL-T the paper's EXP-106 arm A2 trees.
  They are evaluated again with the same command as L6 / L7, so all three depths come from one evaluation protocol.

## Evaluation (fixed now)
- **Rotation:** `scripts/exp064_eval.py --variants DINO --modes greedy,orbit_product --depth L`, symmetry-aware mean / median error, R@2/5/10,
  validation and test, object-balanced, ± seed std. **Primary:** T-LESS orbit_product, YCB-V greedy, SYMSOL-T orbit_product (the
  DINO-only configurations of EXP-131 / the paper's SYMSOL-T configuration); the other mode is secondary.
- **Frame conversion with the GT translation** (`--pose-t gt`, the evaluator default), at every depth, because no translation blocks are
  trained here. EXP-129 measured the effect of predicted vs GT translation on rotation as negligible (15.187 vs 15.196° on T-LESS).
  SYMSOL-T needs no translation (egocentric targets).
- **Log-likelihood** of the tree density at the GT (`scripts/exp131_ll.py --depth L`), single view, averaged over symmetric copies (IPDF protocol).
- No BOP AR (needs translation blocks); AR stays an L5 number from EXP-131.

## Gates (before the queue)
- Depth-7 smoke test (train → depth-aware lookup → eval → LL) on GPU1: passed.
- Run lookup is depth-aware and defaults to depth 5, so the running EXP-131 queue is unaffected (checked on its finished runs).
- SYMSOL-T L5 re-evaluation of the EXP-106 A2 trees (seed 0, on GPU1) against the paper's per-seed values: greedy 2.570 vs 2.569, orbit 2.069 vs 2.068 (per shape ≤ 0.017°). The depth-5 code path is unchanged (literal 5 → `args.depth`); the residual is bf16 head arithmetic on the A400 vs the RTX 6000 used for the paper run, far below the seed std. In the queue all three depths are evaluated on GPU0.

## Cost estimate
Trees: T-LESS 180, YCB-V 126, SYMSOL-T 30. At the current EXP-131 pace (feature loading from the hdd3 caches dominates, ~15–20 min of
each ~17–22 min tree) about 51 h (T-LESS) and 44 h (YCB-V, then SYMSOL-T about 4 h) in two parallel queues, plus evaluation.
The training itself is ~2 min per tree; a faster cache layout would cut this to about 10 h / 8 h.

## Commands
`scripts/run_queue23.sh`: T-LESS queue waits for the EXP-131 T-LESS queue; YCB-V then SYMSOL-T waits for the EXP-131 YCB-V queue.

## Results
_Not run yet._
