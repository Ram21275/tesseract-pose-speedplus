# EXP-117: Distil the orbit system into a single-view student

## Status
**Pre-registered 2026-10-17, before any code or run** (DEC-021, user idea 2: "Use the successful four-rotation model as an offline teacher. Train a one-image student to reproduce its multiple orientation hypotheses and their probabilities across tree levels").

## Question
Can a single-view tree recover a substantial part of the orbit system's accuracy gain at one-view cost (EXP-112: DINOv3-only greedy 15 ms vs + orbit 70 ms; PoE 64 ms vs 279 ms)?

## Teacher (frozen, offline)
- The EXP-109 arm-A trees of the same seed and variant, applied as an **orbit over the training image's available views**: clean + `exp109_train_aug1` + `exp109_train_aug2` (known roll angles; orbit voting needs only known angles, not 90° steps). No new extraction.
- **Candidates:** beam-8 leaves of each of the 3 views, mapped to the clean frame (24).
- **Weights:** the orbit-product score, softmax-normalized over the 24.
- Targets are precomputed once (`teacher_targets`: 24 rotations + weights per image) and cached under the repo (small: about 48k × 24 × 5 floats).

## Student (trained per branch; seeds 0–2; EXP-109 A recipe and views otherwise)
- The input is one view v (clean / aug1 / aug2, uniform).
- Loss = L_GT(v) + β·H(teacher → v), where the teacher candidates are transported into v's frame (c′ = Exp(−θ_v·d) ⊗ c), and H is the EXP-044 level-marginal cross-entropy weighted by the teacher weights. **β = 1.0 (fixed now).**
- Initialized from scratch (same as A).

## Evaluation
- Single-view greedy (PoE and DINOv3-only), the same views and evaluator as EXP-109 / 113 / 115.
- Also orbit, for completeness.
- **Retention** = (A greedy − student greedy) / (A greedy − A orbit) per domain.

## Criteria (synthetic val / val-aug; real descriptive)
1. **Distillation helps:** the student's 3-seed single-view greedy val-aug mean is below A greedy's by more than the larger seed std (DINOv3-only primary; PoE secondary), and not worse on clean synthetic val by more than it.
2. **Retention** is reported per domain (descriptive; no threshold).
3. Real domains: descriptive, with > 150° flips.

## Claim boundary
Pose / distribution distillation is established. The interest is practical (one-view cost). A combined EXP-116 + 117 model is not pre-registered here and would need its own entry.

## Cost
Teacher targets about 0.5 GPU-hour; 6 trainings (2 branches × 3 seeds, about 15–20 min each); evaluation. About 3–4 GPU-hours. No new features.

## Amendment 2026-10-17 (before any EXP-117 run)
- Variant-specific teachers (as pre-registered), giving 3 students per seed:
  - DINOv3 student ← DINO-only orbit teacher (for the DINOv3-only system);
  - DINOv3 and MoGe-2 students ← PoE orbit teacher (for the PoE system).
  - 9 trainings.
- Teacher targets: `scripts/exp117_teacher.py` → `outputs/experiments/EXP-117_orbit_distillation/teacher_{PoE,DINO}_seed{s}.npz`. Trainer: `scripts/train_consistency.py --mode distill --teacher-variant`.

## Runs / Results / Decision
Not started.

## Results (completed 2026-10-20; evaluations `RUN-20261020-113318` (DINOv3-only), `RUN-20261020-114253` (PoE); 3-seed means ± seed std, degrees)
| variant | arm | inference | synth val | val-aug | lightbox | sunlamp | lightbox > 150° |
|---|---|---|---|---|---|---|---|
| DINOv3 | A (EXP-109) | greedy | 4.84 ± 0.05 | 9.88 ± 0.22 | 22.55 | 22.00 | 0.060 |
| DINOv3 | **student** | greedy | **4.57 ± 0.05** | **9.33 ± 0.18** | 21.61 | 20.55 | 0.058 |
| DINOv3 | A | orbit product | 4.21 | 6.25 | 17.40 | 15.27 | 0.048 |
| DINOv3 | student | orbit product | 3.90 | 5.74 | 16.43 | 13.64 | 0.045 |
| PoE | A (EXP-109) | greedy | 4.85 ± 0.03 | 9.75 ± 0.06 | 17.33 | 20.15 | 0.042 |
| PoE | **student** | greedy | **4.56 ± 0.03** | **9.22 ± 0.10** | 15.90 | 18.33 | 0.037 |
| PoE | A | orbit product | 3.87 ± 0.03 | 5.82 | 12.30 ± 0.50 | 12.25 ± 0.37 | 0.030 |
| PoE | **student** | **orbit product** | **3.69 ± 0.02** | **5.40** | **11.21 ± 0.42** | **11.09 ± 0.49** | **0.026** |
| PoE | student | orbit vote32 | 3.27 | 5.59 | 11.35 | 11.91 | 0.027 |

**Criterion 1 (distillation helps): PASS.**
- DINOv3 greedy val-aug 9.33 vs 9.88 (Δ 0.55 > 0.22); PoE 9.22 vs 9.75 (Δ 0.53 > 0.10).
- Clean synthetic val is better, not worse (4.57 vs 4.84; 4.56 vs 4.85).

**Retention** = (A greedy − student greedy) / (A greedy − A orbit), synth / val-aug / lightbox / sunlamp:
- DINOv3: 43% / 15% / 18% / 22%;
- PoE: 30% / 13% / 28% / 23%.
The single-view student recovers about 15–30% of the orbit gain at greedy cost.

**Descriptive: the student is also a better base for orbit inference.** Student + orbit is the best system so far: PoE 3.69 / 11.21 / 11.09 vs EXP-109 3.87 / 12.30 / 12.25. That is −1.1 / −1.2° on real (about 2 seed std), with fewer > 150° flips. Selection on synthetic val (DEC-006) also prefers it (3.69 < 3.87).

**Decision:** distillation is adopted as a training-stage refinement, pending the user's decision on updating the locked recipe (DEC-018).
- The claim is "orbit-consistency distillation improves the single-view tree and the orbit system".
- Prior art: self-distillation from test-time-augmentation ensembles is known, so the novelty is limited to its use with hierarchical rotation targets.
