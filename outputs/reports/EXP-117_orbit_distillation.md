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
