# EXP-115: What causes flip recovery? Photometric-only vs rotation-only vs combined augmentation, each with and without orbit inference

## Status
**Pre-registered 2026-10-16, before any run** (DEC-019, user: "Photometric-only, rotation-only, combined; each with/without orbit inference as this tells us what is actually causing flip recovery").

## Arms (locked recipe otherwise; both variants; seeds 0–2)
- **combined:** the EXP-109 arm-A trees (photometric + in-plane rotation; already trained).
- **photometric-only (new):** 2 copies with the EXP-109 photometric pipeline (same seeds 1, 2), **no rotation**; labels unchanged.
- **rotation-only (new):** 2 copies with the EXP-109 in-plane rotation (same angle seeds 1, 2), **no photometric change**; labels "pre−".
- **none:** the EXP-036 clean trees (reference).
- Each arm is evaluated **greedy** and with **orbit_product**, on the same evaluation views as EXP-109 (`exp109_valaug*`, `exp109_test3_r*`).
- New features: `exp115_train_photo1/2`, `exp115_train_rot1/2` (compact, hdd2).

## Metrics
Synthetic val, val-aug, lightbox, sunlamp: mean / median, fraction > 150°, flip transitions vs the arm's own greedy and vs EXP-036 greedy (denominators).

## Criteria (synthetic val and val-aug; real descriptive)
1. **Rotation augmentation matters:** combined beats photometric-only (each with orbit) by more than the larger seed std on val-aug.
2. **Photometric augmentation matters:** combined beats rotation-only (each with orbit) by the same rule.
3. **Orbit needs rotation training:** the orbit gain (greedy − orbit) is larger for the rotation-trained arms than for photometric-only. Descriptive, with the gains and stds reported.
4. All arms are reported whatever the direction.

Evaluator: `scripts/orbit_eval.py`. Trainer: `scripts/train_probe_p3.py --aug-tags`.

## Runs / Results / Decision
Not started.
