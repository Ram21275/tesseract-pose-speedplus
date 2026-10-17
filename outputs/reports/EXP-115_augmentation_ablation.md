# EXP-115: What causes flip recovery? Photometric-only vs rotation-only vs combined augmentation, each with and without orbit inference

## Status
**Completed 2026-10-17.** Criteria 1–2 PASS on val-aug (combined + orbit 5.82 vs photometric-only 11.16 and rotation-only 10.49, PoE). **But the real domains attribute the gain to photometric augmentation:** photometric-only ≈ combined on lightbox / sunlamp (PoE + orbit 12.29 / 12.37 vs 12.30 / 12.25), while rotation-only is far worse there (18.69 / 20.70) and only helps synthetic precision. Orbit voting helps every arm, including the clean EXP-036 trees. Score **4/5 Support** (it answers the question; it narrows the claim).

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

## Amendment 2026-10-17 (before any EXP-115 model was trained or evaluated)
- The first training attempt (00:46) was stopped by memguard: system available RAM fell below the 16 GB floor (other users' load). Its evaluation ran without the new arms and is archived to `EXP-099_p3_smoke/…-exp115eval-without-new-arms`, not used.
- Re-run: `scripts/run_queue7.sh`.
- The evaluation also reports **weighted voting (vote32)** next to the orbit product. The user decided to keep multi-hypothesis orbit inference with either rule (EXP-114 found them comparable).
- Criteria are unchanged; they are applied to the orbit product, with vote32 reported alongside.

## Runs
- Trainings `RUN-20261017-10…` – `12…` (photometric and rotation arms, both branches, seeds 0–2); features `exp115_train_photo1/2`, `exp115_train_rot1/2`.
- Evaluation: `RUN-20261017-121554-seed0` (greedy, orbit product, weighted vote).
- First attempt stopped by memguard (archived; see amendment).

## Results (3-seed mean, °; ± std on the synthetic sets; > 150° fraction on real)
| variant | training augmentation | inference | synth val | synth val-aug | lightbox (> 150°) | sunlamp (> 150°) |
|---|---|---|---|---|---|---|
| PoE | none (EXP-036) | greedy | 5.54 ± 0.07 | 30.84 ± 0.44 | 26.90 (0.059) | 36.66 (0.074) |
| PoE | none (EXP-036) | orbit | 4.62 ± 0.04 | 19.84 ± 0.34 | 20.25 (0.048) | 24.83 (0.043) |
| PoE | photometric only | greedy | 5.00 ± 0.03 | 17.07 ± 0.08 | 17.38 (0.039) | 19.68 (0.032) |
| PoE | **photometric only** | **orbit** | 4.25 ± 0.06 | 11.16 ± 0.08 | **12.29** (0.027) | **12.37** (0.015) |
| PoE | rotation only | greedy | 4.91 ± 0.00 | 17.55 ± 0.15 | 24.30 (0.060) | 31.52 (0.062) |
| PoE | rotation only | orbit | **3.78** ± 0.03 | 10.49 ± 0.17 | 18.69 (0.052) | 20.70 (0.038) |
| PoE | combined (A) | greedy | 4.85 ± 0.03 | 9.75 ± 0.06 | 17.33 (0.042) | 20.15 (0.033) |
| PoE | **combined (A)** | **orbit** | 3.87 ± 0.03 | **5.82** ± 0.10 | **12.30** (0.030) | **12.25** (0.017) |
| DINOv3 | none | greedy / orbit | 5.35 / 4.98 | 24.73 / 17.22 | 37.07 / 30.98 | 41.46 / 32.12 |
| DINOv3 | photometric only | greedy / orbit | 4.91 / 4.50 | 12.33 / 8.68 | 22.80 / 17.59 | 22.31 / 15.42 |
| DINOv3 | rotation only | greedy / orbit | 4.89 / 4.07 | 19.17 / 12.58 | 35.07 / 28.86 | 37.64 / 28.41 |
| DINOv3 | combined | greedy / orbit | 4.84 / 4.21 | 9.88 / 6.25 | 22.55 / 17.40 | 22.00 / 15.27 |

Weighted voting (vote32) behaves like the product: slightly better on synthetic, slightly worse on real; full table in the run.

## Criteria
1. **Rotation augmentation matters (combined vs photometric-only, orbit, val-aug): PASS** (5.82 vs 11.16 PoE; 6.25 vs 8.68 DINOv3). Caveat: val-aug itself contains random in-plane rotations, so this proxy favours rotation-trained arms by construction.
2. **Photometric augmentation matters (combined vs rotation-only, orbit, val-aug): PASS** (5.82 vs 10.49 PoE; 6.25 vs 12.58 DINOv3).
3. **Orbit gain by arm (greedy − orbit, PoE, val-aug; descriptive):** none 11.0 · photometric 5.9 · rotation 7.1 · combined 3.9.
4. **Real domains (descriptive):**
   - photometric-only ≈ combined (lightbox 12.29 vs 12.30; sunlamp 12.37 vs 12.25), with as few or fewer severe flips (0.027 / 0.015 vs 0.030 / 0.017);
   - rotation-only stays near the clean trees (18.69 / 20.70 vs 20.25 / 24.83);
   - orbit voting alone, on clean trees, already gives −6.7° / −11.8°.

## Interpretation
- **What causes the real-domain flip recovery:** photometric (lighting / noise / blur / glare / occlusion) augmentation plus orbit voting, not the in-plane rotation augmentation. Rotation-only training plus orbit is far worse on real images than photometric-only plus orbit.
- **The rotation augmentation** (with the derived label transform) improves synthetic precision (best synthetic orbit, 3.78°) and the rotated val-aug proxy, but adds nothing on real images once photometric augmentation is present.
- **Orbit voting works without rotation training:** clean trees + orbit already remove many flips. Its effect grows when the trees are made robust to lighting.
- **Consequences for the claims:**
  - The "geometry-correct rotation augmentation" is **not** what fixes real flips. Its role is precision and in-plane robustness.
  - The real-domain story is **lighting robustness + multi-hypothesis orbit voting + the Tesseract tree** (precision, EXP-113).
  - The val-aug proxy over-rewards rotation training. A purely photometric proxy would have been the cleaner selection set (a lesson for EXP-064).

## Decision
- Keep the combined recipe (DEC-018): it is best on clean synthetic val (orbit 3.87 vs photometric 4.25) and on val-aug, and ties photometric-only on real.
- Report the attribution honestly: photometric augmentation and orbit voting carry the real-domain gain.
