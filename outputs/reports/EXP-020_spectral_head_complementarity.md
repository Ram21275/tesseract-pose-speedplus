# EXP-020: Spectral head-matching as a complementarity measure (CASS option 4)

## Status
Completed. Pre-registered on 2026-10-08, before any result. User-requested; option 4 of 4. Descriptive analysis; no claim thresholds.


## Keep score
**2 / 5: Weak.** The spectral head distance carries a weak, inverted signal on real images but none on synthetic validation, so it can't be used under our selection rules. Plain prediction disagreement is a better reliability cue.

_Keep score rubric (0–5): 5 strong support: pre-registered goal met with a large effect, adopt. 4 support: goal met, carry forward. 3 partial: mixed or one-sided evidence, keep for further testing. 2 weak: goal not met but an informative side result. 1 negative: fails, reject, kept only as a recorded result. 0 inconclusive or not run._

## Research question
CASS pairs attention heads by the Wasserstein distance between their key-graph eigenvalue spectra. Does that label-free spectral distance between DINOv3-L and MoGe-2 measure their complementarity, and does it predict per image when they disagree or fail?

## Track
Training-free analysis.

## Pre-registered protocol
- **Data:** per-image, per-head top-20 eigenvalue spectra of the last-block key graphs, saved during the EXP-018 extraction pass. DINOv3-L has 16 heads at 16×16; MoGe-2 has 16 heads at 37×37.
- **Per image:** the 16×16 Wasserstein cost matrix (CASS normalization), the Hungarian matching, and the mean matched distance D.
- **Reported per domain:**
  - the distribution of D;
  - the Spearman correlation of D with the DINO–MoGe-2 prediction disagreement, and with DINOv3-L and MoGe-2 errors (EXP-010/013 seed-0 probes);
  - the AUROC of D for "DINOv3-L error > 20°".
- **Selection:** none. Real-domain numbers are descriptive.

## Runs
`RUN-20261009-001320-seed0`: completed, CPU only, `guarded.sh 16`. Spectra come from the EXP-018 extraction pass.

## Results
Source: `metrics/metrics.csv`. D = mean Wasserstein distance between the spectra of matched DINOv3-L/MoGe-2 heads (CASS cost), per image. Errors are from the seed-0 grid4 probes.

| domain | D mean ± std | Spearman(D, DINO–MoGe disagreement) | Spearman(D, DINO error) | Spearman(D, MoGe error) | AUROC of D for DINO error > 20° | AUROC of *prediction disagreement* for DINO error > 20° |
|---|---|---:|---:|---:|---:|---:|
| synthetic_val | 0.375 ± 0.046 | −0.100 | −0.052 | −0.111 | 0.481 | **0.753** |
| lightbox | 0.320 ± 0.048 | −0.222 | −0.266 | −0.232 | 0.337 | **0.698** |
| sunlamp | 0.346 ± 0.050 | −0.235 | −0.270 | −0.298 | 0.320 | **0.678** |

Using the full head-pair average instead of matched pairs gives the same picture. Head-pair matching frequencies are in `tables/pair_frequency_dino_x_moge.csv`.

## Interpretation (descriptive; no claim thresholds were pre-registered)
- **The sign is the opposite of the CASS intuition.** On real images, *more* spectral dissimilarity between the two models goes with *lower* error and less disagreement (ρ ≈ −0.22 to −0.30). An AUROC of 0.32–0.34 means low D flags DINO failures (0.66–0.68 if inverted).
  - A plausible but untested reading: on cluttered or glare images both attention graphs collapse onto the background structure and become alike, so low D marks hard images.
- **There is no signal on synthetic val** (AUROC 0.48, |ρ| ≤ 0.11). It can't be selected or calibrated under the split policy without a recorded real-domain validation split.
- **Simple prediction disagreement is better on every domain** (AUROC 0.68–0.75), cheaper, and already available.

## Decision
Do not adopt the spectral distance as a reliability signal. Keep prediction disagreement (EXP-014) as the reliability cue for Phase-2 and Phase-5. Revisit D only if a real-domain validation split is created.
