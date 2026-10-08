# EXP-030: Phase-3 MLP hierarchical predictor baseline on the full SPEED+ splits (frozen DEC-003 design)

## Status
Completed. Pre-registered on 2026-10-09, before any result.


## Keep score
**5 / 5: Strong support.** Full data cuts error about 4× (synthetic val 27.7° → 6.6° for DINOv3). The PoE fusion gain holds at full data (6.24 ± 0.07° vs 6.64 ± 0.16°) and is large on real images (lightbox mean −10.7°, sunlamp −9.7°). This is the Phase-3 control.

_Keep score rubric (0–5): 5 strong support: pre-registered goal met with a large effect, adopt. 4 support: goal met, carry forward. 3 partial: mixed or one-sided evidence, keep for further testing. 2 weak: goal not met but an informative side result. 1 negative: fails, reject, kept only as a recorded result. 0 inconclusive or not run._

## Research question
With the frozen design, does training on the full synthetic training split (47,966 images, about 8× the subset) improve each branch and the PoE fusion? Does the fusion gain hold at full data? This becomes the Phase-3 control for EXP-031–035.

The frozen design (DEC-003) is a training-free PoE of a DINOv3-L `grid4` probe and a MoGe-2 `normals16~and` probe.

## Component under test
Predictor (training data scale) and the frozen fusion

## Track
Trainable predictor + training-free fusion

## Pre-registered protocol
- **Manifest `full`:** all of synthetic/train (47,966), synthetic/validation (11,994), lightbox (6,740) and sunlamp (2,791). GT crops as in subset v1 (15% margin).
- **Features:** DINOv3-L/16 (`grid4`), MoGe-2 ViT-L normal checkpoint, and the EXP-015 consensus masks (AND rule, unchanged), all recomputed on `full`.
- **Probes:** the EXP-010/016 hierarchical MLP, unchanged hyperparameters (hidden 512, 100 epochs, AdamW 1e-3 / 0.05, batch 256, hard labels, L5). 3 seeds; selection on synthetic-val mean.
  - Only change: features are stored on the GPU as fp16 (`--gpu-dtype fp16`) because of shared-GPU memory. The model computes in fp32.
- **Fusion:** equal-weight PoE, greedy (EXP-025 script, `--subset full`).
- **Metrics:** mean / median geodesic error, acc@5/10/20, per domain. Lightbox and sunlamp are descriptive.
- **Comparison:** against the subset-v1 runs (EXP-010/016/025), same seeds.
- **Safety:** every step runs alone under `scripts/guarded.sh` (24–32 GB caps) in tmux.

## Runs
- **Chain:** `scripts/run_exp030.sh`, log `outputs/experiments/EXP-030_full_data_mlp_baseline/chain.log`. Every step ran alone under `guarded.sh`, with no failures and no memguard triggers.
- **Times:**
  - manifest `full`: 69,491 rows, hash in `outputs/data_manifests/subset_full.sha256.txt`;
  - DINOv3-L extraction: 13 min;
  - MoGe-2 extraction: 34 min (31 ms/img);
  - consensus masks: 24 min, CPU (`EXP-015_mask_consensus` run on `--subset full`, cache `consensus_masks__subset_full`);
  - 6 probes: 47–100 s each, best epochs 85–100;
  - PoE evaluation: `RUN-20261009-021455-seed0`.

## Results
Greedy decoding, L5, mean over 3 seeds (PoE: 3 paired seeds). Sources: `summary_by_features.csv` and `RUN-20261009-021455-seed0/metrics/metrics.csv`. Lightbox and sunlamp are descriptive.

| method | synthetic_val mean ± std / median | lightbox mean / median | sunlamp mean / median | acc@10 (synth / lightbox / sunlamp) | acc@20 (synth / lightbox / sunlamp) |
|---|---|---|---|---|---|
| A: DINOv3-L `grid4` | 6.64 ± 0.16 / 5.31 | 41.8 / 14.2 | 53.4 / 25.8 | 0.913 / 0.340 / 0.178 | 0.988 / 0.617 / 0.412 |
| B: MoGe-2 `normals16~and` | 17.79 ± 0.31 / 6.31 | 44.7 / 12.0 | 62.0 / 23.0 | 0.774 / 0.430 / 0.269 | 0.883 / 0.627 / 0.474 |
| **PoE(A,B)** | **6.24 ± 0.07 / 5.00** | **31.1 / 10.4** | **43.7 / 17.0** | **0.932 / 0.479 / 0.290** | **0.989 / 0.729 / 0.553** |
| AVG(A,B) | 6.61 ± 0.00 / 5.08 | 32.8 / 11.0 | 45.6 / 18.1 | 0.922 / 0.450 / 0.274 | 0.985 / 0.706 / 0.532 |

**Subset (6k) vs full (48k) training, synthetic val mean:**
- DINOv3: 27.7° → **6.6°**;
- consensus normals: 47.4° → **17.8°**;
- PoE: 25.9° → **6.2°**.

**Tree diagnostics, DINOv3 on synthetic val:**
- root accuracy 0.965;
- path accuracy: L3 0.668, full L5 0.177.

The L5 quantization floor is a **2.5° mean** (EXP-002), now about 40% of the 6.2–6.6° total error.

## Verdict
- **Data scale:** the subset runs were strongly data-limited. Every branch improves 2.7–4×.
- **Fusion:** PoE still beats DINOv3 alone on synthetic val, by 0.40° (more than 2× A's 0.16° std), so DEC-003 holds at full data. On real images the gain is large (descriptive): lightbox mean −10.7° and median −3.8°; sunlamp mean −9.7° and median −8.8°.
- **Phase-3 consequences:**
  - discrete quantization is now a major error component, so the tangent residual (EXP-035) is justified under its plan condition;
  - full-path L5 accuracy is only 0.18, so decoder improvements (EXP-031–034) target the deep levels.

**Not a like-for-like comparison:** SPACE-HOP reports mean rotation errors of 4.0° (synthetic val), 9.0° (lightbox) and 26.6° (sunlamp). It uses a fine-tuned JEPA ViT-B, continuous offsets and different training. The numbers are listed here only for scale.

## Decision
Keep. This is the Phase-3 control: frozen DEC-003 design, full data, MLP tree head, hard targets, greedy.
