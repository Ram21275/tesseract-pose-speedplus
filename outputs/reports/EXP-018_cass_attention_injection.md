# EXP-018: CASS-style spectral attention injection, DINOv3 → MoGe-2 and MoGe-2 → DINOv3 (CASS option 2)

## Status
Completed. Pre-registered on 2026-10-08, before any result. User-requested; option 2 of 4.


## Keep score
**1 / 5: Negative.** Injection did not fix MoGe-2's mask and made pose worse in both directions.

_Keep score rubric (0–5): 5 strong support: pre-registered goal met with a large effect, adopt. 4 support: goal met, carry forward. 3 partial: mixed or one-sided evidence, keep for further testing. 2 weak: goal not met but an informative side result. 1 negative: fails, reject, kept only as a recorded result. 0 inconclusive or not run._

## Research question
CASS injects DINO's object-structure graph into another model's attention. If we inject DINOv3's graph into MoGe-2's last encoder block, do MoGe-2's mask and geometry respect the spacecraft better? Does the reverse injection, MoGe-2 into DINOv3, help DINOv3's pose features?

## Track
Training-free: closed-form, per image, at inference.

## Pre-registered protocol
Mechanism (from the official code, MICV-yonsei/CASS @ `228ecdd`, `clip/model.py::_apply_spectral_strategy`):
1. **Key graphs.** Target graph A_t = K_t K_tᵀ·s and source graph A_s = K_s K_sᵀ·s, per head, from **last-block attention keys** over patch tokens only.
   - Source keys are bilinearly interpolated to the target token grid and range-normalized to the target keys.
   - s = head_dim^−½.
2. **Head matching.** Per head, the top 20 eigenvalues (eigvalsh, sorted descending) are L2-normalized and then L1-normalized. The cost is 1 − Wasserstein-1 between spectra, and Hungarian matching minimizes it, which pairs **complementary** heads.
3. **Low-rank source graph.** Randomized SVD with the smallest rank reaching 0.95 cumulative singular-value energy (start rank 10, step 5).
   - Singular values are rescaled with Eq. 13, ε = 1.5 fixed (CASS skips this for some datasets; we always apply it).
   - The diagonal is zeroed, and the result is range-normalized to A_t.
4. **Blend.** A = (w·10·Ã_s + A_t)/(w·10 + 1), with w = the matched Wasserstein distance. A Gaussian locality prior (std 5 patches) is added, then softmax, then ×V. This replaces the target's patch-to-patch attention.
5. **Deviations from CASS, recorded:**
   - the block's residual and MLP are **kept**, because MoGe-2's heads and DINOv3's later use expect the block intact;
   - the CLS and register tokens use the original attention;
   - DINOv3 keys are taken **before** RoPE (content similarity).

The two arms:
- **18a, DINOv3 → MoGe-2.** Target: MoGe-2 DINOv2-L block 23 (37×37). Source: DINOv3-L block 23 keys (16×16 → 37×37). MoGe-2 is re-extracted on subset v1.
- **18b, MoGe-2 → DINOv3.** Target: DINOv3-L block 23 (16×16). Source: MoGe-2 block 23 keys (37×37 → 16×16). DINOv3-L is re-extracted on subset v1.

Evaluation:
- **Masks:** EXP-015 A1 metrics (leakage, recall) for the injected MoGe-2 mask and the injected-DINO PCA foreground, each against its own original, per domain, with paired bootstrap CIs.
- **Pose:** the probes of EXP-010/013, 3 seeds, compared with the same-seed originals:
  - 18a: MoGe-2 `grid4` and `normals16`;
  - 18b: DINOv3-L `grid4`.
- **Claim:**
  - A direction helps the **mask** if leakage drops (CI excludes 0) on lightbox **and** sunlamp with recall ≥ original − 0.05.
  - It helps **pose** if lightbox **and** sunlamp mean error both drop by more than the larger seed std, with synthetic val no worse by more than 1 std.
- The same extraction pass also saves per-image, per-head eigenvalue spectra of both models' last-block key graphs, for EXP-020.

## Runs
- **Extraction:** a single guarded pass (`scripts/exp018_cass_extract.py`, 773 s, batch 4, peak GPU 4.1 GB). It wrote the caches `moge2_vitl_cassdino`, `dinov3_vitl16_cassmoge` and `cass_spectra`.
- **Mask evaluation:** `RUN-20261009-000716-seed0`.
- **Probes:** 9 runs (3 feature sets × 3 seeds), all completed. See `run_list.csv`.
- **Implementation check:** `tests/test_cass.py` verifies that the vectorized fusion equals a per-head float64 reference (same head pairs, attention within 2e-4).

## Results
**Masks** (EXP-015 A1 metrics; paired 95% CI against the original model):

| domain | MoGe-2 leakage: orig → injected | MoGe-2 recall | DINO-PCA leakage: orig → injected | DINO-PCA recall |
|---|---|---|---|---|
| synthetic_val | 0.279 → 0.282 (+0.003 [+0.001, +0.005]) | 0.989 → 0.963 | 0.116 → 0.094 (−0.022 [−0.026, −0.017]) | 0.962 → 0.970 |
| lightbox | 0.523 → 0.524 (+0.001 [−0.003, +0.004]) | 0.997 → 0.992 | 0.139 → 0.129 (−0.010 [−0.015, −0.005]) | 0.980 → 0.970 |
| sunlamp | 0.565 → 0.567 (+0.002 [+0.000, +0.004]) | 1.000 → 1.000 | 0.121 → 0.123 (+0.002 [−0.003, +0.008]) | 0.994 → 0.993 |

**Pose** (mean ± std over 3 seeds, greedy; controls are the same-seed originals):

| features | synthetic_val mean | lightbox mean | sunlamp mean |
|---|---|---|---|
| MoGe-2 `grid4` | **39.6 ± 0.6** | **83.3 ± 0.9** | **83.9 ± 1.0** |
| MoGe-2 + DINOv3 graph (18a) `grid4` | 45.6 ± 1.0 | 86.8 ± 0.8 | 91.5 ± 1.2 |
| MoGe-2 `normals16` | **56.4 ± 0.8** | **84.0 ± 2.1** | **100.1 ± 0.2** |
| MoGe-2 + DINOv3 graph (18a) `normals16` | 62.4 ± 0.5 | 93.2 ± 2.0 | 105.3 ± 0.2 |
| DINOv3-L `grid4` | **27.7 ± 0.8** | **76.1 ± 1.0** | **79.1 ± 0.6** |
| DINOv3-L + MoGe-2 graph (18b) `grid4` | 34.2 ± 1.0 | 77.1 ± 0.4 | 81.0 ± 3.5 |

## Pre-registered verdict
- **18a, DINOv3 → MoGe-2.**
  - Mask: **NO.** Leakage is unchanged on lightbox (CI includes 0) and slightly worse on sunlamp.
  - Pose: **NO.** It is worse on every domain, by 3.5–9.2°.
- **18b, MoGe-2 → DINOv3.**
  - Mask: **NO.** Leakage improves on lightbox (−0.010) but not on sunlamp (CI includes 0). The rule needs both.
  - Pose: **NO.** It is worse on every domain, including synthetic by 6.5°, which fails the guard.

## Interpretation
- **Rewriting a backbone's last-block attention breaks the components downstream of it.** CASS edits CLIP's *final* block, whose output goes straight to a patch-text similarity. Here the edited block feeds MoGe-2's neck and heads, and the probe, all of which expect the original attention statistics. The edit therefore acts as a distribution shift, not a gain.
- **Copying the other model's graph doesn't remove background structure.** MoGe-2's mask leakage is a decision made by its mask head on background texture. Re-weighting patch-to-patch attention in one block doesn't change it.
- **The output-level fusion of EXP-015/016 (AND mask) works where this attention-level fusion doesn't.**

## Decision
Reject attention injection for this pipeline. The extracted spectra are kept for EXP-020.
