# EXP-018: CASS-style spectral attention injection, DINOv3 → MoGe-2 and MoGe-2 → DINOv3 (CASS option 2)

## Status
Running. Pre-registered on 2026-10-08, before any result. User-requested; option 2 of 4.

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

## Results
(to be filled)
