# EXP-104: Audit of the ~180° failures (user-requested side experiment)

## Status
Completed (2026-10-10). Analysis only: no training, no selection. It uses the frozen EXP-036 seed-0 greedy predictions and EXP-101 flat-head predictions.

## Keep score
**4 / 5: Support.** It rules out bugs and points to a visual-ambiguity (feature-level) cause.

## Research question
Do the ~180° errors come from a bug (encoder/decoder, metric, rotation convention, augmentation), from the Tesseract chart seams, from hierarchical routing, or from visual near-symmetry of Tango?

## Results
Sources:
- `outputs/experiments/EXP-104_flip_audit/RUN-20261010-024303-seed0/metrics/metrics.csv`, `decoder_checks.json`
- `tables/per_image_audit.csv`
- figures `error_hist_and_flip_axes.png`, `chart_margin_and_divergence.png`

| Candidate cause | Test | Result | Verdict |
|---|---|---|---|
| Encoder/decoder bug | GT path → production decoders (`tesseract.decode` and the GPU `decode_torch` used at inference), all 69,491 GT rotations | max 5.78°, mean 2.51° (both decoders identical); manifest paths = encoder 100%; path(q) = path(−q) | **ruled out** |
| Metric bug | quaternion vs matrix geodesic; error(−q_pred) | max difference 0.000° in every domain | **ruled out** |
| Convention (active/passive) | error of R_pᵀ vs R_g on > 150° failures | 0% of failures fixed (< 20°) | **ruled out** (the green GT boxes in the EXP-036 figures also sit on the spacecraft) |
| Augmentation with unchanged labels | code inspection | features come from fixed GT crops; there is no flip, rotation or augmentation anywhere | **ruled out** |
| Chart seam | GT chart margin \|q\|₍₁₎ − \|q\|₍₂₎ of failures vs all images | lightbox failures median 0.31 vs all 0.26; fraction with margin < 0.05: 0.10 vs 0.11 (sunlamp 0.17 vs 0.11) | **not the main cause** (at most a weak effect on sunlamp) |
| Hierarchical routing | first divergent level; flat-classifier control | 94–99% of failures diverge at the root, **but** a ~180° rotation almost always changes the dominant quaternion component, so this is a consequence, not evidence. **Flat** Hopf heads on the same features (EXP-101, subset) flip as much or more: P(>150° \| >90°) = 0.42–0.58, vs 0.35–0.40 for the tree on the same subset | **not the cause** |
| Visual near-symmetry | body-frame dR = R_gᵀ R_p; error after allowing 180° about body x/y/z | Failures concentrate near 180° beyond chance: P(165–180° \| >90°) = 0.36–0.41, vs **0.20 for uniformly random rotations**. Their body-frame axes cluster (body z, body y, and two diagonals in the x–y plane). Allowing 180° about one body axis puts 29–40% of > 150° failures within 20° and cuts the lightbox mean 38.9° → 23.5° (DINOv3) | **main cause** |

**Note on the "pile-up at 180°":** for uniformly random wrong rotations, about 20% of errors above 90° would fall in 165–180°, because SO(3) has more volume at large angles (Haar density ∝ 1 − cos θ). The observed ~37–41% is therefore about twice chance. That excess is the genuine flip mode; the rest is the geometry of SO(3).

**Beam evidence (EXP-040 stage 0):** the GT-near mode is usually among the tree's top-8 complete paths (best-of-8 within 20° on lightbox 0.86). The posterior is multimodal, and greedy picks the wrong mode.

## Interpretation
The 180° errors are wrong-mode selections caused by visual near-symmetry of the spacecraft, which is worse under real lighting. They are not caused by the Tesseract mathematics, the chart seams, the tree, or a convention or metric bug. The synthetic-val rate is tiny for DINOv3 (0.1%) but 3% for MoGe-2 normals, which carry less appearance information.

## Decision
Keep the representation. Address it with mode *selection*: a beam re-ranking verifier (the proposed EXP-103, awaiting approval), the Phase-4 posterior (EXP-042), or features that disambiguate. Symmetry-corrected error is a diagnostic only; the official metric is unchanged.
