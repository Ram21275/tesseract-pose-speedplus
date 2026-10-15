# EXP-045: Phase-4 matched comparison and decision

## Status
**Completed 2026-10-15** (collation; no new training or evaluation). All inputs are finished, pre-registered runs (EXP-036, 040–044, 105), so their results were known when this table was written. The gate below is the execution plan's own rule (Phase 4 "Gate"), applied as written. Score: **4/5 Support** for keeping the coupled posterior, used as a verifier.

## Question (plan, Phase 4)
Which continuous-rotation design to keep: tangent residual, separate cell-conditioned RFM, the image-conditioned SO(3) posterior with flow-induced Tesseract masses, or that posterior distilled into the tree?

## Matched setting
Same frozen features (DINOv3 `tokens8`, MoGe-2 `normals16~and`), same EXP-036 PoE trees, 3 seeds, synthetic train only, selection on synthetic val; lightbox / sunlamp descriptive (DEC-006). Flow systems use NFE 10, M = 32, velocity average across branches.

## Comparison (mean error °, 3-seed mean ± std; real domains descriptive)
| system | synth val | synth median | lightbox | sunlamp | GT-cell recall / modes | uncertainty (coverage @0.9) | parent–child consistency | NFE × samples | latency (ms / img, batched) |
|---|---|---|---|---|---|---|---|---|---|
| Tree only (EXP-036 PoE greedy, reference) | 5.540 ± 0.063 | 4.62 | 26.9 | 36.6 | top-8 leaf recall 0.69 (synth) / 0.28 (lightbox) | none | exact (one tree) | 1 pass | ≈0.4 |
| **Tangent residual** (EXP-040) | 5.414 ± 0.043 | 4.49 | 26.9 | 36.6 | as tree | none | n/a | 1 pass | 0.33 |
| **Separate RFM**, local base 3° (EXP-041) | 5.244 ± 0.077 | 4.39 | 27.1 | 36.7 | best-of-32 2.64 synth | 0.80 (over-confident) | n/a | 10 × 32 | 19.4 |
| **Coupled posterior**, Haar base, KDE mode (EXP-042) | 5.219 ± 0.149 | 3.75 | 29.9 | 37.6 | best-of-32: **9.7 lightbox**, 10.6 sunlamp | **0.98** (over-dispersed) | **exact (0.0)** by construction (EXP-043) | 10 × 32 | 12.7 |
| Flow-induced masses, greedy descent (EXP-043) | 6.210 ± 0.127 | 4.58 | 31.5 | 39.7 | flow top-1 > tree at L2–5 (synth); < tree on real | from masses | exact | 10 × 32 | ≈13 |
| **Distilled tree** (EXP-044, selected β = 0 = tree) | 5.589 (seed 0) | 4.66 | 27.6 | 37.8 | β = 1: top-8 recall 0.77, lightbox 26.2° (descriptive, not selected) | none | tree | 1 pass | ≈0.3 |
| **Posterior as verifier**: tree beams + corners + flips, re-noise / return τ 0.8 (EXP-105 C3) | **4.777 ± 0.106** | 4.11 | 26.5 | **32.7** | oracle of candidates: 7.9 lightbox | via the posterior | n/a | ≈ 2–8 × 16 × 100 candidates | ≈ 100 |

NLL is not reported here: the exact flow log-density and the tree's cell-normalized log-likelihood are measured in EXP-107. Peak memory is also measured in EXP-107 / EXP-063, on the same hardware.

## Gate (plan): keep the coupled posterior only with a measured advantage in accuracy, multimodal recall, uncertainty / coverage or tree consistency that justifies its cost
- **Accuracy:**
  - Used as a verifier over the tree's candidates, the posterior is the most accurate system on synthetic val (4.78 vs 5.54, and vs 5.41 for the residual).
  - It is also the most accurate on sunlamp (32.7 vs 36.6).
  - As a stand-alone KDE point estimate it is no better than the RFM, and worse on real.
- **Multimodal recall:** best-of-32 reaches 9.7° on lightbox (vs 16.9° for the tree's top-8 oracle). The flip mode is represented; it is just not selected (EXP-105).
- **Coverage:** 0.98 at the 0.9 level, vs 0.80 for the RFM. Over-dispersed (conservative), but the only system that does not under-cover.
- **Consistency:** exact parent–child masses by construction.
- **Cost:** 12.7 ms per image (KDE) to about 100 ms (verifier), vs 0.3–0.4 ms for the tree.
- Distillation (EXP-044) did not transfer the advantage into the tree at tree cost.

**Decision: PASS, keep the coupled posterior (EXP-042), in the role of verifier / re-ranker over the tree's beam candidates (EXP-105 C3 + re-noise τ 0.8).**
- The fast fallback is the tree alone or the tree plus tangent residual (sub-millisecond).
- The separate RFM is dropped: dominated by the coupled posterior on coverage, and not better in accuracy.
- Distillation is not used.
- Final adoption, including EXP-105 arm K, happens at the recipe lock (DEC-012), which the user decides.

## Limitations
- One dataset, ground-truth crops, rotation only.
- Latencies from different runs on a shared GPU are indicative only. EXP-107 re-times every component under one protocol.
- The verifier's candidate set needs 100 posterior evaluations per image. A cheaper candidate set (C0 + re-noise: 4.99°) is reported in EXP-105.
