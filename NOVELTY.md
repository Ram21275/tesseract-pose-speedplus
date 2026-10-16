# NOVELTY.md — contributions, best system, and claim boundaries

*Living document. It is updated whenever an experiment result lands. Every number below was read from the run artifacts (`outputs/experiments/*/RUN-*/metrics|tables`), not from summaries.*

| field | value |
|---|---|
| Reviewed commit | `66cb0dc` (local `main`, 3 commits ahead of `origin`; the push is blocked by the gh account switch) |
| Review date | 2026-10-16 13:00 IST |
| Task / data | SPEED+ spacecraft **rotation only**, **GT square crops** (margin 0.15). Train: synthetic/train only. Selection: synthetic/validation (DEC-006); lightbox / sunlamp are test-only. |
| Error metric | E_R = 2·arccos(\|⟨q̂, q⟩\|) in degrees (identical to SPACE-HOP / the ESA challenge rotation error) |
| Key artifacts | EXP-109 eval `outputs/experiments/EXP-109_real_domain_aug_tokens16_orbit/RUN-20261016-014754-seed0` · EXP-036 `RUN-*` (3 seeds) · EXP-042 / EXP-105 / EXP-107 / EXP-108 / EXP-110 latest RUNs · `outputs/DECISIONS.md` (DEC-001…017) |

---

## 0. Initial best result stored for the paper (frozen reference, 2026-10-16)

**Best system: EXP-109 "C orbit on A".** These are augmentation-trained Tesseract trees on frozen DINOv3 + MoGe-2, fused by per-level PoE, with in-plane orbit PoE voting at test time. 3 seeds, deployable (no oracle).

| | synthetic val | synthetic val-aug (proxy) | lightbox | sunlamp |
|---|---|---|---|---|
| **mean ± std (°)** | **3.87 ± 0.03** | **5.82 ± 0.09** | **12.30 ± 0.46** | **12.28 ± 0.40** |
| median (°) | 3.45 | 3.87 | 5.26 | 6.32 |
| acc@10 | 0.982 | 0.939 | 0.837 | 0.760 |
| > 150° fraction | 0.000 | 0.004 | 0.030 | 0.017 |
| prior frozen system EXP-036 PoE greedy | 5.54 ± 0.07 | 30.84 ± 0.40 | 26.91 ± 0.78 | 36.66 ± 0.93 |

It is the best system on **every** domain, so the recorded selection policy (synthetic val) and the real-domain observations agree for this system. Source: `EXP-109 …/RUN-20261016-014754-seed0/tables/metrics.md`, `metrics/per_seed.csv`.

---

### Recipe lock (DEC-018, user, 2026-10-16)
All further experiments use the **EXP-109 methodology** in two variants, **PoE (DINOv3 + MoGe-2)** and **DINOv3-only**, each with and without orbit voting, so the accuracy / time overhead of MoGe-2 can be shown.
- Diagram: [`outputs/figures/architecture_final_exp109.svg`](outputs/figures/architecture_final_exp109.svg) (also `.png` and `.pdf`).
- **EXP-112** (evaluation only, queued after EXP-111) supplies the missing DINOv3-only + orbit accuracy and the end-to-end timing and memory of all four variants.
- EXP-064 / 106 are amended to this recipe.

### Update 2026-10-17 (EXP-111 / 112 / 114 completed)
- **EXP-112 (measured, batch 1):**
  - PoE + orbit 279 ms (3.6 FPS);
  - **DINOv3-only + orbit 70 ms (14 FPS): 4.21 / 17.40 / 15.27**;
  - DINOv3-only greedy 15 ms (68 FPS).
  - MoGe-2 costs about 41 ms per view (20 ms of it is the CPU mask step).
- **EXP-114:** on synthetic val, weighted voting over the 32 candidates beats the orbit product (3.46 vs 3.87), so the pre-registered "product beats voting" criterion **fails**. The product is best on val-aug (marginally) and on real (descriptive). **Claim narrowed:** the contribution is multi-view orbit consistency over hierarchical candidates (product or weighted voting), not the product score itself. Max-confidence and top-1 voting are clearly worse.
- **EXP-111:** a flow on the augmented DINOv3 tree improves synthetic (4.10) but is worse on real. Not adopted.

## 1. Current evidence and best systems

### 1.1 Experiment status (verified from artifacts)
| status | experiments |
|---|---|
| **Completed** | 000–003, 010–020, 025, 026, 029–036, 040–045, 050, 101, 104, 105 (incl. arm K), 107, 108 (DFM), 109, 110 |
| **Running** | **EXP-111** (DINOv3-only augmented tree + flow; KDE and EXP-045-verifier arms). Seed 0 at epoch 9 of 30 at review time; earlier attempts failed (02:55 oomd kill; 12:49 `--centre` bug), both recorded in its report. |
| **Failed / negative (kept)** | EXP-044 (distillation, 2/5), EXP-108 (path DFM, 1/5), EXP-109 arm B (16×16 tokens), EXP-102 (LoRA, cancelled before running) |
| **Proposed / on hold** | EXP-051–055 (adaptive depth: eligible after EXP-050, waiting for the user), EXP-060–063 (Phase 6 SPEED+), **EXP-064** (T-LESS / SYMSOL-T / YCB-V: data manifests built and verified, no model run), EXP-106 (T-LESS multi-path, on hold) |

### 1.2 Comparison of systems (mean / median °; 3 seeds unless marked; deployable unless marked oracle)
| exp · arm | backbones / features (frozen) | fusion | predictor | refinement / verifier | synth val mean ± std / med | lightbox mean / med | sunlamp mean / med | > 150° (LB / SL) | flip recovery / damage vs EXP-036 (LB; SL) | latency, batch 1 (shared GPU, upper bound) | memory |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **109 C on A** | DINOv3-L tokens8 + MoGe-2 normals16~and | per-level PoE (equal) | tree, aug-trained (A) | orbit PoE voting (4 in-plane views, 32 candidates) | **3.87 ± 0.03 / 3.45** | **12.30 / 5.26** | **12.28 / 6.32** | 0.030 / 0.017 | 62% of 401 / 0.4% ; 80% of 206 / 0.4% | **not measured** (est. 4 × (6 + 20.5) ms backbones + trees ≈ 120–150 ms) | eval peak 13.6 GB GPU (batch 512), 30.8 GB RSS |
| 109 A | same | PoE | tree, aug-trained | – (greedy) | 4.86 ± 0.03 / 4.36 | 17.35 / 6.76 | 20.16 / 8.43 | 0.042 / 0.033 | 51% / 1.0% ; 59% / 1.0% | ≈ 39 ms (6.0 + 20.5 + 12.8, measured components) | training ≤ 31 GB RSS, 1.3–3.6 GB GPU |
| 109 A, DINOv3 branch only (training-run eval) | DINOv3 tokens8 | – | tree, aug-trained | – | 4.84 (3-seed avg) | 22.5 | 22.0 | not measured | not measured | ≈ 6.0 + ~6.5 ms (est.) | – |
| 109 C on EXP-036 | DINOv3 + MoGe-2 | PoE | tree, clean (EXP-036) | orbit voting | 4.62 ± 0.04 / 3.91 | 20.25 / 7.26 | 24.83 / 9.96 | 0.048 / 0.043 | 36% / 0.8% ; 46% / 0.9% | not measured | – |
| 109 B | DINOv3 tokens16 + MoGe-2 | PoE | tree | – | 5.56 ± 0.03 / 4.63 | 27.01 / 9.15 | 35.97 / 12.95 | 0.057 / 0.073 | 14% / 1.1% ; 13% / 1.2% | not measured (4× tokens; 2 h training per seed) | memmap streaming |
| **036** (prior frozen) | DINOv3 + MoGe-2 | PoE | tree, clean | – | 5.54 ± 0.07 / 4.62 | 26.91 / 8.99 | 36.66 / 12.98 | 0.059 / 0.074 | – | ≈ 39 ms; trees 0.22 ms/img batched | 4.5 M trainable params |
| 105 C3 + re-noise τ 0.8 (= EXP-045 verifier) | DINOv3 + MoGe-2 | PoE | tree, clean | flow re-noise / return over 96 candidates | 4.78 ± 0.11 / 4.11 | 26.5 / – | 32.7 / – | about 0.10 / 0.13 (> 90°) | not measured in this form | ≈ 100 ms (scoring) + backbones | – |
| 042 KDE | DINOv3 + MoGe-2 | velocity average | flow (Haar base) on clean trees | KDE mode, M 32 | 5.22 ± 0.15 / 3.75 | 29.85 / – | 37.61 / – | – | – | 12.7–15 ms flow | – |
| 042 best-of-32 (**oracle**) | | | | | 2.60 | 9.7 | 10.6 | – | – | – | – |
| 105 C3 oracle (**oracle**) | | | | | 2.25 | 7.9 | 11.4 | – | – | – | – |
| 108 DFM (NFE 4, M 8) | DINOv3 + MoGe-2 | per-token PoE | discrete path flow | KDE | 7.22 ± 0.12 | 29.80 | 39.84 | 0.062 / 0.073 | 4.9% / 0.5% ; 4.0% / 0.4% | not measured | – |
| 105 arm K (Kurz corners) | DINOv3 + MoGe-2 | PoE | 8-facet corner tree | corner vote | 5.20 ± 0.09 / 4.43 | 28.8 / 10.3 | 39.1 / 16.2 | – | – | – | – |
| 110 w = 0.8 | DINOv3 + MoGe-2 | weighted PoE | tree, clean | – | 5.14 | 32.04 | 36.74 | – | – | – | – |
| 044 β = 1 (seed 0 only) | DINOv3 + MoGe-2 | PoE | tree distilled from flow | – | 5.62 (1 seed) | 26.17 | 37.67 | – | – | – | – |

Timing source: EXP-107 `timing` table, RTX-class shared GPU (concurrent load recorded). DINOv3-L 6.0 ms (166 FPS), MoGe-2 20.5 ms (49 FPS), both trees 12.8 ms at batch 1 (0.22 ms/img batched), flow NFE 10 × M 32 15 ms. **The end-to-end latency of the best system (C on A) has not been measured.**

### 1.3 The four "best" notions
1. **Best synthetic-validation system:** 109 **C on A** (3.87°). It also wins on synthetic val-aug (5.82°).
2. **Best observed real-domain results:** 109 **C on A** (lightbox 12.30°, sunlamp 12.28°). Oracle best-of-M numbers (EXP-042 9.7°, EXP-105 7.9°) are **not** deployable and are reported only as headroom.
3. **Best accuracy–compute trade-off (measured components):** 109 **A** (single pass, ≈ 39 ms): 4.86 / 17.35 / 20.16. The fastest option is the **DINOv3-only A tree** (≈ 12–13 ms est.): 4.84 / 22.5 / 22.0. Its exact timing is pending in EXP-111.
4. **Selected under the recorded validation policy (DEC-006):** 109 **C on A**. It is best on synthetic val and on the pre-registered val-aug proxy, so no real-domain number was needed to select it.

**Policy conflict recorded:** EXP-110. The val-aug proxy picks w_DINO = 0.8, but lightbox is 5.1° worse there than at w = 0.5. The weight is therefore **not adopted**, and real-domain data was **not** used to choose an alternative. There is no conflict for C on A.

---

## 2. Contribution inventory

| # | candidate contribution | type | evidence (measured effect) | closest prior work | difference from prior work | limitations / confounds / missing controls | status | role |
|---|---|---|---|---|---|---|---|---|
| a | **Tesseract hierarchical rotation prediction on frozen foundation features**: autoregressive Transformer over a 4-chart recursive tesseract tree (depth 5, 131k leaves, cell centres, unique paths) | architectural / system | EXP-030–036: synth 5.54°; EXP-101: tree head 1.2 ms vs flat 3.3 ms at 131k hypotheses, 9 M vs 76 M params, more accurate with DINOv3 (29.2° vs 50.7°, subset v1) | Kurz et al. 2017 (tesseract subdivision); Implicit-PDF / HEALPix grids; SpyroPose (hierarchical SE(3) pyramids); Hopf grids (Yershova 2010; SPACE-HOP) | Uses **cell-centre** representatives with unique paths and a learned autoregressive Transformer decoder over the path, on frozen features. Kurz is a discretization; SpyroPose is a coarse-to-fine importance-sampling pyramid over SE(3) | A tree-vs-flat-Hopf comparison under the *final* recipe (aug + orbit) is missing; EXP-101 was on subset v1 with the old features. Only one object | supported (for the system), modest novelty | supporting / main-system component |
| b | **DINOv3 (appearance) × MoGe-2 (geometry) complementarity with training-free per-level PoE** | empirical + architectural | EXP-036: PoE real lightbox 26.9 vs DINOv3 37.1 / MoGe-2 43.7; EXP-029: learned concat fusion = DINO-only on synth and much worse on real (40.8 vs 27.9) | PoE (Hinton 2002); late fusion; multi-branch pose | Applies the PoE **per tree level**, renormalized, with zero trained fusion parameters, and shows that learned fusion over-fits the synthetic domain | With augmentation (A), MoGe-2 adds little on its own (single-branch MoGe-aug 40.6° lightbox). PoE-A vs DINO-A-only on real: 17.35 vs 22.5 lightbox, 20.16 vs 22.0 sunlamp (PoE still better). EXP-110 shows the weight matters, with the proxy and real data disagreeing | supported | supporting result |
| c | **Continuous SO(3) posterior on the tree's memory (EXP-042), flow-induced tree masses (043), distillation (044)** | methodological (application) | 042: synth 5.22° KDE; GT log-likelihood flow 3.85 vs tree 0.98 nats (lightbox); best-of-32 9.7°. 043: masses consistent; flow-mass decoding worse (6.21°). 044: distillation **fails** (β = 0 best) | Riemannian flow matching (Chen & Lipman, ICLR 2024); LiePose (Hsiao et al., CVPR 2024); Implicit-PDF | Conditions an SO(3) flow on a frozen hierarchical predictor's encoder memory, and maps samples to tree masses | Parent–child consistency of counted masses holds **by construction** (not a finding). Coverage is over-dispersed (0.98 at the 0.9 level): **not calibrated**. The velocity average across branches is a heuristic, not an exact PoE | supported (042), unsupported (044) | appendix / supporting |
| d | **Flow-based candidate verification (EXP-045 / 105)**: tree beams + 180° flips + cell corners, re-ranked by re-noise / return consistency | methodological (modest) | synth 4.78 vs 5.54; sunlamp 32.7 vs 36.6; lightbox ≈ unchanged; flips **not** repaired on real (> 90° ≈ 10%) | SDEdit-style re-noising; Implicit-PDF scoring; render-and-compare verifiers | A return-consistency score from an SO(3) flow over a hierarchical candidate set | The return score is **not a normalized likelihood**. Adding flip candidates did **not** repair real flips (the oracle shows they are present: 9.8°). Built on clean trees; on A it is pending (EXP-111) | supported (synthetic), weak on real | ablation / appendix |
| e | **Diagnosis: real-domain error is dominated by 180° body-axis mode selection** | empirical finding | EXP-104: decoders exact, failures at 180° occur at 2× chance with clustered body axes, and a 180° allowance cuts lightbox 38.9 → 23.5. EXP-105: the correct flip is in the candidate set (oracle 9.8° lightbox), but tree, flow and templates all pick the wrong one. EXP-109: aug + orbit removes 62–80% of severe flips | SPACE-HOP Fig. 4 (W-shaped error histogram); symmetry-ambiguity literature (LiePose, SpyroPose) | Isolates **selection** (the mode is present but not chosen) from **representation**, quantifies it, and shows which intervention fixes it | Single object; GT crops; frozen features | supported | **main** (with f) |
| f | **Geometry-correct in-plane rotation augmentation + orbit-consistent hierarchical inference (EXP-109 A + C)** | methodological (modest) + system + empirical | A + C: synth 3.87, lightbox 12.30, sunlamp 12.28 (−14.6° / −24.4° vs EXP-036); recovery 62% / 80% of > 150°, damage 0.4%. C alone on clean trees: 20.25 / 24.83 | Test-time augmentation / rotation voting; SPNv2 heavy augmentation (Park & D'Amico 2022); PoE | Labels rotated by a **derived** transform (q′ = Exp(−θ·d) ⊗ q, d = undistorted ray through the crop centre; verified 0.2° p95 by PnP), and the orbit is scored as a **product of hierarchical tree likelihoods** over 32 back-mapped candidates rather than by averaging point estimates | Augmentation and TTA are **standard**: the novelty is in the combination and the exact label transform, not the idea. **Missing control:** photometric-only augmentation, to separate the rotation augmentation's share. Latency not measured. Orbit size (4) not ablated | supported (empirical), modest method novelty | **main** (system + finding) |
| g | **Direct discrete path flow (EXP-108)** | methodological | **fails**: 7.22 vs 5.54 synth; worse on real | Discrete flow matching (Gat et al., NeurIPS 2024; Campbell et al., ICML 2024); Flow6D (discrete-to-continuous FM, arXiv 2606.23293) | Tree-conditioned source coupling on a nested Tesseract path | Token mixing is **not geometry-preserving** (45–53% of mid-t paths are > 30° from both endpoints, measured); no no-geometry control (dropped by the user), so the geometry token's effect is unidentifiable; the cross-branch per-token PoE is heuristic | unsupported | appendix (negative result) |
| h | **16×16 tokens (EXP-109 B)** | ablation | no gain (30.51 vs 30.84 val-aug; 5.56 vs 5.54) at 4× tokens | – | – | – | unsupported | ablation |
| i | **Weighted PoE (EXP-110)** | ablation | proxy gain, real disagreement | – | – | Done on clean trees, not on the final A trees | preliminary | ablation |
| j | **DINOv3-only augmented tree + flow / verifier (EXP-111)** | system | **running**: DINOv3-A tree alone 4.84 / 22.5 / 22.0 (training-run eval) | EXP-042 / 045 | Fast variant without MoGe-2 | Pending; must be compared with its **own** tree (A-DINOv3), not with EXP-036 | proposed | pending |
| k | **Cross-object validation (EXP-064 / 106)** | empirical | **none yet**: manifests only (T-LESS 100k, SYMSOL-T 250k, YCB-V 54.6k crops; conventions verified) | LiePose (SYMSOL-T, T-LESS), SpyroPose, Implicit-PDF | Same frozen backbones, retrained heads | Not run; heads are **retrained per object**, so this is not unseen-object prediction | proposed | future work / pending |

**Methodological novelty vs practical value vs evidence (qualitative):**
- **(f)** has the highest practical value and strong evidence, with modest methodological novelty.
- **(e)** is a clear, well-evidenced empirical finding.
- **(a), (b)** are a solid system with modest novelty.
- **(c), (d)** have moderate novelty, mixed value and evidence.
- **(g)** is novel in framing, but negative.

---

## 3. Prior-art checks and claim boundaries

**Verified primary sources:**
- Kurz, Pfaff, Hanebeck, *Discretization of SO(3) Using Recursive Tesseract Subdivision*, IEEE MFI 2017, doi:10.1109/MFI.2017.8170406 — [PDF](https://isas.iar.kit.edu/pdf/MFI17_Kurz-Tesseract.pdf). The recursive tesseract and its corner formulation pre-exist; ours is a cell-centre, unique-path variant.
- Haugaard, Hagelskjær, Iversen, *SpyroPose: SE(3) Pyramids for Object Pose Distribution Estimation*, ICCV Workshops 2023 — [CVF](https://openaccess.thecvf.com/content/ICCV2023W/R6D/html/Haugaard_SpyroPose_SE3_Pyramids_for_Object_Pose_Distribution_Estimation_ICCVW_2023_paper.html), [arXiv 2303.05308](https://arxiv.org/abs/2303.05308). Hierarchical pose distributions pre-exist.
- Murphy et al., *Implicit-PDF*, ICML 2021 — [arXiv 2106.05965](https://arxiv.org/abs/2106.05965). Grid-based SO(3) densities; SYMSOL.
- Yershova, Jain, LaValle, Mitchell, *Generating Uniform Incremental Grids on SO(3) Using the Hopf Fibration*, IJRR 29(7) 2010, doi:10.1177/0278364909352700 — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2896220).
- Chen & Lipman, *Flow Matching on General Geometries* (Riemannian FM), ICLR 2024 — [proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d1f9936d3be6997ffffab692977eebe6-Abstract-Conference.html).
- Hsiao et al., *Confronting Ambiguity in 6D Object Pose Estimation via Score-Based Diffusion on SE(3)* (LiePose), CVPR 2024 — [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Hsiao_Confronting_Ambiguity_in_6D_Object_Pose_Estimation_via_Score-Based_Diffusion_CVPR_2024_paper.html).
- Gat et al., *Discrete Flow Matching*, NeurIPS 2024 — [proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f0d629a734b56a642701bba7bc8bb3ed-Abstract.html); Campbell et al., *Generative Flows on Discrete State-Spaces*, ICML 2024 — [arXiv 2402.04997](https://arxiv.org/abs/2402.04997).
- *Flow6D: Discrete-to-Continuous Flow Matching for … Category-Level 6D Pose Estimation* — [arXiv 2606.23293](https://arxiv.org/abs/2606.23293), [project](https://flow6d.github.io/). Discrete pose-bin flow followed by continuous residual flow already exists (category level, point clouds; benchmark numbers not verified by us).
- Park & D'Amico, *Robust multi-task learning and online refinement for spacecraft pose estimation across domain gap* (SPNv2), Advances in Space Research 2022 — [arXiv 2203.04275](https://arxiv.org/abs/2203.04275). Heavy augmentation (incl. style) against the SPEED+ domain gap pre-exists.
- Dutta et al., *SPACE-HOP*, CVPR Workshops (AI4Space) 2026 — [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026W/AI4Space/papers/Dutta_SPACE-HOP_Spacecraft_6-DoF_Pose_Estimation_using_Embedding-Predictive_Pretraining_and_Hopf_CVPRW_2026_paper.pdf). Hopf-grid classification + offsets; reports the same W-shaped flip histogram.
- Backbones: DINOv3 ([arXiv 2508.10104](https://arxiv.org/abs/2508.10104)), MoGe-2 (NeurIPS 2025, [arXiv 2507.02546](https://arxiv.org/abs/2507.02546)).

**Where our distinction can lie:**
- Not in "a hierarchical SO(3) grid", "flow matching for pose", "discrete FM", "augmentation", "TTA voting" or "PoE": all exist.
- It lies in:
  1. the **demonstrated finding** that real-domain error on SPEED+ is a *mode-selection* problem (the mode is present; scorers on clean features cannot choose it);
  2. the **combination** that fixes it, with a geometry-derived in-plane label transform and an orbit **product of hierarchical likelihoods**, on frozen backbones with about 4.5 M trainable parameters.
- Tree-conditioned initialization (EXP-108) and return-consistency scoring (EXP-045) are distinct in detail, but neither is a winning component. **No "first" claims.**

**Claims checked against the implementation (do not make):**
- EXP-108 "uses geometry to repair paths": its token interpolation is **not geometry-preserving**, there is **no no-geometry control**, and it did not help.
- "Parent–child consistency of flow masses" is a finding: it follows **by construction** (EXP-043).
- "Return consistency is a likelihood": it is not normalized. Only the CNF log-density (EXP-107) is a likelihood.
- "The flow is calibrated": coverage is over-dispersed (0.98 at 0.9) and the RFM under-covers (0.80).
- "Flip candidates repair real flips": EXP-105 shows they do not. Only aug + orbit (EXP-109) does, and only 62–80%.
- "Training-free" / "zero-shot": the backbones are frozen, but the trees are **trained** (≈ 4.5 M params). Cross-object heads are **retrained** per object.
- Comparisons with SPACE-HOP / SPNv2 numbers are **unmatched**: rotation only, GT crops, frozen backbones.

**Negative findings kept because they bound the claims:** EXP-044, EXP-108, EXP-109 B, EXP-029 (learned fusion), EXP-105 templates, EXP-105 arm K on real, and EXP-110's proxy–real disagreement.

---

## 4. Paper contribution shortlist

**Main contributions (completed evidence):**
1. **A lightweight hierarchical rotation predictor on frozen foundation features.** A Tesseract-path Transformer per frozen backbone (DINOv3 appearance, MoGe-2 geometry) is fused per level by a training-free product of experts. About 4.5 M trainable parameters and about 1 GPU-hour of training. Fusion improves real-domain accuracy where learned fusion over-fits synthetic data. *(EXP-030–036, 029, 101.)*
2. **Real-domain failures are a mode-selection problem.** On SPEED+, most large real-domain errors are 180° body-axis flips whose correct mode is already among the predictor's hypotheses (oracle 9.8° lightbox). Continuous posteriors, flow-based verifiers, synthetic templates and discrete path flows do not select it. *(EXP-104, 105, 042/045, 108.)*
3. **Orbit-consistent inference with geometry-correct rotation augmentation resolves most flips.** Training the trees with in-plane rotations whose labels follow a derived camera-ray transform, and scoring candidates by the product of tree likelihoods over a 4-view in-plane orbit, reduces error from 26.9 / 36.7° to **12.3 / 12.3°** (lightbox / sunlamp) and 5.54 → 3.87° (synthetic). It corrects 62–80% of severe flips while breaking 0.4% of good predictions. *(EXP-109.)* Augmentation and test-time voting are standard; the contribution is the derived transform, the hierarchical-likelihood orbit scoring, and the demonstrated effect on the diagnosed failure mode.

**Supporting findings:**
- The flow posterior has a much higher GT log-likelihood on real images than the tree (3.85 vs 0.98 nats, lightbox), despite a worse point estimate (EXP-107).
- Speed–accuracy: NFE 10–20 is enough; more samples help (EXP-107).
- Finer tokens do not help (EXP-109 B).
- Learned fusion and distillation do not help (EXP-029, 044).

**Proposed, awaiting results:**
- DINOv3-only fast variant ± flow / verifier (EXP-111).
- Timing of the final system.
- Cross-object validation (EXP-064).

**Do not combine experiments never evaluated together:**
- e.g. "A + C + verifier" or "A + C with w = 0.8" have **not** been run.

---

## 5. Final architecture decision

| candidate | synth / LB / SL (mean °) | compute | verdict |
|---|---|---|---|
| **109 C on A** | **3.87 / 12.30 / 12.28** | ≈ 4× backbone passes (not measured, est. 120–150 ms batch 1) | **recommended** |
| 109 A (single pass) | 4.86 / 17.35 / 20.16 | ≈ 39 ms | fast fallback |
| DINOv3-only A tree | 4.84 / 22.5 / 22.0 | ≈ 12–13 ms (est.) | fastest fallback (EXP-111 to confirm) |
| EXP-045 verifier (clean trees) | 4.78 / 26.5 / 32.7 | ≈ +100 ms | superseded by C on A |
| EXP-108 DFM | 7.22 / 29.8 / 39.8 | – | rejected |
| EXP-111 (A-DINO + flow / verifier) | pending | – | compare against **A-DINO tree** (4.84 / 22.5 / 22.0), not EXP-036 |

- **Recommendation:** **EXP-109 C on A**, adopted by the user as the locked recipe (DEC-018), together with its DINOv3-only variant. It is best on every domain, selected under the recorded synthetic-validation policy, and needs no flow or extra training beyond the augmented trees.
- **Limitations:**
  - 4× inference cost (unmeasured);
  - rotation only; GT crops; one object;
  - the photometric vs rotation-augmentation shares are not separated.
- **Locked by the user (DEC-018):** PoE and DINOv3-only variants, each ± orbit.

---

## 6. Minimum work to finalize the paper (prioritized)

**Essential (pre-registered 2026-10-16 as DEC-019 / EXP-112–115; EXP-064 next):**
1. **Fair timing and memory of the final system** (being done in **EXP-112**: C on A; A; DINOv3-only A; DINOv3-only + orbit) on one GPU, batch 1 and 64, synchronized. *Validates the cost claims in C1 and C3 and the trade-off table.*
2. **Augmentation ablation (EXP-115):** photometric-only vs photometric + rotation (A) vs rotation-only, each ± orbit. *Validates C3 (that the derived rotation transform, not augmentation in general, drives flip repair).*
3. **Matched flat baseline under the final recipe (EXP-113)** and **orbit scoring rules (EXP-114: product vs max-confidence vs cluster voting, same views and candidates)**: a flat Hopf-grid classifier (SPACE-HOP-style anchors + offset) on the same frozen features with the same augmentation ± orbit. *Validates C1 (that the tree matters) and gives the closest external method a matched comparison.*
4. **Cross-object validation (EXP-064, at least SYMSOL-T + T-LESS; amended to the locked recipe)** with the locked recipe. *Validates the generality of C1/C3; without it the paper stays SPEED+-only.*
5. **Figures and tables:**
   - the architecture diagram, updated to A + C;
   - the error histogram before/after (W-shape removal);
   - the flip-transition table with denominators;
   - SO(3) distribution plots for the final system;
   - the main comparison table above.
   *Validates C2/C3 visually.*

**Optional:**
- EXP-111 (fast variant); the orbit size ablation (2/4/8 views); EXP-061 corruptions; detected crops instead of GT; the weighted PoE on A trees; EXP-051–055.

---

## Conclusion
- **Strongest story supported today:** a lightweight Tesseract tree on frozen DINOv3 + MoGe-2 features fails on real images mainly through 180° mode selection. Geometry-correct rotation augmentation plus orbit-consistent hierarchical inference fixes most of it: 12.3° on both real domains, from 26.9° / 36.7°.
- **Uncertain:**
  - the share of rotation vs photometric augmentation;
  - whether the tree beats a matched flat grid under the final recipe;
  - the real end-to-end latency;
  - behaviour on other objects.
- **Could change the story:**
  - if a matched flat Hopf classifier with the same augmentation + orbit matches C on A, contribution 1 weakens to "any discretization works";
  - if EXP-064 fails to transfer, the paper becomes SPEED+-specific;
  - if EXP-111's DINOv3-only variant comes close, the recommended system may simplify.
- **Minimum evidence before freezing:** essential items 1–3, plus at least one cross-object dataset (item 4).

## Changelog
- 2026-10-16: created. Initial best result stored (EXP-109 C on A). EXP-111 running.
- 2026-10-17: EXP-111 / 112 / 114 results; contribution 3 narrowed (orbit consistency, scoring rule open); timing measured.
- 2026-10-16: DEC-019: EXP-113 (tree vs matched flat Hopf), EXP-114 (orbit scoring rules), EXP-115 (augmentation ablation) pre-registered and queued; EXP-064 next.
- 2026-10-16: DEC-018 recipe lock (EXP-109 methodology, PoE + DINOv3-only variants); final architecture diagram added; EXP-112 pre-registered (DINOv3-only + orbit, end-to-end timing).
