# Phase-Wise Execution Plan: DINOv3–VGGT Fusion and Hierarchical Tesseract Pose Prediction

## 1. Objective

Build the method in result-driven phases. Each phase contains only the experiments needed to answer its immediate research question. Complete its report and decision gate before moving forward. Diverge into optional methods only when the recorded results justify doing so.

The first milestone ends with a validated rotation predictor that compares fixed-depth classification, tangent residuals, Riemannian flow matching (RFM), a flow-consistent Tesseract posterior, and adaptive depth. Translation, temporal reasoning, and continual learning come later.

---

## 2. Dataset rule

The SPEED+ dataset remains unchanged.

- Read original images and annotations directly.
- Perform image preprocessing in memory inside the data loader.
- Normalize and canonicalize quaternions at loading time.
- Generate Tesseract chart, path, leaf, soft targets, and residuals at loading time.
- If cached labels are useful, write derived CSV/TSV files only under `outputs/data_manifests/`.
- Derived tables reference original images using relative paths; images are never copied or rewritten.
- Validate cached labels against runtime derivation.
- Never write generated files inside the dataset directory.

Minimum derived fields are:

```text
image_relpath, domain, split,
source quaternion, canonical quaternion,
root chart, child path, leaf id,
leaf-centre quaternion, quantization error,
translation and source-annotation reference
```

---

## 3. Phase summary

| Phase | Question | Required result |
|---|---|---|
| 0 | Is the Tesseract representation correct and sufficiently precise? | Validated GT paths, grid comparison and selected maximum depth |
| 1 | What useful information comes from DINOv3 and VGGT separately? | Strongest single-branch baselines and complementarity analysis |
| 2 | Does fusion improve over the strongest branch? | One selected fusion method or a justified decision not to fuse |
| 3 | Which fixed-depth predictor works best? | Frozen fixed-depth model and decoder |
| 4 | Can one continuous posterior support cells, refinement and uncertainty? | Tangent, separate-RFM and coupled-posterior comparison |
| 5 | Can depth be selected per image? | Adaptive controller compared with all relevant fixed depths |
| 6 | Does rotation generalize and remain robust? | Complete rotation milestone report |
| 7 | How should translation and 6DoF be recovered? | Full pose system after rotation succeeds |
| 8 | Which extensions are justified? | Deployment, temporal or continual work chosen from results |

---

# Phase 0 — Data, ground truth and rotation geometry

## Goal

Validate the labels and rotation representation before extracting visual features.

## Required experiments

### EXP-000 — Loader and coordinate conventions

- Read original SPEED+ files without modification.
- Confirm quaternion order, rotation direction, handedness, translation frame and camera intrinsics.
- Test identity, axis rotations, 180-degree rotations and matrix/quaternion round trips.

### EXP-001 — Ground truth to Tesseract path

For each quaternion:

1. normalize;
2. canonicalize (q\sim-q) using the first dominant-magnitude coordinate;
3. apply (L_\infty) projection;
4. assign one of four root charts;
5. recursively generate eight-way child labels;
6. decode the leaf centre;
7. calculate geodesic quantization error.

Test `path(q) == path(-q)`, seam behavior, ties, finite values and encode/decode consistency.

### EXP-002 — Fixed-depth precision

Evaluate (L=1,2,3,4,5) using:

- number of cells;
- mean, median, 95th-percentile and maximum covering error;
- occupancy and neighbor-distance variation;
- memory and search cost.

### EXP-003 — Grid comparison

Compare Tesseract, Hopf and cubochoric grids at matched:

- cell count;
- covering radius where possible;
- search budget.

## Phase output

- validated runtime loader;
- optional derived CSV/TSV manifest;
- selected rotation representation;
- selected maximum depth (L_{\max}).

## Gate

Proceed only when the Tesseract mapping is correct and its hierarchy offers a defensible precision or search advantage. Otherwise keep it as an ablation and reconsider the representation.

---

# Phase 1 — Frozen representation controls

## Goal

Determine what DINOv3 and VGGT contribute before attempting fusion.

Use controlled or ground-truth crops initially so localization errors do not hide rotation behavior.

## Required experiments

### EXP-010 — DINOv3-only baseline

- Freeze DINOv3.
- Compare selected intermediate and final features.
- Train a common MLP Tesseract predictor.
- Record rotation error and cross-domain feature drift.

### EXP-011 — VGGT controls

Run separately:

- VGGT tokens;
- depth and confidence summaries;
- point-map or geometric summaries.

Use the same predictor capacity and training budget as EXP-010.

### EXP-012 — Complementarity audit

Measure:

- error correlation between branches;
- cases solved by only one branch;
- posterior agreement/disagreement;
- frequency behavior under glare, shadow and scale changes;
- whether VGGT adds useful evidence beyond DINOv3.

## Phase output

- strongest DINO baseline;
- strongest VGGT baseline;
- evidence for or against multimodal complementarity.

## Gate

Retain only useful feature sources. If VGGT adds no measurable information, use DINOv3 alone or invoke VGGT only for uncertain cases.

---

# Phase 2 — Fusion selection

## Goal

Find the simplest fusion method that improves over the strongest single branch.

## Required experiments

### EXP-020 — Basic controls

- best single branch;
- equal posterior averaging;
- fixed product of experts.

### EXP-021 — Training-free reliability signals

Test one at a time using the same predictors:

- entropy weighting;
- top-two margin weighting;
- DINO–VGGT agreement;
- VGGT confidence;
- spectral-energy reliability;
- query/template spectral coherence.

### EXP-022 — Best training-free combination

Combine only signals that individually helped. Optionally test guided aggregation on comparable spatial cost maps.

## Conditional experiments

Run only if training-free fusion is insufficient and Phase 1 showed complementarity:

- `EXP-023`: learned scalar or frequency-band gates;
- `EXP-024`: shallow channel projection or small reliability network;
- cross-attention/Perceiver fusion only if the shallow methods remain insufficient.

## Phase output

One selected input representation/fusion method for all fixed-depth predictor experiments.

## Gate

- Keep training-free fusion if it is competitive.
- Use the smallest learned fusion that produces a repeatable real-domain gain.
- If no fusion beats the strongest branch, continue with that branch alone.

---

# Phase 3 — Fixed-depth Tesseract predictor

## Goal

Choose the predictor and decoder before introducing adaptive depth.

All models use the same selected representation and (L_{\max}).

## Required experiments

### EXP-030 — MLP baseline

Use pooled representation, level embedding, parent-cell embedding and hierarchical root/child heads.

### EXP-031 — GRU path decoder

Test whether explicitly modelling the sequence of previous child decisions helps.

### EXP-032 — Lightweight Tesseract Transformer

Use a small autoregressive Transformer decoder:

- current cell and level form the query;
- the query cross-attends to frozen or selected visual memory;
- root output has four classes;
- later outputs have eight children;
- beam decoding is supported.

If it consumes both raw DINO and VGGT tokens, classify it as learned fusion.

### EXP-033 — Target ablation

Compare hard path labels with sparse geodesic soft targets.

### EXP-034 — Decoder ablation

Compare greedy decoding with beam widths 2, 4 and 8.

## Conditional experiment

- `EXP-035`: tangent residual only if discrete quantization limits accuracy.

## Phase output

- one fixed-depth predictor;
- one decoding method;
- fixed-depth results for every evaluated depth;
- frozen checkpoint used to construct adaptive-depth targets.

## Gate

Do not train RFM or the adaptive controller until the fixed predictor is stable and evaluated without teacher-forced paths.

---

# Phase 4 — Continuous rotation and a shared flow–Tesseract posterior

## Goal

Test whether cell probabilities and continuous rotations should be separate predictions or two views of one learned posterior on \(SO(3)\).

This positioning is essential: RFM is established on manifolds; RFMPose applies it to probabilistic 6D pose; and Flow6D studies discrete bin localization followed by continuous flow refinement. Therefore, “select a cell, then run a flow” is a baseline. The proposed contribution is to test whether one continuous rotation posterior can **induce** the Tesseract tree, continuous samples, uncertainty, and later adaptive-depth decisions.

Primary references:

- [Flow Matching on General Geometries (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d1f9936d3be6997ffffab692977eebe6-Abstract-Conference.html)
- [RFMPose (NeurIPS 2025)](https://papers.neurips.cc/paper_files/paper/2025/hash/618c0a236e7aace748d09bc8fbed3e4d-Abstract-Conference.html)
- [Flow6D (arXiv:2606.23293)](https://arxiv.org/abs/2606.23293)

## Common flow formulation

For image representation \(z_I\), model an angular velocity field on \(SO(3)\):

\[
\frac{dR_\tau}{d\tau}=R_\tau\widehat{\omega_\theta(R_\tau,\tau,z_I)},\qquad \tau\in[0,1].
\]

For a sampled base rotation \(R_0\) and GT \(R^\star\), use the geodesic path

\[
a=\operatorname{Log}(R_0^\top R^\star)^\vee,\qquad
R_\tau=R_0\operatorname{Exp}(\tau\widehat a),
\]

with the initial matching objective

\[
\mathcal L_{FM}=\mathbb E\left[\lVert\omega_\theta(R_\tau,\tau,z_I)-a\rVert_2^2\right].
\]

Record the chosen base distribution, treatment of the log-map cut locus, solver, number of function evaluations (NFE), endpoint samples \(M\), and sampling latency. Keep flow time \(\tau\) separate from tree depth \(l\).

## Required comparisons

### EXP-040 — Tesseract plus tangent residual

Use the selected cell centre \(R_c\) and predict \(\delta\in\mathbb R^3\):

\[
\widehat R=R_c\operatorname{Exp}(\widehat\delta).
\]

This is the low-cost continuous baseline and already preserves valid rotations.

### EXP-041 — Tesseract plus separate RFM

Condition a local \(SO(3)\) flow on the selected cell and image representation. Keep its cell classifier loss and flow loss separate. Measure whether it improves multimodal recall, rotation error, and coverage over the tangent residual.

### EXP-042 — Image-conditioned \(SO(3)\) posterior

Train the global/image-conditioned flow. For every evaluation image, generate endpoint rotations \(R_1^{(1)},\ldots,R_1^{(M)}\). Report:

- best-of-\(M\), mean/median sample error, and sample diversity;
- angular credible-region coverage at nominal levels;
- NFE, latency, and memory versus \(M\);
- behavior for visually ambiguous or symmetric cases.

Do not call the samples calibrated unless measured coverage supports it.

### EXP-043 — Flow-induced Tesseract masses

Let \(\Omega_c\) be cell \(c\)'s rotation region. The desired cell mass is

\[
\mu_1(c\mid I)=\int_{\Omega_c}\rho_1(R\mid I)\,dV(R),
\qquad
\mu_1(c\mid I)=\sum_{k=0}^{7}\mu_1(ck\mid I).
\]

For the first implementation, do not estimate boundary fluxes. Map the **same** endpoint samples to Tesseract paths and count them:

\[
\widehat\mu_1(c\mid I)=\frac{1}{M}\sum_{m=1}^{M}\mathbf 1[R_1^{(m)}\in\Omega_c].
\]

The same sample set must produce root, parent, child, and leaf masses so empirical parent–child consistency is exact. Report mass error versus \(M\), GT-cell recall by depth, occupancy, and seam cases.

### EXP-044 — Distilled probability-consistent tree

Train the lightweight hierarchical predictor with

\[
\mathcal L_{tree}=\mathcal L_{GT\ path}+\beta\sum_l
\operatorname{KL}\!\left(\operatorname{stopgrad}[\widehat\mu_1^{(l)}]\,\Vert\,p_\phi^{(l)}\right).
\]

Compute \(p_\phi^{(l)}\) from root and conditional child probabilities. Compare \(\beta=0\) with a small validated set of nonzero values. Save per-level flow masses and distilled probabilities. The distilled tree is the primary route to fast inference; subdividing samples after a completed ODE does not save flow computation.

### EXP-045 — Matched comparison and decision

Compare only these first:

| System | Main question |
|---|---|
| Tesseract + tangent residual | Is one-step continuous correction sufficient? |
| Tesseract + separate RFM | Does a probabilistic refiner add useful modes or uncertainty? |
| Flow-induced Tesseract posterior | Does one posterior improve cross-depth consistency and routing? |
| Coupled posterior + distilled tree | Can the posterior be approximated with useful speed and accuracy? |

Use geodesic rotation error, GT-cell recall, parent–child consistency error, credible-region coverage, NLL if valid, NFE, samples, latency, and memory.

## Gate

Continue with the coupled posterior only if it offers a measured advantage in accuracy, multimodal recall, uncertainty/coverage, or tree consistency that justifies its cost. Otherwise use the tangent residual or separate RFM baseline and document the negative result.

---

# Phase 5 — Adaptive depth

## Goal

After the fixed representation and continuous-rotation method are selected, decide per image whether to stop, descend through one child, or retain multiple branches.

## EXP-050 — Oracle adaptive targets

Run the selected fixed-depth system on the training split without GT routing. At each level record decoded error \(e_l\), computation \(C_l\), posterior, beam, and the best deeper result. Define

\[
l^*=\arg\min_l[e_l+\lambda_{cost}C_l]
\]

and generate `STOP`, `DESCEND_ONE`, and `BRANCH` targets. If the coupled posterior is retained, also record within-cell angular spread and credible mass.

## EXP-051 — Feature validation

Test each signal independently with logistic regression or the same small MLP:

1. child entropy;
2. top-two margin;
3. object resolution/token count;
4. DINO–VGGT agreement;
5. VGGT confidence;
6. spectral quality/coherence;
7. current-cell angular radius;
8. cumulative path or posterior mass;
9. within-cell angular spread and credible mass from the flow, if available;
10. the full validated subset.

Report AUROC/AUPRC for “deepen,” action accuracy, rotation error, credible coverage, nodes visited, NFE, and latency. Do not combine a signal until its individual value is known.

## EXP-052 — Rule-based controller

Use interpretable thresholds as the adaptive-depth baseline.

## EXP-053 — MLP controller

Use validated scalar signals, level embedding, and current-cell embedding.

## Conditional experiment

- `EXP-054`: add a small cross-attentive Transformer `STOP/DESCEND/BRANCH` head only if the MLP is limited and token-level evidence is demonstrably useful.

## EXP-055 — Fixed versus adaptive comparison

Compare the controller with every relevant fixed depth at matched budgets. Report error, coverage, nodes visited, NFE, latency, memory, early-stop rate, branch rate, and false-precision cases by domain and object size.

## Gate

Keep adaptive depth only if it approaches the deepest model's accuracy and coverage while reducing computation or unjustified fine predictions.

---

# Phase 6 — Rotation validation milestone

## Goal

Establish whether the selected rotation method generalizes.

## Required experiments

### EXP-060 — Domain evaluation

Report synthetic validation, lightbox, and sunlamp separately.

### EXP-061 — Robustness

Generate corruptions at loading time and test glare/saturation, shadow, blur, target resolution, occlusion, and background changes.

### EXP-062 — Confidence and failure detection

Evaluate entropy, agreement, beam diversity, posterior spread, credible-region coverage, calibration, and risk–coverage curves. Use only signals available to the selected method.

### EXP-063 — Efficiency

Profile backbone extraction, fusion, hierarchy, flow solver/sampling, distillation benefit, adaptive controller, and refinement separately.

## Milestone output

The rotation report must include matched-grid results, single-branch and fusion results, tangent/separate-RFM/coupled-posterior comparisons, fixed and adaptive-depth results, cross-domain robustness, calibration/coverage, the accuracy–compute frontier, and the selected final architecture.

Stop here for the first research milestone.

---

# Phase 7 — Translation and complete 6DoF

Begin only after Phase 6.

## Required baseline

### EXP-070 — Localization control

Compare controlled crop with the simplest viable automatic localization method.

### EXP-071 — DINO/fused correspondences with RANSAC-PnP

This is the first translation baseline.

## Conditional alternatives

- `EXP-072`: VGGT depth/point-map alignment with the CAD model;
- `EXP-073`: learned translation head if geometric methods are inadequate;
- `EXP-074`: local multi-hypothesis \(SE(3)\) refinement.

### EXP-075 — Full evaluation

Report orientation error, normalized translation error, and official SPEED+ pose score by domain.

---

# Phase 8 — Optional extensions

Choose only after the single-image 6DoF results indicate what is needed.

Possible branches:

- DINO-first compute cascade that invokes VGGT only for uncertain inputs;
- temporal Tesseract prior and beam propagation;
- Kalman or particle filtering;
- learned motion model or temporal flow prior;
- continual calibration, prototype updates, or adapters;
- new-spacecraft CAD onboarding.

Each becomes a new phase with its own baseline and report. Do not mix these extensions into the first rotation result.

---

## 4. Reporting after every experiment

For each experiment:

1. create its Markdown report with status `Running`;
2. save all artifacts under its unique `outputs/experiments/` run directory;
3. write the numerical results into the report;
4. record failures and negative outcomes;
5. update `outputs/EXPERIMENT_INDEX.md`;
6. select `Keep`, `Reject`, `Modify` or `Repeat`;
7. identify the next experiment permitted by the current phase gate.

The agent should not automatically continue into an optional branch merely because it appears in this plan.
