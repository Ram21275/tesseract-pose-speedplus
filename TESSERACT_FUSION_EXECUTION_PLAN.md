# Phase-Wise Execution Plan: DINOv3–MoGe-2 Fusion and Hierarchical Tesseract Pose Prediction

> **Branch change (2026-10-09).** MoGe-2 replaced VGGT as the geometric branch (DEC-001, `outputs/DECISIONS.md`). Completed VGGT experiments (EXP-011, EXP-012) stay in the experiment index as historical results; their IDs are not renamed or overwritten. All new experiments use MoGe-2 wherever this plan previously said VGGT.

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

**Translation additions (for Phases 7–8):**

- Do not modify SPEED+ images or annotations.
- Generate translation targets during loading.
- Generate translation-octree paths, cell centres and residuals during loading.
- Camera-frame conversions and translation normalization must also happen during loading.
- If caching is needed, create a new CSV/TSV under `outputs/data_manifests/`.
- The cached table must reference the original image path and original annotation.
- Fit normalization values and translation bounds using the training split only.
- Save those values in the experiment configuration.

Possible additional derived fields:

```text
translation_x, translation_y, translation_z,
translation_range,
translation_direction_x,
translation_direction_y,
translation_direction_z,
projected_center_x,
projected_center_y,
log_depth,
translation_root,
translation_child_path,
translation_leaf_id,
translation_cell_center,
translation_residual
```

---

## 3. Phase summary

| Phase | Purpose | Question | Required result |
|---|---|---|---|
| 0 | Ground truth and rotation geometry | Is the Tesseract representation correct and sufficiently precise? | Validated GT paths, grid comparison and selected maximum depth |
| 1 | DINOv3 and MoGe-2 representations | What useful information comes from DINOv3 and MoGe-2 separately? (VGGT: historical, EXP-011/012) | Strongest single-branch baselines and complementarity analysis |
| 2 | Fusion experiments | Does fusion improve over the strongest branch? | One selected fusion method or a justified decision not to fuse |
| 3 | Fixed-depth Tesseract prediction | Which fixed-depth predictor works best? | Frozen fixed-depth model and decoder |
| 4 | Continuous rotation and SO(3) flow | Can one continuous rotation posterior support cells, refinement and uncertainty? (rotation only) | Tangent, separate-RFM and coupled-posterior comparison |
| 5 | Adaptive rotation depth | Can rotation depth be selected per image? | Adaptive controller compared with all relevant fixed depths |
| 6 | Rotation milestone | Does rotation generalize and remain robust? | Complete rotation milestone report |
| 7 | Translation and standard 6DoF prediction | How should translation and full 6DoF be recovered without flow? | Standard 6DoF system (EXP-070–079) |
| 8 | Joint translation–rotation flow on SE(3) | Can one continuous pose posterior induce consistent rotation-tree and translation-octree probabilities, and be distilled? | Factorized vs coupled SE(3) flow and distilled hierarchy (EXP-080–085) |
| 9 | Optional extensions | Which extensions are justified? | Deployment, temporal or continual work chosen from results |


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

Determine what DINOv3 and the geometric branch contribute before attempting fusion. The geometric branch is **MoGe-2** (EXP-013/014, DEC-001); the VGGT controls below (EXP-011/012) are completed historical results.

Use controlled or ground-truth crops initially so localization errors do not hide rotation behavior.

## Required experiments

### EXP-010 — DINOv3-only baseline

- Freeze DINOv3.
- Compare selected intermediate and final features.
- Train a common MLP Tesseract predictor.
- Record rotation error and cross-domain feature drift.

### EXP-011 — VGGT controls (historical; superseded by EXP-013 MoGe-2 controls)

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
- whether the geometric branch (MoGe-2; historically VGGT) adds useful evidence beyond DINOv3.

## Phase output

- strongest DINO baseline;
- strongest geometric-branch baseline (MoGe-2; VGGT historical);
- evidence for or against multimodal complementarity.

## Gate

Retain only useful feature sources. If MoGe-2 adds no measurable information, use DINOv3 alone or invoke MoGe-2 only for uncertain cases.

---

# Phase 2 — Fusion selection

> **Numbering (DEC-002):** the Phase-2 experiments listed below as EXP-020–024 are run as **EXP-025–029** (EXP-015–020 were used by user-requested side experiments).

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
- DINO–MoGe-2 agreement;
- MoGe-2 confidence (foreground-mask / geometry consistency);
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

If it consumes both raw DINOv3 and MoGe-2 tokens, classify it as learned fusion.

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

> **Rotation only.** Do not add translation to Phase 4. It must stay rotation-only so we can determine whether SO(3) flow matching is useful before introducing the extra complexity of SE(3) (Phase 8).

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
4. DINO–MoGe-2 agreement;
5. MoGe-2 confidence;
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

Begin Phase 7 only after the Phase 6 rotation milestone is complete.

Use controlled or ground-truth crops for the first translation experiments. Test automatic localization later, after the translation head is stable.

## EXP-070 — SPACE-HOP-style translation baseline

This is the first translation experiment.

Implement a translation head that follows the SPACE-HOP formulation as closely as possible:

- use the exact scale-invariant translation target defined by the paper or the official implementation;
- start with the selected frozen DINOv3 representation;
- use the same crop and camera conventions as the rotation experiments;
- begin with a small MLP translation head;
- use Smooth L1 / Huber loss with the paper's stated \(\beta=0.01\).

Do not guess the definition of the scale-invariant vector. Confirm it from the paper or the official code, and record the exact equations in the experiment report. The SPACE-HOP paper describes its translation head as predicting a scale-invariant vector with a Smooth L1 loss.

Report:

- translation direction error;
- absolute translation error in metres;
- normalized translation error;
- error as a function of object distance and image size;
- results separately for the synthetic, lightbox and sunlamp domains.

Create:

```text
EXP-070_SPACEHOP_TRANSLATION_BASELINE.md
outputs/experiments/EXP-070_spacehop_translation/
```

---

## EXP-071 — Translation-target parameterization

Run directly after EXP-070, with the same representation, MLP size and training budget. Compare:

### A. SPACE-HOP target

The target selected in EXP-070.

### B. Direct camera-frame translation

\[
\widehat{\mathbf t}=
(\widehat t_x,\widehat t_y,\widehat t_z).
\]

### C. Direction and logarithmic range

\[
r=\lVert\mathbf t\rVert_2,\qquad
\mathbf n=\frac{\mathbf t}{r}.
\]

Predict:

\[
(\widehat{\mathbf n},\log \widehat r),
\qquad
\widehat{\mathbf t}
=
\exp(\log\widehat r)
\frac{\widehat{\mathbf n}}
{\lVert\widehat{\mathbf n}\rVert_2}.
\]

### D. Projected centre and logarithmic depth

Using the camera intrinsics:

\[
x_n=\frac{t_x}{t_z},\qquad
y_n=\frac{t_y}{t_z}.
\]

Predict:

\[
(x_n,y_n,\log t_z).
\]

Recover the translation with:

\[
t_z=\exp(\widehat{\log t_z}),\qquad
t_x=\widehat x_n t_z,\qquad
t_y=\widehat y_n t_z.
\]

Select one translation representation before changing the model architecture.

Create:

```text
EXP-071_TRANSLATION_PARAMETERIZATION.md
outputs/experiments/EXP-071_translation_targets/
```

---

## EXP-072 — Translation-feature comparison

Run after the target parameterization is selected. Using the target chosen in EXP-071, compare:

1. DINOv3 features only;
2. MoGe-2 depth/geometry summaries only;
3. DINOv3 and MoGe-2 concatenation;
4. the selected fusion representation from Phase 2;
5. optionally, learned scalar gating, but only if simple concatenation shows complementarity.

Use the same translation head for every comparison.

This experiment answers whether MoGe-2 is more useful for translation than it was for rotation.

Create:

```text
EXP-072_TRANSLATION_FEATURES.md
outputs/experiments/EXP-072_translation_features/
```

---

## EXP-073 — Translation-head architecture

Run after EXP-072. Compare a small set of models using the selected target and features:

1. MLP baseline;
2. residual MLP;
3. a shallow token cross-attention head, if token-level information appears useful;
4. an uncertainty-aware head predicting translation and variance.

Do not begin with a large Transformer. Keep the MLP as the main baseline.

A possible objective for the uncertainty-aware regression head:

\[
\mathcal L_t=
\frac{\lVert\mathbf t^\star-\widehat{\mathbf t}\rVert_2^2}
{2\sigma_t^2}
+\frac{1}{2}\log\sigma_t^2.
\]

Keep uncertainty prediction only if coverage or failure detection improves.

Create:

```text
EXP-073_TRANSLATION_HEADS.md
outputs/experiments/EXP-073_translation_heads/
```

---

## EXP-074 — Hierarchical translation octree

Run after the ordinary regression baseline is stable. This is the translation equivalent of the Tesseract subdivision.

Define a bounded translation region from training-set statistics and camera constraints, and recursively split each cell into eight children.

For a selected translation cell \(c_t\), predict a local residual:

\[
\widehat{\mathbf t}
=
\mathbf t_{c_t}
+
\mathbf s_{c_t}\odot\widehat{\boldsymbol\delta}_t,
\]

where:

- \(\mathbf t_{c_t}\) is the cell centre;
- \(\mathbf s_{c_t}\) is the cell size;
- \(\widehat{\boldsymbol\delta}_t\) is a normalized residual.

Compare:

1. direct regression;
2. fixed-depth octree classification;
3. octree classification plus residual;
4. hard targets versus neighbouring soft targets.

Report the number of samples outside the translation bounds explicitly. Do not silently clip them.

Create:

```text
EXP-074_TRANSLATION_OCTREE.md
outputs/experiments/EXP-074_translation_octree/
```

---

## EXP-075 — Joint Tesseract–octree predictor

Run after EXP-074. Combine:

- the rotation Tesseract hierarchy;
- the translation octree;
- the selected continuous rotation correction;
- the selected continuous translation correction.

Test two probability structures.

### Independent heads

\[
p(c_R,c_t\mid I)
=
p(c_R\mid I)p(c_t\mid I).
\]

### Conditioned translation head

\[
p(c_R,c_t\mid I)
=
p(c_R\mid I)p(c_t\mid c_R,I).
\]

Use a joint loss such as:

\[
\mathcal L_{\mathrm{pose}}
=
\mathcal L_R
+\lambda_t\mathcal L_t
+\lambda_{\mathrm{res},R}\mathcal L_{\mathrm{res},R}
+\lambda_{\mathrm{res},t}\mathcal L_{\mathrm{res},t}.
\]

Select the weights on the validation set, and record their units and normalization.

Create:

```text
EXP-075_JOINT_TESSERACT_OCTREE.md
outputs/experiments/EXP-075_joint_pose_tree/
```

---

## EXP-076 — Adaptive translation or joint depth

Run after the fixed-depth joint predictor.

Do not reuse the rotation controller straight away. First test the translation signals one at a time:

1. translation-child entropy;
2. top-two translation margin;
3. translation-cell physical size;
4. predicted distance;
5. object pixel size;
6. MoGe-2 confidence;
7. regression variance;
8. rotation–translation consistency;
9. difference between successive translation-depth predictions.

Compare:

- fixed rotation and translation depths;
- adaptive rotation depth only;
- adaptive translation depth only;
- adaptive rotation and translation depths.

The controller may choose different depths \((l_R,l_t)\), because rotation and translation uncertainty need not decrease at the same rate.

Create:

```text
EXP-076_ADAPTIVE_6DOF_DEPTH.md
outputs/experiments/EXP-076_adaptive_6dof/
```

---

## EXP-077 — Geometric translation controls

Run this after the learned translation experiments, so that it remains an independent baseline. Test:

1. DINOv3 or fused correspondences with RANSAC-PnP;
2. MoGe-2 depth/point-map alignment with the CAD model, if compatible;
3. optionally, local \(SE(3)\) refinement starting from the learned prediction.

Use the same crops and test images as the learned methods.

Create:

```text
EXP-077_GEOMETRIC_TRANSLATION_CONTROLS.md
outputs/experiments/EXP-077_geometric_controls/
```

---

## EXP-078 — Localization control

Compare:

1. the ground-truth / controlled crop;
2. a predicted bounding box or automatic localization;
3. deliberately perturbed crops.

This separates pose-estimation error from localization error.

Create:

```text
EXP-078_LOCALIZATION_CONTROL.md
outputs/experiments/EXP-078_localization/
```

---

## EXP-079 — Standard full 6DoF evaluation

This concludes the standard, non-flow translation phase.

Report by domain:

- geodesic rotation error;
- translation direction error;
- absolute translation error;
- normalized translation error;
- official SPEED+ pose score;
- inference time;
- memory;
- parameter count;
- failure rate;
- performance against distance, scale, glare and shadow.

The main comparisons are:

1. rotation model plus direct translation regression;
2. rotation model plus SPACE-HOP-style translation;
3. Tesseract plus translation octree;
4. joint Tesseract–octree predictor;
5. geometric PnP/CAD baseline;
6. adaptive joint hierarchy.

Create:

```text
EXP-079_FULL_6DOF_EVALUATION.md
outputs/experiments/EXP-079_full_6dof/
```

---

# Phase 8 — Joint flow matching on SE(3)

This phase comes after the complete standard 6DoF evaluation.

The aim is not simply to attach a translation flow after rotation classification. The main research question is:

> Can one continuous pose posterior induce consistent probabilities over both the Tesseract rotation tree and the translation octree, and can that posterior be distilled into a fast adaptive predictor?

Do not claim this as novel until the literature review is complete. Treat it as the proposed contribution to test.

## EXP-080 — Factorized \(SO(3)\times\mathbb R^3\) flow

Use the selected \(SO(3)\) flow from Phase 4 and add a Euclidean translation flow.

For rotation:

\[
a=\operatorname{Log}(R_0^\top R^\star)^\vee,
\qquad
R_\tau=R_0\operatorname{Exp}(\tau\widehat a).
\]

For translation:

\[
\mathbf t_\tau
=
(1-\tau)\mathbf t_0+\tau\mathbf t^\star,
\qquad
\mathbf v^\star
=
\mathbf t^\star-\mathbf t_0.
\]

Train:

\[
\mathcal L_{\mathrm{factorized}}
=
\lVert\widehat{\boldsymbol\omega}-a\rVert_2^2
+
\lambda_t
\lVert\widehat{\mathbf v}-\mathbf v^\star\rVert_2^2.
\]

This is the main baseline for the coupled \(SE(3)\) model.

Create:

```text
EXP-080_FACTORIZED_POSE_FLOW.md
outputs/experiments/EXP-080_factorized_flow/
```

---

## EXP-081 — Coupled Lie-algebra \(SE(3)\) flow

Represent a pose as

\[
T=
\begin{bmatrix}
R & \mathbf t\\
0 & 1
\end{bmatrix}
\in SE(3).
\]

For base pose \(T_0\) and target \(T^\star\):

\[
\boldsymbol\xi^\star
=
\operatorname{Log}(T_0^{-1}T^\star)^\vee,
\]

with the path

\[
T_\tau
=
T_0\operatorname{Exp}
\left(\tau\widehat{\boldsymbol\xi^\star}\right).
\]

Learn

\[
\frac{dT_\tau}{d\tau}
=
T_\tau\widehat{\boldsymbol\xi_\theta(T_\tau,\tau,z_I)}.
\]

Angular and linear velocity have different units, so use a declared translation scale \(\ell_t\):

\[
\mathcal L_{SE(3)}
=
\lVert\widehat{\boldsymbol\omega}
-\boldsymbol\omega^\star\rVert_2^2
+
\frac{1}{\ell_t^2}
\lVert\widehat{\mathbf v}
-\mathbf v^\star\rVert_2^2.
\]

Choose \(\ell_t\) from training-set statistics or the spacecraft scale, and record it explicitly.

Create:

```text
EXP-081_COUPLED_SE3_FLOW.md
outputs/experiments/EXP-081_se3_flow/
```

---

## EXP-082 — Flow-induced Tesseract–octree probabilities

Use the same endpoint pose samples for both rotation and translation. For samples

\[
(R_1^{(m)},\mathbf t_1^{(m)}),\qquad m=1,\ldots,M,
\]

estimate the joint cell probability by counting:

\[
\widehat\mu(c_R,c_t\mid I)
=
\frac{1}{M}
\sum_{m=1}^{M}
\mathbf 1
\left[
R_1^{(m)}\in\Omega_{c_R},
\mathbf t_1^{(m)}\in\Omega_{c_t}
\right].
\]

The same samples must generate:

- Tesseract parent and child masses;
- translation-octree parent and child masses;
- joint rotation–translation cell masses;
- rotation and translation marginals.

This guarantees empirical parent–child consistency.

Measure how the estimated masses change with the sample count \(M\).

Create:

```text
EXP-082_FLOW_INDUCED_PRODUCT_TREE.md
outputs/experiments/EXP-082_product_tree_masses/
```

---

## EXP-083 — Distillation into the fast joint hierarchy

Train the Tesseract–octree predictor to reproduce the joint flow distribution:

\[
\mathcal L_{\mathrm{distill}}
=
\mathcal L_{\mathrm{GT\ pose}}
+
\beta_R
\sum_l
\operatorname{KL}
\left(
\widehat\mu_R^{(l)}
\Vert
p_R^{(l)}
\right)
+
\beta_t
\sum_k
\operatorname{KL}
\left(
\widehat\mu_t^{(k)}
\Vert
p_t^{(k)}
\right)
+
\beta_{Rt}\mathcal L_{\mathrm{joint}}.
\]

Detach the sampled flow probabilities in the first implementation.

The aim is to use the expensive flow during training while keeping fast hierarchical inference at test time.

Create:

```text
EXP-083_SE3_FLOW_DISTILLATION.md
outputs/experiments/EXP-083_se3_distillation/
```

---

## EXP-084 — Joint-flow adaptive depth

Test whether flow-derived uncertainty improves depth decisions. Possible signals:

- Tesseract posterior mass;
- translation-octree posterior mass;
- within-cell angular spread;
- within-cell translation spread;
- joint pose entropy;
- number of significant pose modes;
- rotation–translation dependence;
- credible-region size.

Compare these signals with the existing entropy, margin, agreement and object-scale signals.

Create:

```text
EXP-084_FLOW_ADAPTIVE_DEPTH.md
outputs/experiments/EXP-084_flow_adaptive_depth/
```

---

## EXP-085 — Final matched comparison

Compare at matched compute or sampling budgets:

1. Tesseract plus tangent residual and MLP translation;
2. Tesseract plus tangent residual and translation octree;
3. separate rotation and translation flows;
4. factorized \(SO(3)\times\mathbb R^3\) flow;
5. coupled \(SE(3)\) flow;
6. flow-distilled Tesseract–octree;
7. flow-distilled hierarchy with adaptive depth.

Report:

- rotation error;
- translation error;
- official pose score;
- best-of-\(M\) results;
- credible-region coverage;
- negative log-likelihood, if correctly defined;
- parent–child probability consistency;
- NFE and sample count;
- inference latency;
- memory and parameter count.

Continue with the coupled \(SE(3)\) approach only if it gives a measurable benefit over the simpler factorized flow.

Create:

```text
EXP-085_FINAL_SE3_COMPARISON.md
outputs/experiments/EXP-085_final_se3/
```

---

# Phase 9 — Optional extensions

Choose these only after the single-image 6DoF results show what is needed. Do not begin temporal models, continual learning or spacecraft onboarding until EXP-085 is complete.

Possible branches:

- a DINO-first compute cascade that invokes MoGe-2 only for uncertain inputs;
- a temporal Tesseract prior and beam propagation;
- Kalman or particle filtering;
- a learned motion model or temporal flow prior;
- continual calibration, prototype updates or adapters;
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

## 5. Required reporting rule

Every experiment must have its own Markdown report and output directory.

Before running an experiment:

1. create its report with status `Running`;
2. record the hypothesis, configuration, dataset split and evaluation metrics.

After running it:

1. write all numerical results into the same Markdown report;
2. link CSV, TSV, plots, images, checkpoints and logs;
3. record failed runs and negative findings;
4. update `outputs/EXPERIMENT_INDEX.md`;
5. assign `Keep`, `Reject`, `Modify` or `Repeat`;
6. state which experiment is allowed next by the phase gate.

Do not merge several experiment results into one report, and do not silently continue to conditional experiments.
