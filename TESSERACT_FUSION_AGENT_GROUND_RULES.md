# Agent Ground Rules: DINOv3–VGGT Fusion and Hierarchical Tesseract Pose Prediction

## 1. Project objective

Develop and evaluate a spacecraft-pose pipeline that combines frozen DINOv3 semantic features and frozen VGGT geometric evidence, then predicts rotation through a hierarchical Tesseract discretization of \(SO(3)\).

The immediate research question is:

> Can DINOv3 and VGGT be fused without task-specific fusion training while retaining enough semantic and geometric evidence for accurate hierarchical Tesseract-cell prediction? If not, what is the smallest trainable fusion module that produces a meaningful, reproducible improvement?

The fusion method and the Tesseract predictor are separate components and must always be evaluated separately. After the fixed-depth predictor is stable, test whether an image-conditioned \(SO(3)\) flow can define one posterior whose probability masses induce the Tesseract tree, continuous rotation estimates, and adaptive-depth decisions.

## 2. Non-negotiable scope rules

1. **Start with training-free fusion.** Implement and evaluate deterministic fusion before introducing a learned fusion module.
2. **Training-free fusion is preferred, not mandatory.** Keep it if it is competitive. If it fails, document the failure and move to a minimal trainable fusion model.
3. **A trainable Tesseract predictor is allowed from the beginning.** The predictor may learn root cells, child paths, confidence, or local residuals.
4. **Keep DINOv3 and VGGT frozen in the principal experiments.** Any backbone fine-tuning is a separate, explicitly named experiment and cannot be mixed into the main comparison.
5. **Do not silently alter the architecture, dataset split, evaluation metric, or task.** Propose and record any such change before implementing it.
6. **Do not claim that the method eliminates the sim-to-real gap.** Measure the gap independently on SPEED+ synthetic, lightbox, and sunlamp domains.
7. **Do not describe pretrained frozen models as untrained.** Use the terms `no task-specific fusion training`, `frozen-backbone`, or `training-free fusion`, as applicable.
8. **Phase 1 must isolate rotation.** Translation and complete 6DoF estimation may be implemented later, but must not obscure whether the Tesseract rotation representation works.
9. **Do not present “cell selection followed by flow refinement” as the main novelty.** RFMPose already applies Riemannian flow matching to probabilistic pose estimation, and Flow6D already studies discrete localization followed by continuous flow refinement. The proposed contribution to test is a continuous \(SO(3)\) posterior that induces a probability-consistent Tesseract tree and supports adaptive depth.
10. **Treat uncertainty claims as hypotheses.** Sampling does not by itself prove calibration. Test credible-region coverage, selective risk, and calibration error.

## 3. Fusion policy: training-free first, learned only when justified

### Track A — Strictly training-free fusion

The following are allowed:

- deterministic spatial resizing and grid alignment;
- fixed normalization such as L2 normalization;
- fixed low-, mid-, and high-frequency masks;
- FFT, inverse FFT, Laplacian pyramids, phase correlation, spectral energy ratios, and cross-power coherence;
- fixed confidence rules derived from VGGT confidence, saturation masks, foreground support, or score entropy;
- deterministic guided filtering or bilateral filtering;
- fixed weighted sums, products, robust means, ranks, or product-of-experts combinations;
- deterministic score-level or residual-level fusion.

The following are **not** strictly training-free and must not be placed in Track A:

- learned projection matrices;
- attention or MLP fusion layers;
- weights optimized on pose labels;
- label-tuned thresholds or temperatures;
- regression fitted to select fusion weights;
- end-to-end gradient flow through fusion.

PCA, whitening, clustering, or statistics estimated from a dataset must be reported as a separate **label-free calibrated** variant. Do not call that variant strictly training-free.

### Track B — Minimal trainable fusion

Track B may begin only after at least one complete Track-A baseline has been run and reported. Start with the smallest plausible module:

1. scalar or band-wise gates;
2. channel projection plus gated sum;
3. shallow MLP or 1\(\times\)1 convolution;
4. cross-attention or deeper fusion only if simpler models fail.

For every learned fusion model, report:

- trainable parameter count;
- FLOPs or measured latency;
- training data and labels used;
- whether DINOv3 and VGGT remain frozen;
- improvement over the best training-free fusion under the same predictor and split;
- improvement over DINO-only and VGGT-only baselines;
- synthetic-to-real generalization change.

A trainable fusion module is justified only when the improvement is consistent across seeds or clearly improves the real-domain results without unacceptable cost.

## 4. Feature-alignment rules

1. Never assume DINOv3 and VGGT have a one-to-one token correspondence merely because both are vision transformers.
2. Record the exact model variants, patch sizes, input sizes, token grids, channel dimensions, preprocessing, and layers used.
3. Align spatial grids explicitly. Record interpolation mode, target grid, padding, and crop transformations.
4. Equal spatial sizes do not imply equal channel semantics. Raw addition or concatenation requires a clearly defined normalization and comparison baseline.
5. Prefer initial fusion of scalar evidence maps, correspondence scores, residual maps, or confidence maps over direct addition of heterogeneous feature channels.
6. Preserve FFT phase whenever spatial localization is required. Spectral energy alone may be used for reliability estimation but cannot recover spatial structure.
7. Do not assume that DINOv3 contains only low-frequency information or VGGT only high-frequency information. Measure their band energies and task relevance.
8. Guided filtering must operate on a spatial map. Specify the filtered quantity and the guide. Do not claim that it removes glare without a direct glare-corruption experiment.

## 5. Tesseract representation rules

1. Normalize every ground-truth quaternion.
2. Handle the quaternion equivalence \(q\sim -q\) deterministically.
3. Use the first maximum-magnitude coordinate as the chart index and canonicalize its sign to positive.
4. Project using \(x=q/\lVert q\rVert_\infty\), then recursively split the three free coordinates.
5. Use a documented boundary convention; equality must always go to the same child, such as the upper half.
6. With antipodal canonicalization, use four unique root charts and eight children per recursive level. If eight signed faces are retained, document the duplicate antipodal representation.
7. Never call a grid “ultra-fine” without measuring its empirical angular covering radius.
8. Compute rotation distance with

   \[
   d_{SO(3)}(q_1,q_2)=2\arccos\!\left(\left|q_1^\top q_2\right|\right).
   \]

9. Geodesic soft targets must be based on this distance, not Euclidean cell indices.
10. Training may use teacher forcing for child heads. Inference must never receive the ground-truth path.
11. Inference must support greedy and beam-search decoding. Report beam width and visited-node count.
12. Keep discrete prediction and optional continuous residual refinement as separate ablations.

Required geometry-only tests:

- `path(q) == path(-q)`;
- `encode(decode(path)) == path`;
- finite outputs for all valid inputs;
- deterministic seam and tie behavior;
- decreasing mean quantization error with depth;
- mean, median, 95th-percentile, and maximum covering error;
- cell occupancy and nearest-neighbor-distance variation;
- seam-neighbor consistency.

### Flow–Tesseract coupling rules

1. Keep flow time \(\tau\) and Tesseract subdivision depth \(l\) as different variables. Deeper subdivision is not later flow time.
2. Implement the first flow experiment on \(SO(3)\), using valid rotations and geodesic interpolation. Do not run an unconstrained Euclidean quaternion flow and renormalize it without a separate ablation and justification.
3. Respect \(q\sim-q\) when mapping flow samples to Tesseract paths. Chart seams are coordinate seams, not physical barriers.
4. Let \(\Omega_c\) be the region represented by cell \(c\). For an endpoint density \(\rho_1(R\mid I)\), define

   \[
   \mu_1(c\mid I)=\int_{\Omega_c}\rho_1(R\mid I)\,dV(R).
   \]

   The hierarchy must satisfy \(\mu_1(c\mid I)=\sum_k\mu_1(ck\mid I)\), up to numerical error.
5. In the first implementation, estimate masses by assigning the **same endpoint samples** to paths at every level. This gives exact empirical parent–child consistency. Do not begin with boundary-flux integration.
6. Distill flow-derived masses into the fast tree predictor with detached targets. Keep GT-path supervision and flow-posterior supervision as separate loss terms and ablate their weight.
7. Report the flow base distribution, interpolation/path construction, ODE solver, integration steps or function evaluations, endpoint sample count, sample temperature if any, and sampling latency.
8. Compare four systems before expanding scope: Tesseract plus tangent residual; Tesseract plus a separate RFM refiner; flow-induced Tesseract probabilities; and the coupled model with adaptive depth.
9. A tangent residual is a strong baseline: composing a tangent vector with the exponential map already preserves valid rotations. The flow must demonstrate value through multimodality, uncertainty, cross-depth consistency, or accuracy—not merely continuity.
10. Do not claim computational savings from subdividing completed flow samples. Cheaper inference must come from the distilled tree, fewer flow evaluations, fewer samples, or adaptive computation, and must be measured.

## 6. Dataset and evaluation discipline

Use the exact dataset name **SPEED+** when evaluating the SPACE-HOP-related pipeline.

Default split policy:

- synthetic training split: predictor and any Track-B fusion training;
- synthetic validation split: model selection and hyperparameter selection;
- lightbox: held-out real-domain evaluation;
- sunlamp: held-out glare/illumination-domain evaluation.

Do not tune on lightbox or sunlamp results and then report those domains as untouched tests. If real-domain labels are used for tuning, create and record a new real-domain validation/test split.

Primary rotation metrics:

- mean and median geodesic error in degrees;
- 95th-percentile geodesic error;
- accuracy below 1°, 3°, 5°, 10°, and 20°;
- root accuracy, full-path accuracy, and top-\(k\) leaf recall;
- calibration or selective-risk curve if confidence is predicted.

Later 6DoF metrics:

- normalized translation error;
- orientation error;
- official SPEED+ pose score;
- runtime, memory, and failure rate.

F1 score is not a primary pose metric. Use it only for a clearly defined classification, detection, visibility, or foreground-mask task.

## 7. Mandatory baselines

At minimum, compare:

1. DINOv3 only;
2. VGGT only;
3. simple deterministic score average;
4. best training-free spectral fusion;
5. guided aggregation added to the best training-free fusion;
6. minimal learned fusion, if Track B is activated;
7. flat Tesseract leaf classification or retrieval;
8. hierarchical Tesseract prediction;
9. matched-size Hopf baseline;
10. matched-size cubochoric or another near-uniform \(SO(3)\) grid;
11. greedy versus beam decoding;
12. fixed depth versus adaptive stopping.
13. Tesseract plus tangent residual versus Tesseract plus separate RFM refinement;
14. separate RFM versus flow-induced, parent–child-consistent Tesseract probabilities;
15. coupled posterior with fixed depth versus coupled posterior with adaptive depth.

When comparing rotation grids, match the number of hypotheses and report measured covering radius. A comparison at equal names such as “Level 3” is invalid.

## 8. Experiment isolation rules

1. Assign every experiment a stable identifier: `EXP-000`, `EXP-001`, and so on.
2. Change one main factor per experiment whenever possible.
3. Use the same dataset split, predictor, training budget, seeds, and evaluation code for fusion comparisons.
4. Use at least three seeds for any learned component before calling a gain reliable.
5. Keep failed and negative experiments. Never delete them from the index.
6. A failed run is still a result. Record the failure, traceback location, likely cause, and next action.
7. Do not overwrite a previous run. Create a new run directory.
8. Do not manually type final metrics if they can be exported by evaluation code. Generate machine-readable metrics, then copy or render them into the report.
9. Every table and figure in a report must point to its source file under `outputs/`.
10. No experiment is complete until its Markdown report contains results or an explicit failure record.

## 9. Required output-directory structure

All generated artifacts must be stored under `outputs/`. Do not scatter CSV, TSV, images, predictions, checkpoints, or logs throughout the repository.

```text
outputs/
├── EXPERIMENT_INDEX.md
├── reports/
│   ├── EXP-000_codebook_geometry.md
│   ├── EXP-001_dinov3_baseline.md
│   └── EXP-XXX_short_name.md
├── experiments/
│   └── EXP-XXX_short_name/
│       └── RUN-YYYYMMDD-HHMMSS-seedN/
│           ├── config.yaml
│           ├── command.txt
│           ├── environment.txt
│           ├── status.json
│           ├── logs/
│           ├── metrics/
│           │   ├── metrics.json
│           │   ├── metrics.csv
│           │   └── metrics.tsv
│           ├── predictions/
│           ├── figures/
│           ├── tables/
│           ├── samples/
│           ├── checkpoints/
│           └── caches/
└── comparisons/
    ├── fusion_summary.csv
    ├── fusion_summary.tsv
    ├── rotation_grid_summary.csv
    └── figures/
```

Create only the subdirectories needed by a run, but preserve this organization.

Large immutable feature caches may be shared under `outputs/shared_cache/`. Their filename or manifest must include model name, checkpoint, feature layer, preprocessing hash, dataset split, and code version.

## 10. Mandatory Markdown report for every experiment

Create one report at `outputs/reports/EXP-XXX_short_name.md` for every experiment. Use this template:

```markdown
# EXP-XXX: Descriptive title

## Status
Planned | Running | Completed | Failed | Inconclusive

## Research question

## Hypothesis

## Component under test
Fusion | Predictor | Tesseract grid | Decoder | Filtering | RFM | Posterior distillation | Adaptive depth | Translation | Other

## Track
Training-free | Label-free calibrated | Trainable fusion | Trainable predictor | Full system

## Fixed setup
- Dataset and exact split:
- DINOv3 variant/layer:
- VGGT variant/output/layer:
- Predictor:
- Tesseract depth and codebook size:
- Input resolution and preprocessing:
- Seeds:
- Compute device:
- Git commit:
- Flow base distribution, solver, NFE and endpoint samples, if applicable:

## Changed variable

## Method

## Commands

## Runs
| Run ID | Seed | Status | Runtime | Artifact directory |
|---|---:|---|---:|---|

## Quantitative results
Include the actual metric table here. Do not leave this section as a link only.

## Domain-wise results
Report synthetic validation, lightbox, and sunlamp separately.

## Qualitative results
Embed or link representative successes, failures, glare cases, shadows, small targets, and ambiguous rotations.

## Resource results
- Parameters:
- Trainable parameters:
- Peak memory:
- Mean/median latency:
- Tesseract nodes visited:
- Flow sampling latency and NFE, if applicable:
- Parent-child mass consistency error, if applicable:
- Credible-region coverage, if applicable:

## Comparison with control

## Interpretation
State what the evidence supports. Separate observation from speculation.

## Failure analysis

## Decision
Keep | Reject | Modify | Repeat

## Next experiment

## Artifact index
- Metrics JSON:
- CSV:
- TSV:
- Figures:
- Predictions:
- Checkpoint, if any:
- Logs:
```

The quantitative results must be written into the Markdown report after the run. A report containing only intentions, commands, or links is incomplete.

## 11. Experiment index rules

Maintain `outputs/EXPERIMENT_INDEX.md` as the single overview table:

| ID | Question | Track | Status | Best result | Decision | Report |
|---|---|---|---|---|---|---|

Update it after every run. The index summarizes; it does not replace the separate experiment reports.

## 12. Reproducibility requirements

Every run must save:

- complete resolved configuration;
- exact command;
- random seed;
- Git commit hash and dirty-worktree status;
- Python, PyTorch, CUDA, GPU, and dependency versions;
- dataset split or manifest hash;
- model checkpoint identifiers and hashes when possible;
- start/end timestamps and wall-clock time;
- peak GPU and CPU memory when available;
- machine-readable metrics;
- prediction files sufficient to recompute metrics.

For flow experiments, also save endpoint samples or a reproducible sample manifest, per-level empirical cell masses, solver settings, number of function evaluations, and the random seeds used for base rotations.

Set all relevant random seeds. Record nondeterministic CUDA settings. If exact determinism is disabled for speed, say so in the report.

## 13. Code-quality and testing rules

1. Keep feature extraction, fusion, Tesseract encoding, predictor, decoding, evaluation, and reporting as separate modules.
2. Add unit tests before expensive training.
3. Freeze DINOv3 and VGGT explicitly and assert that their parameters have no gradients in frozen experiments.
4. Detach cached backbone features before the predictor unless a fine-tuning experiment explicitly permits gradients.
5. Keep coordinate-frame conventions documented at every interface.
6. Record quaternion component order, handedness, camera convention, and object-to-camera versus camera-to-object transforms.
7. Validate all predicted quaternions before scoring: finite values, nonzero norm, and unit normalization.
8. Fail loudly on missing data, unknown coordinate conventions, incompatible shapes, or absent checkpoints. Do not silently substitute defaults.
9. Preserve existing working code. Make focused changes and do not rewrite unrelated modules.
10. Never fabricate results, fill missing metrics with zeros, or present example numbers as measured results.

## 14. Phase discipline and experiment order

Run only the required experiments in the current phase. Do not automatically execute every optional method. Use the phase result to decide which branch to investigate next.

### Phase 0 — Data and rotation geometry

- `EXP-000`: immutable loader and coordinate conventions;
- `EXP-001`: ground-truth quaternion to Tesseract path tests;
- `EXP-002`: quantization and covering radius by depth;
- `EXP-003`: Tesseract versus matched Hopf/cubochoric grids.

**Gate:** select the rotation representation and maximum fixed depth.

### Phase 1 — Frozen representation controls

- `EXP-010`: DINOv3-only representation and MLP predictor;
- `EXP-011`: VGGT-token and VGGT-geometry controls;
- `EXP-012`: complementarity and frequency audit.

**Gate:** retain only feature sources that add measurable information.

### Phase 2 — Fusion selection

- `EXP-020`: single-branch controls and equal posterior average;
- `EXP-021`: entropy, margin, confidence, agreement, and spectral weighting tested separately;
- `EXP-022`: best combined training-free fusion;
- `EXP-023`: learned scalar/band gate only if training-free fusion is insufficient;
- `EXP-024`: shallow projection or cross-attention only if simpler fusion fails despite demonstrated complementarity.

**Gate:** freeze one fusion design for predictor experiments. If fusion does not beat the strongest branch, use that branch alone.

### Phase 3 — Fixed-depth Tesseract predictor

- `EXP-030`: MLP hierarchical predictor baseline;
- `EXP-031`: GRU path decoder;
- `EXP-032`: lightweight cross-attentive Tesseract Transformer;
- `EXP-033`: hard versus geodesic-soft targets;
- `EXP-034`: greedy versus beam decoding;
- `EXP-035`: optional tangent residual.

**Gate:** select and freeze the fixed-depth predictor before continuous-flow and adaptive-depth experiments.

### Phase 4 — Continuous and shared-posterior rotation

- `EXP-040`: Tesseract plus tangent residual control;
- `EXP-041`: Tesseract plus a separate, cell-conditioned \(SO(3)\) RFM refiner;
- `EXP-042`: image-conditioned \(SO(3)\) flow and endpoint-sample evaluation;
- `EXP-043`: project the same endpoint samples into per-level Tesseract masses and verify parent–child consistency;
- `EXP-044`: distill flow masses into the hierarchy using GT-path loss plus detached KL supervision;
- `EXP-045`: matched comparison of tangent residual, separate RFM, and the coupled posterior.

**Gate:** retain the coupled model only if it improves multimodal recall, continuous error, uncertainty/coverage, or cross-depth consistency enough to justify its sampling and training cost.

### Phase 5 — Adaptive depth

- `EXP-050`: generate oracle stop/deepen/branch targets using the selected fixed representation;
- `EXP-051`: test entropy, margin, object resolution, DINO–VGGT agreement, VGGT confidence, spectral quality, cell radius, cumulative mass, and within-cell flow spread individually;
- `EXP-052`: rule-based controller;
- `EXP-053`: MLP controller;
- `EXP-054`: Transformer `STOP/DESCEND/BRANCH` controller only if the MLP is limited;
- `EXP-055`: adaptive versus every relevant fixed-depth baseline at matched budgets.

**Gate:** retain adaptive depth only if it preserves accuracy and coverage while reducing nodes, flow calls, latency, or false precision.

### Phase 6 — Rotation validation

- `EXP-060`: synthetic, lightbox, and sunlamp evaluation;
- `EXP-061`: glare, shadow, blur, scale, and occlusion stress tests;
- `EXP-062`: confidence, credible-region coverage, and failure detection;
- `EXP-063`: runtime, memory, sampling, and solver profile.

**Gate:** rotation is the first completed research milestone. Do not begin full 6DoF work until it is reported.

### Phase 7 — Translation and complete 6DoF

- `EXP-070`: automatic localization control;
- `EXP-071`: DINO/fused correspondences with RANSAC-PnP;
- `EXP-072`: VGGT-to-CAD alignment;
- `EXP-073`: learned translation head only if geometric approaches are insufficient;
- `EXP-074`: local multi-hypothesis SE(3) refinement;
- `EXP-075`: official domain-wise 6DoF evaluation.

### Phase 8 — Optional extensions

- compute-aware DINO-first cascade;
- temporal filtering and Tesseract priors;
- learned motion model or temporal flow prior;
- continual adaptation and new-spacecraft onboarding.

These are not part of the first rotation milestone and require separate approval based on preceding results.

## 15. Decision rule for choosing the fusion method

Choose the simplest method on the accuracy–robustness–cost Pareto frontier.

- Retain training-free fusion if it matches or approaches learned fusion across domains at materially lower complexity.
- Use learned fusion if it produces a repeatable and meaningful gain, especially on lightbox and sunlamp, and the added cost is acceptable.
- Do not choose a method solely because it has the best synthetic validation result.
- If fusion does not outperform the best single backbone, report that result and do not force fusion into the final architecture.

## 16. Agent reporting behavior

The agent must:

1. state the planned experiment ID before running it;
2. create its Markdown report immediately with status `Running`;
3. save all generated outputs under that experiment's run directory;
4. update the report with actual results after evaluation;
5. update `outputs/EXPERIMENT_INDEX.md`;
6. summarize what changed, what was measured, what failed, and what should happen next;
7. ask before making a major scope or architecture change;
8. never announce success based only on a completed training loop—evaluation and the written report are required.
