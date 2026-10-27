# Decision log

Architecture, scope and split decisions that change the plan in `TESSERACT_FUSION_EXECUTION_PLAN.md`. Each entry records what was decided, by whom, and the evidence behind it.

## DEC-001 — MoGe-2 replaces VGGT as the geometric branch (2026-10-08)
- **Decided by:** the user ("update the code to have moge-2 for now"). The recommendation came from EXP-013/014.
- **Change:** the geometric branch is now MoGe-2 (`Ruicheng/moge-2-vitl-normal`, MIT) instead of VGGT-1B. It is configured in `configs/branches.yaml`. VGGT code and results (EXP-011/012) are kept for reproducibility but retired from new experiments.
- **Evidence (synthetic validation only):**
  - Better single branch: mean error 39.6 ± 0.6° vs VGGT 48.7 ± 0.7°; median 21.2° vs 25.4°.
  - About equally complementary to DINOv3-L: oracle-of-two acc@10 0.289 ± 0.020 vs 0.278 ± 0.022.
  - Slightly stronger agreement cue: 3.2× vs 2.6× (3.0× for VGGT grid4).
- **Open caveats:**
  - A non-geometric second branch (DINOv3-B) was *more* complementary (oracle 0.308). Geometry-specific value is therefore unproven; it is to be re-tested at full data and in Phase 2 against that control.
  - MoGe-2's foreground mask fails on lightbox and sunlamp images (EXP-013).
- **Scope note:** the project title in the ground-rules file still says "DINOv3–VGGT". The rules file itself is left unchanged. This log is the authoritative record of the substitution.

## DEC-000 — Split policy (2026-10-05)
- **Decided by:** the user.
- **Policy:**
  - train only on SPEED+ synthetic/train;
  - use synthetic/validation for model selection;
  - lightbox and sunlamp are test-only and never used for selection or tuning.

## DEC-002 — Experiment numbering (2026-10-09)
- **Why:** user-requested side experiments EXP-015–020 used IDs that collide with the plan's Phase-2 IDs (EXP-020–024).
- **Rule:**
  - The plan's Phase-2 experiments will be numbered **EXP-025–029** (plan EXP-020 → 025, 021 → 026, 022 → 027, 023 → 028, 024 → 029). Phase 3 onwards keeps the plan's numbers.
  - New user-requested side experiments use the **EXP-1xx** series, starting with EXP-101.
  - Existing IDs are never renamed.

## DEC-003 — Phase-2 gate: fusion design frozen (2026-10-09)
- **Decision:** the representation for Phase 3 is the **training-free product of experts** of two Tesseract probes:
  - DINOv3-L `grid4`;
  - MoGe-2 `normals16~and` (normals masked by the EXP-015 DINO∧MoGe-2 consensus).
- **Evidence (synthetic val):** PoE 25.9 ± 0.5° vs DINOv3 alone 27.7 ± 0.8° (EXP-025). No reliability weighting improves on equal weights (EXP-026). Real domains (descriptive): lightbox 62.5° vs 76.1°, sunlamp 71.4° vs 79.1°.
- **Skipped plan items:**
  - EXP-027 (no signal helped individually);
  - EXP-028/029 (learned gates and cross-attention are conditional on training-free fusion failing; it did not).
- **Caveat:** all Phase-1/2 numbers come from the 6k-image subset. Phase 3 starts with a full-data repeat of the frozen design (EXP-030).

## DEC-004 — Phase-3 gate (2026-10-09)
- **Frozen predictor:** MLP Tesseract tree with **geodesic soft targets** (EXP-033), **greedy** decoding (EXP-034), no tangent residual (EXP-035), and equal-weight PoE fusion (DEC-003).
- **Evidence:** PoE synthetic val 6.068 ± 0.067° vs EXP-030 6.242 ± 0.072°. Among the pre-registered variants only EXP-033 passed; EXP-031 (GRU) missed by 0.005°.
- **Open question for the user (selection policy):**
  - Under soft targets, DINOv3 *alone* has the lower synthetic-val error (5.69° vs PoE 6.07°).
  - PoE is far better on the real domains: lightbox 28.2° vs 37.0°, sunlamp 42.1° vs 48.1°.
  - DEC-000 forbids selecting on lightbox or sunlamp. Keeping PoE therefore rests on DEC-003 (EXP-025/030), not on EXP-033.
  - Options:
    - (a) keep PoE per DEC-003;
    - (b) re-select by synthetic val (DINOv3 alone);
    - (c) record a real-domain validation split (e.g. a held-out part of lightbox) and select on it, reporting sunlamp and the rest of lightbox as test.
- **Not yet tested:** soft targets + GRU decoder combined. Each helps the single branches; it would need a new pre-registered experiment.

## DEC-004 update — Phase-3 gate re-decided after EXP-032 and EXP-036 (2026-10-09)
- **Frozen predictor:** the **Transformer decoder + geodesic soft targets** (EXP-036), greedy decoding, PoE fusion (DEC-003). PoE synthetic val is **5.540 ± 0.063°**, the lowest of EXP-030–036. It replaces EXP-033.
- **Selection-policy question still open:** with the winner, DINOv3 alone is below PoE on synthetic val (5.35° vs 5.54°), but PoE is far better on lightbox (26.9° vs 37.1°) and sunlamp (36.6° vs 41.4°). The real-domain validation split chosen by the user awaits their clarification and has not been applied.

## DEC-005 — Real-domain validation split (2026-10-10)
- **Decided by:** the user, choosing option (c) of the DEC-004 selection-policy question ("Real-domain val split").
- **Why now:** the Phase-4 gate judges multimodal recall and coverage. Synthetic val has almost no errors above 90° (0.1% for EXP-036 DINO seed 0, vs 16% on lightbox and sunlamp), so it cannot register that benefit.
- **Split:** `lightbox_val` = a seeded 20% of lightbox (1,348 images): sorted names, `numpy.random.default_rng(0).choice(k=round(0.2·N), replace=False)`. List: `outputs/data_manifests/lightbox_val_DEC005.txt` (sha `57b45ce4262ce3a4`); code: `tfpose.data.eval_groups`. `lightbox_test` = the other 5,392 lightbox images. Sunlamp (2,791) stays test-only.
- **Leakage check:** lightbox has no sequence structure (consecutive names are a median 132° apart; nearest-neighbour rotation distance has median 7.1° and is never below 1°), so a random split is used.
- **Use:** from Phase 4 on, selection and gates use synthetic val **and** `lightbox_val`, with the rule pre-registered per experiment. Only `lightbox_test` and sunlamp are reported as untouched tests. Earlier experiments keep their reported numbers. Their full-lightbox figures were not used for tuning, but they overlap `lightbox_val`.

## DEC-006 — DEC-005 withdrawn; strict split policy restored (2026-10-10)
- **Decided by:** the user. They clarified that lightbox and sunlamp must stay test-only and chose "Strict: synth val only".
- **Policy:** DEC-000 again applies unchanged:
  - train on synthetic/train only;
  - select and gate on synthetic/validation only;
  - lightbox (all 6,740) and sunlamp are test-only and descriptive.
- **Effect:** DEC-005 never influenced any training run or checkpoint. Phase-4 heads train on synthetic/train and select checkpoints on a synthetic-val sample, and their training-noise width comes from synthetic-val errors. Its only effect was the pre-registered Phase-4 gate, which is rewritten here before any result. Per-run files still list `lightbox_val` / `lightbox_test` rows from the DEC-005 split; these are purely descriptive, and fused evaluations report the whole of lightbox.
- **Known limitation:** synthetic val has almost no gross (> 90°) errors, so gains in multimodality or flip recovery can only show up in the descriptive lightbox and sunlamp numbers. Those cannot be used to select.

## DEC-007 — Two plan additions: EXP-029 (PanSt3R-inspired fusion) and EXP-064 (broader objects) (2026-10-10)
- **Decided by:** the user (the request text is the specification). IDs were allocated under DEC-002 by phase slot.
- **EXP-029** = the plan's Phase-2 EXP-024 slot ("shallow channel projection", → EXP-029 under DEC-002). DEC-003 had skipped it because training-free fusion did not fail; the user has now requested it explicitly, so it is reopened.
  - **Order:** the Phase-4 runs already started (EXP-040–042) complete first, as the user asked. EXP-029 runs next, then EXP-043–045 in numeric order, all before Phase 5.
  - **Controls:** Phase-4 results stay as controls on the DEC-003/DEC-004 representation.
- **EXP-064** = the next free Phase-6 ID (after EXP-060–063). It runs near the end of Phase 6. Status: **paused until the user adds YCB-V and T-LESS (BOP)**. A search of the local disks found neither (only an LM-O copy in another user's folder, which is not used).
- **Unchanged:** the cancelled LoRA experiment (EXP-102) stays cancelled.

## DEC-008 — After EXP-029: training-free PoE kept; learned feature fusion not adopted (2026-10-13)
- **Decided by:** the user ("continue") after the recommendation to keep PoE. The representation is unchanged (DEC-003/DEC-004).
- **Evidence:**
  - EXP-029 concat + MLP passed its literal pre-registered rule vs the 8×8 PoE on synthetic val (5.341 ± 0.031° vs 5.582 ± 0.103°).
  - But it equals DINOv3-alone (5.345°), so the geometry tokens add nothing measurable.
  - It is much worse on the real domains (descriptive: lightbox 40.8° vs 27.9°, sunlamp 46.2° vs 37.9°).
  - Adopting it on synthetic val alone would contradict ground rule 15.
- **Consequence:** Phase-4 results (EXP-040–042) stay matched to the representation in use. EXP-043 onwards uses the same PoE representation, so no new controls are needed.

## DEC-009 — EXP-064 datasets: T-LESS, SYMSOL-T and YCB-Video; data located (2026-10-13)
- **Decided by:** the user ("for exp 64 use tless symsol t and ycb"). SYMSOL-T is added to the two planned datasets.
- **Correction to DEC-007 / the EXP-064 report:** the 2026-10-10 search that found neither YCB-V nor T-LESS missed `/media/kavinder/hdd2`. All three datasets are there, in the user's own folders (read-only use):
  - **T-LESS (BOP):** `hdd2/liepose_data/bop_datasets/tless` (HF mirror `bop-benchmark/tless`). `train_primesense` (30 objects, real), `train_pbr` (50 scenes), `test_primesense` = the **BOP19 subset only** (20 scenes × 50 = 1,000 images, 4,904 targets), CAD / reconstructed / eval models with symmetry metadata.
  - **SYMSOL-T:** `hdd2/liepose_data/symsolt`. Five symmetric shapes (tet, cube, icosa, cone, cyl) with random translation (SYMSOL-T as in LiePose), 224², rendered **locally** (`synth_train.log`, seeds 12345 / 82973), so not guaranteed identical to the published sets. `test-5k` (5k per shape), `train-45k`, `train-225k` and `train-45k-synth`. The `config.yaml` of `train-45k` / `train-225k` lists only `tet` though all five shape folders are present; shape counts will be taken from `annotations.gz`, and whether the two train folders duplicate each other will be checked. Annotations carry `rotations_equivalent` (discretized for cone / cyl).
  - **YCB-Video:** `hdd2/FFb6D/datasets/ycb/YCB_Video_Dataset`, all 92 sequences + `data_syn` + models + keyframes. This is the **original PoseCNN release, not the BOP version** the EXP-064 protocol names: a recorded deviation. Test = the keyframes of sequences 0048–0059 (`dataset_config/test_data_list.txt`, 2,948 entries; to be checked against `image_sets`), which stay out of training and validation. The zip archives (and a `.part` download) in the parent folder are ignored.
- **Not reused:** `liepose_data/tless_crops`, `tless_tokens`, `bop_cache*` (another pipeline; allocentric rotations of unknown convention). EXP-064 builds its own crops and manifests; egocentric vs allocentric rotation is a convention to pre-register.
- **Order unchanged:** EXP-064 stays in its Phase-6 slot (after Phase 5 and EXP-060–063), per the run-by-number rule. Status changes from "paused for data" to "data located; waiting on Phase-6 prerequisites". Dataset paths are added to `configs/paths.yaml`; nothing is extracted or cached yet (hdd2 is 97% full, root has ~100 GB free).

## DEC-010 — Add EXP-105 (SPEED+ best-fit selection) and EXP-106 (T-LESS multi-path tree) (2026-10-13)
- **Decided by:** the user. Request: "try the idea on t-less for keeping corners as rotations ... after prediction if there would be same flips in the answer we can extract the best fit ... do this on speed+ as well so that we can compare with other baselines"; "you can try render and compare and flow matching or diffusion based methods to get to the best fit"; "there is no 3d file take synthetic for comparison. add flow or diffusion based experiment for best selection. for t-less propose something for similar looking orientations there should be different tree paths or probability chain paths".
- **EXP-105 (SPEED+):** candidate expansion (180° body-axis flips, leaf-cell corners) and best-fit selection by tree, flow density, diffusion re-noise / return, and synthetic-template comparison. No SPEED+ 3-D model exists, so synthetic-train images stand in for renders. The proposed EXP-103 ("rotate and match") is merged into it as the template arm.
- **EXP-106 (T-LESS):** a multi-path tree (look-alike orientations get separate tree paths) vs single-path and canonical labels, then best fit by CAD render-and-compare and flow/diffusion.
- **Order:** the running EXP-044 chain finishes first; then EXP-105, then EXP-106, before EXP-045. T-LESS work is pulled ahead of EXP-064's Phase-6 slot by this request; EXP-064 is unchanged.
- **Diffusion:** deferred earlier ("later, after the original plan"); now requested for best-fit selection. On SO(3) the diffusion score equals the flow-matching target, so the EXP-042 flow serves as the diffusion model; no separate SO(3) diffusion model is trained.
- **Known caveat:** EXP-104 found that failures do not concentrate at chart seams, so corners are tested as a separate arm (precision vs flip repair) rather than assumed to cause the flips.
- **Addendum (2026-10-13, before any run):** the user clarified that the "multi-path tree" they meant is the **original Kurz et al. (2017) formulation** (corner representatives, all 8 facets, so one rotation has many redundant paths), not only the post-hoc corner candidates and symmetry paths first pre-registered. User's choice: "Add as extra arm". Arm K is added to both EXP-105 and EXP-106; the existing arms stay. A correction was given: q and −q are the same rotation, not a 180° flip, so the expectation that the redundancy alone fixes flips is stated as unlikely; the arm tests it.

## DEC-011 — EXP-106 on hold; SPEED+ phases first, cross-dataset work last (2026-10-13)
- **Decided by:** the user: "making 106 hold we will move forward with the other phases and then after we have all our findings we will run on other datasets that includes 064 exp as well."
- **Order:** EXP-105 (running, including arm K) → EXP-045 → Phase 5 (EXP-050–055) → Phase 6 SPEED+ experiments (EXP-060–063) → cross-dataset validation: EXP-064 (T-LESS, SYMSOL-T, YCB-Video) together with EXP-106 (T-LESS multi-path tree).
- **What stays:** EXP-106's pre-registration (with arm K) is unchanged and resumes in the cross-dataset block. Its uncommitted manifest-builder draft (`scripts/exp106_build_manifest.py`, crashed before finishing) is not used until then.
- **Goal behind it (user):** show that the method generalizes beyond the spacecraft. The recipe is to be locked on SPEED+ first, then applied to other objects, retraining only the heads.

## DEC-012 — Run-order amendments, paper-style visuals/metrics, one-week target (2026-10-13)
- **Decided by:** the user: "you can add these points and run after the experiment 105 or if I tell"; "we should have visualisations like the ones there in lie pose diffusion to show our flow matching approach and other things, also similar evaluation metrics if we need to compare with other models and also fps, steps and other things. Lets try to complete till generalisation this week. keep 102 cancelled for now".
- **Adopted (each starts after EXP-105 finishes, or earlier only if the user says):**
  1. The cross-dataset block (EXP-064 + EXP-106) runs at the **end of Phase 6**, i.e. inside the first rotation milestone, not after Phases 7–8.
  2. **Recipe lock:** a decision entry after Phase 6 SPEED+ work and before any cross-dataset run. It fixes backbones, features, depth, fusion, flow on/off, any EXP-105 selector, recipe and budget. Nothing is tuned per dataset.
  3. **Early data pipelines:** CPU-only manifests and conventions for T-LESS, SYMSOL-T and YCB-V (RGB loading, allocentric rotation, symmetry sets, crops). Correctness checks only; no model results.
  4. **Phase 5 reduced** to the oracle check EXP-050. EXP-051–055 run only if EXP-050 shows a meaningful adaptive-depth headroom. Reason: errors are dominated by 180° flips, not depth (EXP-104, EXP-044).
- **Added:** paper-style visualizations and comparison metrics in the LiePose style (SO(3) distribution plots, sampling trajectories, accuracy vs steps, FPS/latency), as a separate analysis experiment pre-registered before its first run.
- **Unchanged:** EXP-102 (LoRA) stays cancelled ("might take too much time and only give us a trivial positive result").
- **Timeline target:** generalisation done within the week of 2026-10-13. This is tight on a shared GPU; if needed, scope is cut in a pre-stated order (see the next status report), never by dropping pre-registered criteria.

## DEC-013 — Add EXP-108 (direct path flow); real-domain improvement ideas proposed (2026-10-15)
- **Decided by:** the user: "continue and add an experiment direct path flow that iteratively revises the root and child decisions themselves using image and rotation geometry"; "we can wait" (our queue waits for GPU memory rather than lowering the guard).
- **EXP-108:** discrete flow matching on the tree path, with a geometry-aware corruption kernel (180° flips, nearby cells, uniform). Revisers per branch, fused by PoE, iterative top-down re-decoding. Runs after the queued EXP-050 / EXP-107 jobs and **before Phase 6**, because it could change the recipe at the lock.
- **Proposed to the user, not yet added:** to cut real-domain error, (A) style/lighting augmentation of the synthetic training images, (B) 16×16 DINOv3 tokens, (C) in-plane test-time orbit voting, (D) a flip verifier trained on augmented features. Unlabelled real-image adaptation is not proposed for use, because DEC-006 keeps lightbox / sunlamp test-only.

## DEC-014 — EXP-109 (augmentation A + 16×16 tokens B + orbit voting C); run order (2026-10-15)
- **Decided by:** the user: "if you want to generalise the model for real world images in proposed A you can add augmentations to the train set while training, we can test a new transformer with the augmentations. add these A+B+C as an experiment. run 108 first and the other experiments when I tell you to. also I first need the generalisation claim once we are done with rotations and a final architecture. so as soon as 050 finishes we should test transformer once with augmentations."
- **Clarified by the user (question asked):** order **EXP-050 → EXP-109 → EXP-108 → EXP-107 jobs (last)**. Phase 6, EXP-064 / EXP-106 and anything else wait for the user's go.
- **Generalisation claim:** produced after the rotation work and the final architecture (recipe lock), as already planned (DEC-011/012).
- **Selection stays DEC-006-compliant:** EXP-109 selects on a synthetic val-aug proxy and on clean synthetic val; real domains are descriptive.

## DEC-015 — EXP-108 becomes real discrete flow matching; baselines dropped (2026-10-15)
- **Decided by:** the user: "make 108 as the real flow matching experiments I do not want to give time to other baselines now we already have enough and then we also need to run on other datasets as well."
- **Change (before any EXP-108 run):** the iterative path denoiser and its no-geometry matched control (review 2026-10-15) are **not run**. Their code is kept and disabled. EXP-108 is now discrete flow matching on the Tesseract path: image-conditional coupling (x0 ~ frozen PoE tree, x1 = GT), factorized mixture probability path κ(t) = t, a time-conditioned posterior model, and a CTMC Euler sampler. Pre-registration amended accordingly.
- **Consequence, recorded honestly:** the review's matched control is dropped by the user's choice. So whether the geometry token adds value **cannot be tested**, and EXP-108 makes no claim about it.
- **Next:** cross-dataset runs (EXP-064 / 106) follow the SPEED+ work. They still need the recipe lock (DEC-012), which the user decides.

## DEC-016 — EXP-110 weighted PoE (2026-10-15)
- **Decided by:** the user: "you can also test different weights for moge and dino decisions while PoE."
- **EXP-110:** per-level weighted PoE of the frozen EXP-036 trees, w ∈ {0, 0.2, …, 1}. Selected on the EXP-109 synthetic val-aug proxy, because clean synthetic val is known to favour DINOv3 alone; real domains are descriptive. Runs after EXP-108 in the same chain.

## DEC-017 — EXP-111: DINOv3-only augmented tree + flow; runs before EXP-108 (2026-10-16)
- **Decided by:** the user: after the question "are the above results of the same architecture" (answer: the EXP-109 A numbers are the DINOv3 tree alone, without MoGe-2, PoE or flow): "yes pre-register it and run it before 108 it should be a quick experiment."
- **EXP-111:** a flow head with the EXP-042 recipe on the EXP-109 arm-A DINOv3 tree, trained with the same augmented views, 3 seeds, plus DINOv3-only timing. Queue order: EXP-109 eval (running) → **EXP-111** → EXP-108 → EXP-110 → EXP-107.
- **Reordered (user, 2026-10-16):** "move experiment 111 after the old scheduled runs". Queue: EXP-109 eval → EXP-108 → EXP-110 → EXP-107 → **EXP-111** (KDE + verifier arms).

## DEC-018 — Recipe lock: EXP-109 methodology, PoE and DINOv3-only variants (2026-10-16)
- **Decided by:** the user: "for any further pending experiments we will use the architecture or methodology used in 109 best result. and it should contain both poe and single dinov3 as we can show it for comparing time overhead with and without MOGE2."
- **Locked recipe** (the DEC-012 recipe lock, closing the SPEED+ architecture choice):
  - frozen DINOv3-L `tokens8` (+ MoGe-2 `normals16~and` in the PoE variant), GT square crop;
  - one Tesseract tree Transformer per branch (EXP-036 architecture and budget), trained with the EXP-109 arm-A augmentation (2 augmented copies: photometric + in-plane rotation, labels q′ = Exp(−θ·d) ⊗ q);
  - inference: the per-level PoE across branches (equal weights) in the PoE variant, or the single DINOv3 tree;
  - followed by the EXP-109 **C** in-plane orbit voting (4 views, beam-8 candidates mapped back, product of tree likelihoods).
- **Two variants** are reported everywhere: **PoE (DINOv3 + MoGe-2)** and **DINOv3-only**, with and without orbit voting, for the accuracy / time-overhead comparison.
- **Not part of the recipe:** 16×16 tokens (EXP-109 B), the weighted PoE (EXP-110), DFM (EXP-108), distillation (EXP-044), arm K (EXP-105). The flow / verifier is optional and judged in EXP-111.
- **Consequences:**
  - EXP-112 (new; evaluation only) evaluates both variants with end-to-end timing.
  - EXP-064 / EXP-106 and any later experiment use this recipe.

## DEC-019 — Experiments to strengthen the contribution: EXP-113, 114, 115, then EXP-064 (2026-10-16)
- **Decided by:** the user: "we need the generalisation claim and comparison of this architecture with Final Tesseract versus matched Hopf/flat classifier, both with identical augmentation and orbit inference … Same four views and candidate pool: your orbit-product score versus maximum confidence and simple rotation-cluster voting … Photometric-only, rotation-only, combined; each with/without orbit inference … And the exp 64 for knowing if this extends beyond tango. Lets focus on these points."
- **Added:**
  - EXP-113: Tesseract vs a matched flat Hopf classifier;
  - EXP-114: orbit scoring rules;
  - EXP-115: augmentation ablation.
  - All use the locked recipe (DEC-018) and the shared evaluator `scripts/orbit_eval.py`.
- **Order:** after EXP-111 / 112: 114 (evaluation only) → 113 (6 flat trainings) → 115 (extraction + 12 trainings) → evaluations → **EXP-064** (cross-dataset; its implementation follows its amended protocol).
- **Focus:** other optional work (EXP-051–055, EXP-107 re-run, EXP-106) is deferred.

## DEC-020 — Caches moved to the 8 TB drive; obsolete caches deleted (2026-10-16)
- **Decided by:** the user: "move the cache embeddings or tokens that are taking space in the main ssd to 8 tb drive inside a folder name it RAM … and change the paths wherever they are used"; "/media/kavinder/hdd2/RAM this is the directory you can use this for the space intensive things"; deletion approved ("yes you can delete them").
- **Done** between jobs (queue paused via `outputs/experiments/QUEUE_PAUSE`, no job interrupted):
  - `outputs/shared_cache` (105 GB) → `/media/kavinder/hdd2/RAM/shared_cache`;
  - compact caches → `/media/kavinder/hdd2/RAM/compact`;
  - verified (identical file sizes, no broken links, a real read check).
  - Paths are now set in `configs/paths.yaml` (`cache_root`, `compact_cache_root`) and read through `src/tfpose/cachepaths.py`. `outputs/shared_cache` remains as a link for old commands. Script: `scripts/migrate_cache_to_hdd2.sh`, log `outputs/experiments/migrate_cache.log`.
- **Deleted (55 GB):**
  - `tokens16_full_std.npy` (rejected EXP-109 arm B; re-creatable with `scripts/build_tokens16_memmap.py`);
  - all `subset_v1` caches (Phase 1–2 sanity runs: VGGT, DINOv3-B / L, MoGe-2, CaSS, spectral parts, consensus masks v1; re-creatable with `scripts/extract_features.py --subset v1` and the EXP-015 / 017 / 018 scripts).
  - Consequence: EXP-109's B arm and Phase 1–2 cannot be re-evaluated without re-extracting. Their results and run records are unchanged.
- **Free space after:** SSD 145 GB, 8 TB drive 104 GB.
- **Follow-up fix (2026-10-16 15:40):**
  - After the move, `featsets.load` failed: it recorded the cache path relative to the repository, and the new location is outside it.
  - That broke EXP-111 seed 2, EXP-112 (both blocks), EXP-114 and the EXP-113 trainings (about 14:00). The failed runs are kept as records.
  - **Fixed** (absolute path when outside the repo; the same pattern was removed from three metadata writers).
  - All loader paths were verified (full caches, augmented and rotated views, `aug.csv`).
  - The failed blocks are re-queued in `scripts/run_queue6.sh`. EXP-115 feature extraction was unaffected and continues.

## DEC-021 — Pre-register EXP-116 (roll-consistent tree) and EXP-117 (orbit distillation); EXP-064 paused for disk (2026-10-17)
- **Decided by:** the user: "you can pre register idea 1 and 2 also we will be using the 8tb drive so either I will free up some space or need to delete some cache will let you know keep it paused for now."
- **EXP-116:** geometry-transported hierarchical consistency under known camera roll, with controls (paired-view augmentation λ = 0; leaf-only equivariance).
- **EXP-117:** single-view student distilled from the orbit teacher over the existing training views.
- Neither needs new features or disk. They run after EXP-115.
- **EXP-064** stays paused until the user frees space on the 8 TB drive (its caches must go there).

## DEC-022 — Queue order EXP-064 → EXP-117; EXP-064 zero-shot arm; EXP-118 6-DoF translation (2026-10-19)
- **Decided by:** the user, in three messages:
  - "run 117 after 64 first write it and complete it."
  - "you can keep the same recipe but do one thing test also on the frozen tessaract transformer to test zero shot accuracies. resume the tasks that we paused."
  - "after all these queue the translation test with the space-hop formulation keep it same … it will make the paper for 6dof."
- **EXP-064:**
  - per-object trees, unchanged recipe;
  - added a zero-shot arm: the frozen SPEED+ EXP-109 trees with no retraining, metrics pre-set (raw, chance, frame-free pair error, 3-DOF calibrated), CPU job;
  - a shared one-model-per-dataset variant was discussed and is parked (the user kept the recipe).
- **EXP-118 (pre-registered):** SPACE-HOP translation formulation kept identical (normalized box offsets U_x, U_y; depth U_z with α = 1.6; MSE). Translation block on the tree's token encoder, per branch.
  - Arms: joint, frozen tree, and a box-only baseline that measures the ground-truth crop's own translation content.
  - Queued after EXP-117.
- **Order update (user, 2026-10-19: "do 117 after 118"):** EXP-064 → EXP-118 → EXP-117 (`scripts/run_queue12.sh`).

## DEC-023 — Locked recipe unchanged; EXP-117 recorded as a successful enhancement (2026-10-21)
- **Decided by:** the user: "we keep the locked recipe and add this distillation as a successful experiment that enhances the results."
- **DEC-018 (EXP-109 methodology: PoE and DINOv3-only variants, orbit inference) stays the locked recipe.**
- **EXP-117 (orbit-consistency distillation)** is reported as a successful add-on that improves both the single-view tree and the orbit system:
  - PoE + orbit 3.69 / 11.21 / 11.09° vs 3.87 / 12.30 / 12.25°;
  - student greedy val-aug 9.22 vs 9.75.
  - It is not part of the recipe used by later experiments unless the user says so.

## DEC-024 — Cross-dataset fixes 1–3: EXP-122, EXP-106 (amended), EXP-119; ID numbering (2026-10-27)
- **Decided by:** the user: "do 1,2,3", answering the proposal of how to make the method work across benchmarks:
  1. YCB-V: add the synthetic `data_syn` renders to training;
  2. T-LESS (and SYMSOL-T): train on all symmetry-equivalent orientations (EXP-106 multi-path arm);
  3. orbit distillation on T-LESS and SYMSOL-T.
- **IDs:**
  - **EXP-122** = YCB-V with synthetic training data. It is a new experiment, not an EXP-064 amendment; the EXP-064 YCB-V failure stays reported.
  - **EXP-106** = the existing pre-registration, run now with the amendment of 2026-10-27 (written before any run).
  - **EXP-119** = cross-dataset orbit distillation.
  - The three on-hold ideas take later IDs when the user approves them: EXP-120 (3-D geometry into DINO), EXP-121 (smaller backbones / Jetson / C-RADIOv4), EXP-123 (DINO / MoGe flip-confusion probe). Earlier chat messages also called the probe "EXP-119"; this entry supersedes that.
- **Order:** EXP-122 → EXP-106 → EXP-119 (EXP-119's base arm depends on EXP-106). `scripts/run_queue13.sh`.
- **GPU stages wait** until the SPACE-HOP SymSol training (another project's job) has finished. CPU stages (manifests, composites, code checks) run now.
- The locked recipe (DEC-018) is unchanged.
