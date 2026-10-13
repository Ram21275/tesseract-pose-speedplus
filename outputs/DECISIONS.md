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
