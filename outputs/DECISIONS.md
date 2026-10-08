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
