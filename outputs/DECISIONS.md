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
