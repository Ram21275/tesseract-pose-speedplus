# Tesseract pose estimation on SPEED+

Spacecraft rotation estimation from frozen foundation-model features, predicted as a path through a hierarchical **Tesseract** discretization of SO(3).

The Tesseract follows the recursive tesseract subdivision of Kurz, Pfaff & Hanebeck, MFI 2017. This project uses cell centres and 4 antipodal charts so that every rotation has a unique root-to-leaf path. See `outputs/reports/EXP-001_gt_tesseract_paths.md`.

- **Plan and rules:** `TESSERACT_FUSION_EXECUTION_PLAN.md`, `TESSERACT_FUSION_AGENT_GROUND_RULES.md`
- **Results:** `outputs/EXPERIMENT_INDEX.md`, with one report per experiment in `outputs/reports/`
- **Decisions:** `outputs/DECISIONS.md`. Current branches are set in `configs/branches.yaml`:
  - **primary:** DINOv3 ViT-L/16, 4×4 patch grid;
  - **geometric:** **MoGe-2** ViT-L (normal), which replaced VGGT under DEC-001.

## Setup
```bash
python -m venv --system-site-packages .venv            # torch 2.11 / CUDA from the base env
.venv/bin/pip install --no-deps "git+https://github.com/microsoft/MoGe.git"
.venv/bin/pip install --no-deps "git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37"
.venv/bin/pip install --no-deps "git+https://github.com/facebookresearch/vggt.git"  # only to reproduce EXP-011/012
```
Dataset paths are set in `configs/paths.yaml`. The dataset is read-only and nothing is ever written into it.

## Running (always through the memory guard)
The machine is shared, so every heavy job runs through `scripts/guarded.sh`:
- a cgroup memory cap with no swap, and TasksMax 4096;
- an in-process watchdog at 85% of the cap;
- a reaper that marks killed runs as failed.

Run long jobs inside tmux.
```bash
.venv/bin/python -m pytest -q                             # unit and guard tests
scripts/run_phase1.sh v1 subset                           # Phase-1 controls: DINOv3 + MoGe-2 (+ DINOv3-B control)
scripts/guarded.sh 24 -- .venv/bin/python scripts/exp012_complementarity.py --run-exp EXP-014_moge2_complementarity
```

## Layout
- `src/tfpose/`:
  - `rotations`, `tesseract` (encode/decode), `grids` (Hopf, Super-Fibonacci);
  - `data` (SPEED+ loader and conventions), `features` (DINOv3 / MoGe-2 / VGGT extractors), `featsets`;
  - `predictor` (hierarchical MLP and beam search), `metrics`;
  - `runlog` (run directories and provenance), `memguard`.
- `scripts/`: experiment entry points and drivers.
- `tests/`: geometry, predictor and guard tests.
- `outputs/`: experiments, reports and comparisons. Feature caches and checkpoints are gitignored.
