# Method figures (manim)

These are the explanatory figures in the paper. They are drawn with manimgl from `video/paper_figures/` and use the same colour code as the other figures: GT is green, prediction is magenta, frozen modules are grey and dashed with a snowflake icon, and trained modules are coloured with a flame icon.

| File | In the paper | Shows | Idiom borrowed from |
|---|---|---|---|
| `architecture` | Fig. 2, start of Sec. 3 (`fig:pipeline`; replaces the matplotlib `figures/pipeline.pdf`) | the general method for any known object: shared frozen backbones, cached token maps, per-object trained trees and translation block; training flow (3 views with exact roll labels, multi-path loss, optional distillation) and test flow (4 rolled views, per-level PoE, beam search, orbit voting) | GDRNPP, FoundPose and SurfEmb overview figures; snowflake/flame marks for frozen/trained modules |
| `teaser` | Fig. 1, under the title (`sec/teaser.tex`) | (a) frozen shared backbones feeding small per-object heads, with median-error test predictions on all four benchmarks; (b) rotation as a coarse-to-fine path; (c) orbit voting repairing a 159° flip | ZebraPose, CheckerPose and SpyroPose teasers: one concrete visual of the core idea |
| `tesseract_partition` | Sec. 3.3 (`fig:tesseract`) | (a) chart choice; (b) nested cells of levels 1–5 in the projected cube, with a magnified inset; (c) the path digits, cell counts and per-level error | CheckerPose Fig. 3 and ZebraPose Fig. 2: level index written under each level |
| `tree_decoder_multipath` | Sec. 3.3 (`fig:tree`) | (a) the tree Transformer as implemented; (b) beam search with beam 8 (which nodes are kept is schematic); (c) multi-path targets for a real T-LESS label | GDRNPP and CheckerPose block diagrams; HiPose colour-coded paths |
| `orbit_voting` | Sec. 3.6 (`fig:orbit`) | the 4 rolled views, each view's top hypothesis mapped back, the orbit score of all 32 candidates, and the result | CheckerPose (a)/(b)/(c) panelled flow (no direct precedent for orbit voting) |

## Where the content comes from
Nothing in these figures is illustrative except the kept nodes in `tree_decoder_multipath` (b).

- **Geometry:** chart, cells, path digits and level errors are computed at render time with `src/tfpose/tesseract.py`, via `video/common.py`.
- **Example images, overlays, candidate scores and errors:** these come from the saved predictions of the final systems.
  - `video/paper_figures/prep_assets.py` writes them to `video/paper_figures/assets/`.
  - It re-uses the drawing code of `scripts/fig_qualitative.py`.
- **Example selection:** each example follows the same fixed rules as `paper/figures/qualitative` (see `selection.csv`).
  - Teaser images: the median-error test image per benchmark.
  - SPEED+: the orbit example.
  - Multi-path example: the T-LESS object-5 image of the teaser.
  - Architecture (`prep_arch.py`): the first object-5 T-LESS PBR training crop with visib_fract > 0.95 and its two cached augmented copies, re-created with the extraction code (homography roll at the cached angle, photometric rng([seed, row])); the teaser T-LESS test image, its homography-rolled views and its cached features (DINOv3 8x8 tokens as a per-image PCA image; MoGe-2 normals under the consensus mask).

## Rebuild
```
.venv/bin/python video/paper_figures/prep_assets.py      # data assets (project venv; reads the prediction npz files)
.venv/bin/python video/paper_figures/prep_arch.py        # architecture-figure assets (reads the T-LESS images and feature caches)
bash video/paper_figures/render_all.sh                    # conda env manim311; writes PNG and PDF here, trimmed
```
The output is raster (PNG, about 3,600 px wide, printed at full text width), because manimgl renders through OpenGL. A PDF copy of each PNG is written alongside it.
