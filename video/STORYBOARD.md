# Storyboard: "Selecting the Right Mode" explainer (3b1b/manim, manimgl 1.7.2)

Narration: `video/script.py` (one beat = one audio clip; durations in `/media/kavinder/hdd2/RAM/video/audio/durations.json`).
Shared code: `video/common.py`. Import with `from common import *`.
- `Beat(self, "<Scene>_<bid>")`: adds the clip at the current time; `b.play(anims, frac=…)`, `b.wait(t)`, `b.done()`.
- Colours: `C_GT` = true pose, `C_PRED` = hypotheses/ours, `C_FLIP` = wrong mode, `C_DINO`, `C_MOGE`, `C_POE`, `C_BASE` = baselines, `C_FT` = fine-tuned, `CHART_COLORS[0..3]`, `LEVEL_COLORS[0..5]`. Background `BG`.
- Math: `T.encode` / `T.decode` (repo code), `encode`, `centre`, `free_coords`, `cell_box`, `quat`, `qmul`, `qmat`, `geodesic_deg`. **Use them; never fake the geometry.**

**Render:** `cd video && /home/kavinder/miniconda3/envs/manim311/bin/python render.py scenes/<file>.py <SceneClass> -w --hd --fps 60 --video_dir /media/kavinder/hdd2/RAM/video/scenes`
(use `-l` for a quick low-res check; extract frames with ffmpeg and look at them).

## Hard rules (from the brief)
- **Animation first, full screen.**
  - No slides, bullet lists, card grids, dashboards, or box-and-arrow architecture diagrams.
  - Text is short labels and progressively built equations attached to the geometry.
  - Large objects fill the 16:9 frame.
- **Every beat has visual progression.** Spend the beat's duration on meaningful motion; do not pad with long waits. Do not slow anything down artificially. If a beat is long, add genuinely useful animation steps.
- **Faithful.** Equations exactly as in the paper (`paper/sec/3_method.tex`), numbers exactly as in `video/script.py`.
- **Schematic objects are labelled as such.** For example, the spacecraft is a simple box-with-panel stand-in, not the real Tango model.
- **Colour conventions are fixed.**
- **3D:** use `ThreeDScene`, apply depth testing where faces overlap (`mob.apply_depth_test()`), use real rotation matrices (`qmat`), and move the camera with `self.frame`. Fixed-in-frame text: `mob.fix_in_frame()`.
- **Background colour:** each scene sets `self.camera.background_rgba` via `setup_scene(self)`, or simply adds a full-frame rectangle of `BG` behind everything. Be consistent.
- **Timing:** total scene length = sum of its beats (+ small gaps). Check that the `Beat.done()` warnings do not fire.

## Scenes

**S01_Problem** (2D + 3D)
- b1: Split-screen *without boxes*: a schematic L-shaped bracket (industrial part) on one side and a schematic satellite on the other, each with a camera frustum; a pinhole projection ray-casts the object into an image plane. Then the pose arrows: R (rotating frame axes) and t (vector from camera to object). Equation $(R, t)\in SO(3)\times\mathbb{R}^3$.
- b2: Represent a backbone as a large lattice (e.g. 40×24 grid of small dots/edges, "303 M parameters") and a small head as a tiny lattice. Fine-tuning: gradient pulses (orange) flow backward through the whole big lattice per object (three objects → three coloured copies). Ours: the big lattice turns grey (frozen, shared) and stays single; small heads sprout per object, with gradient pulses only in the heads. Labels: "DINOv3 303 M (frozen)", "MoGe-2 331 M (frozen)", "head 2.4 M".
- b3: Zoom onto one small head; the question as a short title.

**S02_Ambiguity** (3D)
- b1: A cube rotating by 90° about a face axis; overlay its silhouette before and after (identical). Then show all 24 rotations as 24 small copies/axes or as 24 points in the axis-angle ball.
- b2: A box with one small asymmetric feature (e.g. an antenna stub) turned by 180°: nearly identical silhouette; the feature is tiny; a "lighting change" dims details.
- b3: Axis-angle ball of radius π (translucent sphere) with point clouds:
  - the cube's 24 equivalent rotations as yellow (all correct) modes;
  - for the near-symmetric object, one yellow mode plus a red look-alike mode at 180°;
  - a density blob (Gaussian splats) around each mode;
  - "select one" arrow.
  - **Compute** the axis-angle vectors from the actual rotations (axis × angle).

**S03_Quaternions**
- b1: The unit 3-sphere cannot be drawn. Show a 2-sphere as the stand-in (labelled "S³, drawn as S²"), a point q, the equation $q = w + xi + yj + zk,\ \|q\|=1$.
- b2: Antipodal point −q joined through the centre, both mapping to one rotated object (same orientation). Equations: $v' = q\,v\,\bar q$ and $\delta(q_1,q_2) = 2\arccos|\langle q_1, q_2\rangle|$. Animate a vector being rotated by q.

**S04_Tesseract (HERO)**: implemented by the lead.

**S05_Features**
- b1: An image crop (use a real SPEED+ synthetic image, see `configs/paths.yaml` speedplus_root, or a rendered schematic) is cut into 16×16 patches. Patches fly into a 16×16 token sheet, are average-pooled 2×2 into an 8×8 grid, and each token extends into a tall 1024-length bar (stack). Label "64 × 1024".
- b2: A normal-map visualisation (colour = normal direction; compute normals of a schematic object procedurally). Two masks (DINOv3 foreground, MoGe-2 mask) intersect (AND) into the consensus mask. Then pooling to 16×16 cells, each a 4-vector (three normal components + mask fraction). "256 × 4".
- b3: Both backbone lattices freeze (grey, lock icon made from shapes); a cache cylinder fills once; then a clock showing "minutes" for head training.

**S06_TreeHead**
- b1: The 8×8 token grid as memory; a decoder column of tokens. The root token attends (lines of varying opacity to memory tokens) and outputs 4 chart probabilities as a bar fan in `CHART_COLORS`.
- b2: An autoregressive unrolling: each step appends a token (labelled with level ℓ and the cell centre) and emits 8 child probabilities. Highlight the argmax. Six steps.
  - **Use a real path:** pick q, compute `encode(q)`.
  - Make the bars **soft targets computed exactly:** for level ℓ, the children centres of the true prefix, δ = angle to the true q, $t\propto\exp(-(\delta/\tau_\ell)^2)$ with τ = [39.27, 39.27, 19.95, 10.01, 5.01, 2.51] (`phase3.TAU_DEG`).
- b3: Equation $t^{(\ell)}_k \propto \exp\!\big(-(\delta(q, c_{\ell,k})/\tau_\ell)^2\big)$; the Gaussian over 8 bars sharpens level by level as τ halves.
- b4: PoE: two bar charts (DINO teal, MoGe orange) for the same level, log-probabilities add (animate bars as log values summing), then renormalise into a sharper purple distribution. Equation $\log p^{(\ell)} \propto \log p_{\mathrm{DINO}}^{(\ell)} + \log p_{\mathrm{MoGe}}^{(\ell)}$. **Use plausible but clearly illustrative numbers** (label "illustrative").

**S07_Symmetry** (3D)
- b1: A square prism (symmetry set |S| = 4 about its axis, or a cube) rendered twice, identical images, with two different labels q and q·s (axes drawn in yellow). "contradictory targets".
- b2: Canonical labels: rotate the object continuously about its symmetry axis. Compute the canonical member (the equivalent closest to identity) at each angle and draw its frame. Show the canonical frame snapping/jumping when the closest member switches. Plot the canonical label's angle vs object angle (a sawtooth), computed exactly.
- b3: Multi-path:
  - all equivalents q·s drawn as frames at once;
  - their paths through the tree, from `encode(q·s)` for each s, drawn as several highlighted branches;
  - the loss equation $\mathcal{L}_{\text{sym}}=\frac{1}{|\mathcal{S}'|}\sum_{s\in\mathcal{S}'}\mathcal{L}_{\text{GT}}(q\,s)$, with "$|\mathcal S'|\le 8$";
  - probability splitting across paths;
  - final: "no symmetry ⇒ identical to standard loss" (single path).

**S08_Roll** (3D)
- b1: A camera (frustum) looking at an object off the optical axis; the image plane with the crop; the ray d through the crop centre; rotate the crop by θ in the image while the camera rotates about d. Both views coincide.
- b2: Equation $q' = \mathrm{Exp}(-\theta\, d)\otimes q,\quad t' = R(d,-\theta)\,t$. Show the object frame transforming. **Compute with `quat`/`qmul`.**
- b3: Off-axis object: an affine rotation of the crop vs the homography $K R(d,-\theta) K^{-1}$ warp. Show the grid warping under the homography (compute it), "exact".

**S09_Orbit** (3D + 2D)
- b1: Four rotated crops (0/90/180/270) arranged in a ring around the object; beam-8 candidate frames emerge from each (as small coordinate frames or points on an axis-angle ball).
- b2: Each candidate set rotates back through the inverse roll and lands in the common frame: 32 points on the axis-angle ball, clustered around the true pose (yellow) and one look-alike (red, 180°).
- b3: Score accumulation: for each candidate, four bars (one per view) stack up into a total. Equation $s(c)=\sum_{k=1}^{4}\log p_k\big(\mathrm{path}(\mathrm{Exp}(-\theta_k d)\otimes c)\big)$.
- b4: The flip candidate is high in one view and low in the others; the true candidate is consistently high; the argmax lights up. "cost: 4 passes".
- **Label the probabilities "illustrative"** unless computed.

**S10_Distill**
- b1: Teacher: three training views → 24 candidates with softmax weights (bars).
- b2: The student tree's level distributions morph toward the teacher's level marginals plus the GT target; then a single view at test time. Equation: the loss from `paper/sec/3_method.tex` eq. distill, with β = 1.

**S11_Translation**
- b1: Camera, image, crop box; one query token attends to the encoded tokens; output vector u (3 bars).
- b2: Draw the geometry:
  - box centre $(b_x, b_y)$ and size $(b_w, b_h)$;
  - projected object centre $(f_x X/Z + c_x,\ f_y Y/Z + c_y)$;
  - $U_x, U_y$ as normalised offsets;
  - $U_z = \tfrac12(\alpha b_w/W + b_h/H)\,Z$ with α = 1.6.
  - Then invert: from u back to $(X, Y, Z)$; animate the 3D point placement.
  - "0.33 M parameters".

**S12_Flat**
- b1: The flat head: a single vector of 131k logits drawn as a long dense strip (subsample); the weight matrix 256 × 131,072 as a huge sheet ("33.6 M"). The tree: a thin 6-level path ("2.4 M").
- b2: Greedy decode lights up 4 + 8·5 = 44 nodes vs the flat strip all lit. Error bars: tree 3.87° vs flat 7.53° (synthetic, PoE + orbit).
- b3: Grid floors: SPACE-HOP's 12-roll Hopf grid vs Tesseract cells, shown on a sphere of directions with 12 rolls per direction; floor 6.06° vs 2.51°; "same-grid control pending".

**S13_General**
- Animated results, not tables: bars growing, per-shape bars, recall curves, with numbers exactly from `script.py` and `paper/tables/*.tex`.
- **T-LESS** recall at 2/5/10:
  - A0 orbit 4.0 / 36.0 / 73.5;
  - A2 orbit 5.7 / 48.1 / 79.0;
  - LiePose 34.9 / 78.2 / 88.9.
- **SYMSOL-T** per shape, A2 DINOv3 orbit (from `paper/tables/symsol_sota.tex`) vs A0, plus the LiePose bar.
- **YCB-V:** real-only test 69.4 → with renders 12.9 (85% within 10°); orbit voting hurts (val 8.97 → 11.53).
- **Translation T@2/5/10:**
  - T-LESS 57.3 / 88.0 / 96.7 vs LiePose 68.0 / 88.4 / 95.2;
  - YCB-V 1.6 cm vs box-only 8.1 cm.
- Small caveat label: "rotation uses GT translation for frame conversion (T-LESS, YCB-V)".

**S14_Spacecraft**
- b1: Schematic satellite and two real-domain looks. Use real lightbox / sunlamp images from `speedplus_root` if readable (ImageMobject).
- b2: Flip: the satellite rotating 180° about a body axis. Bar: oracle 9.8 vs prediction 26.9 (lightbox).
- b3: Error staircase (lightbox / sunlamp):
  - clean tree 26.9 / 36.7;
  - augmentation 17.3 / 20.2;
  - orbit voting 12.3 / 12.3;
  - distillation 11.21 / 11.09.
  - Flip repair 62% / 80%; 0.4% broken.
- b4: Detector boxes vs GT boxes; error 23.9 / 16.4 with the detector; detector IoU 0.76 on lightbox.

**S15_FineTuning**
- b1: Bars on the same frozen DINOv3 features: frozen head 15.47 / 24.49, LoRA 1.17 / 1.56, ours 11.21 / 11.09.
- b2: Back-propagation passes per target as a log-scale strip:
  - Church & Nikolian 1.4 M;
  - SPNv2 1.0 M;
  - SPNv3 4.3 M;
  - PAID-ViT 4.3 M;
  - FastPose-ViT 7.2 M;
  - ours 0.
  - Backbone sizes 12–840 M.
- b3: Gap closed: 30% (lightbox) and 58% (sunlamp), as a filling bar between the frozen and LoRA levels. Direct regressors: PAID-ViT 12.29 / 11.76, FastPose 12.26 / 18.28. One backbone pass fanning out to many heads.

**S16_Synthesis**: return to the hero (the tesseract cube with the decoded path), then the orbit ring, then the frozen backbone with many small heads. A closing line.
