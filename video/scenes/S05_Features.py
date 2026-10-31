"""S05_Features: the frozen feature interface.
- DINOv3 (ViT-L/16) sees a 256x256 crop -> 16x16 patch tokens -> 2x2 average pool -> 8x8 grid of 1024-vectors (64 x 1024).
- MoGe-2 normals are masked by the consensus of a DINOv3 foreground mask and the MoGe-2 mask, then pooled to 16x16 cells
  of 4 numbers (3 masked-normal channels + mask fraction) -> 256 x 4.

Assets (computed here, written to /media/kavinder/hdd2/RAM/video/assets/s05/):
- crop: SPEED+ synthetic img000001, grayscale, the GT square crop from the manifest (crop_cx, crop_cy, crop_side), with
  the same affine as tfpose.data.crop_square (bilinear, zero padding), at 256 px (what DINOv3 receives) and 512 px (display).
- normal map: NOT MoGe-2 output. A schematic stand-in computed procedurally: the Tango bus box ('box+plate' model of
  tfpose.masks: bottom-quad footprint extruded to the top height, plus the top plate) posed with the image's GT q, t,
  projected with the SPEED+ camera (K and distortion), visible faces filled with their camera-frame normals.
- masks: illustrative. 'DINOv3 foreground' = smoothed intensity threshold of the real crop; 'MoGe-2 mask' = the stand-in
  silhouette dilated, with a blob of background leakage. Consensus = AND.
- token colours are illustrative (patch mean intensity), the pooling arithmetic is exact (2x2 means / 32x32-pixel cell means)."""
import json
from pathlib import Path

from common import *  # noqa: F401,F403

ASSET = Path("/media/kavinder/hdd2/RAM/video/assets/s05")
ROOT = Path("/media/kavinder/hdd2/RFO/speed_pose_moe/speedplusv2")
IMG_REL = "synthetic/images/img000001.jpg"
# manifest row (outputs/data_manifests/subset_full.csv) for img000001: q (w,x,y,z), t (m), GT crop
Q_SRC = np.array([-0.336458, -0.093409, -0.8286, 0.437599])
T_CAM = np.array([-0.129896, 0.069519, 6.457073])
CROP = (981.608185, 551.741862, 697.049863)


# ---------------------------------------------------------------------------------------------- asset computation
def crop_square_pil(img, cx, cy, side, out):
    """Same affine as tfpose.data.crop_square: x_out = s*(x - cx) + out/2, bilinear, zero padding."""
    from PIL import Image
    s = out / side
    # PIL wants the inverse map: input = (x_out - out/2)/s + cx
    a, c = 1 / s, cx - (out / 2) / s
    e, f = 1 / s, cy - (out / 2) / s
    return img.transform((out, out), Image.AFFINE, (a, 0, c, 0, e, f), resample=Image.BILINEAR, fillcolor=0)


def project(P, R, t, K, dist):
    """OpenCV pinhole + Brown-Conrady distortion (k1, k2, p1, p2, k3), as cv2.projectPoints."""
    X = (R @ P.T).T + t
    x, y = X[:, 0] / X[:, 2], X[:, 1] / X[:, 2]
    k1, k2, p1, p2, k3 = dist
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], 1)


def to_crop(uv, out):
    cx, cy, side = CROP; s = out / side
    return np.stack([(uv[:, 0] - cx) * s + out / 2, (uv[:, 1] - cy) * s + out / 2], 1)


def prepare_assets(force=False):
    from PIL import Image, ImageDraw, ImageFilter
    import scipy.io
    ASSET.mkdir(parents=True, exist_ok=True)
    if (ASSET / "done.json").exists() and not force:
        return json.loads((ASSET / "done.json").read_text())
    img = Image.open(ROOT / IMG_REL).convert("L")
    c256 = crop_square_pil(img, *CROP, 256); c256.save(ASSET / "crop256.png")
    c512 = crop_square_pil(img, *CROP, 512); c512.convert("RGB").save(ASSET / "crop512.png")
    # 16x16 patches of the 256 crop (16 px each), saved upscaled for display
    a256 = np.asarray(c256, np.float32)
    (ASSET / "patches").mkdir(exist_ok=True)
    for i in range(16):
        for j in range(16):
            p = c512.crop((j * 32, i * 32, j * 32 + 32, i * 32 + 32))
            p.convert("RGB").save(ASSET / "patches" / f"p{i:02d}_{j:02d}.png")
    patch_mean = a256.reshape(16, 16, 16, 16).mean(axis=(1, 3)) / 255.0

    # ---- normal map of the box stand-in at the GT pose
    cam = json.loads((ROOT / "camera.json").read_text())
    K = np.array(cam["cameraMatrix"], float); dist = np.array(cam["distCoeffs"], float)
    pts = scipy.io.loadmat(str(ROOT / "src/utils/tangoPoints.mat"))["tango3Dpoints"].T.astype(float)
    q = Q_SRC / np.linalg.norm(Q_SRC); R = qmat(q)            # p_cam = R p + t (SPEED+ convention "R")
    bot = pts[4:8]; ztop = pts[0:4, 2].max()
    lo = np.array([bot[:, 0].min(), bot[:, 1].min(), bot[:, 2].min()]); hi = np.array([bot[:, 0].max(), bot[:, 1].max(), ztop])
    faces = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        for side, val in ((-1, lo[i]), (1, hi[i])):
            quad = []
            for a, b_ in ((0, 0), (1, 0), (1, 1), (0, 1)):
                p = np.zeros(3); p[i] = val; p[j] = lo[j] if a == 0 else hi[j]; p[k] = lo[k] if b_ == 0 else hi[k]
                quad.append(p)
            n = np.zeros(3); n[i] = side
            faces.append((np.array(quad), n))
    plate = pts[0:4]
    nP = np.cross(plate[1] - plate[0], plate[2] - plate[0]); nP /= np.linalg.norm(nP)
    faces.append((plate, nP))
    N = 512
    nmap = np.zeros((N, N, 3), np.float32); obj = np.zeros((N, N), bool)
    draw_list = []
    for quad, n in faces:
        Xc = (R @ quad.T).T + T_CAM
        nc = R @ n
        if np.dot(nc, Xc.mean(0)) > 0:          # faces away from the camera
            if len(quad) == 4 and np.allclose(quad, plate):
                nc = -nc                          # the plate is two-sided
            else:
                continue
        uv = to_crop(project(quad, R, T_CAM, K, dist), N)
        draw_list.append((Xc.mean(0)[2], uv, nc))
    draw_list.sort(key=lambda t: -t[0])          # far first
    for _, uv, nc in draw_list:
        m = Image.new("L", (N, N), 0); ImageDraw.Draw(m).polygon([tuple(p) for p in uv], fill=255)
        m = np.asarray(m) > 0
        nmap[m] = nc; obj |= m
    vis = np.stack([nmap[..., 0], -nmap[..., 1], -nmap[..., 2]], -1)    # display convention: x right, y up, z to viewer
    rgb = np.where(obj[..., None], (vis + 1) / 2, 0.0)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(ASSET / "normals512.png")

    # ---- illustrative masks
    smooth = np.asarray(c512.filter(ImageFilter.GaussianBlur(9)), np.float32)
    dino = smooth > 7
    sil = Image.fromarray((obj * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(21))
    moge = np.asarray(sil) > 0
    yy, xx = np.mgrid[0:N, 0:N]
    leak = ((xx - 400) / 95.0) ** 2 + ((yy - 430) / 60.0) ** 2 < 1.0          # background leakage blob
    moge = moge | leak
    cons = dino & moge
    for name, m, col in (("mask_dino", dino, C_DINO), ("mask_moge", moge, C_MOGE), ("mask_cons", cons, "#FFFFFF")):
        rgba = np.zeros((N, N, 4), np.uint8); rgba[..., :3] = (np.array(color_to_rgb(col)) * 255).astype(np.uint8)
        rgba[..., 3] = (m * 200).astype(np.uint8)
        Image.fromarray(rgba, "RGBA").save(ASSET / f"{name}.png")
    masked = np.where(cons[..., None], rgb, 0.0)
    Image.fromarray((masked * 255).astype(np.uint8)).save(ASSET / "normals_masked512.png")

    # ---- pooling to 16x16 cells (32x32 px each): 3 masked-normal channels (cell means of n * mask) + mask fraction
    mn = (np.where(cons[..., None], nmap, 0.0)).reshape(16, 32, 16, 32, 3).mean(axis=(1, 3))
    frac = cons.reshape(16, 32, 16, 32).mean(axis=(1, 3))
    info = dict(patch_mean=patch_mean.tolist(), cell_normal=mn.tolist(), cell_frac=frac.tolist())
    (ASSET / "done.json").write_text(json.dumps(info))
    return info


# ---------------------------------------------------------------------------------------------- scene helpers
def make_lattice_pts(n, cols, origin, sp):
    idx = np.arange(n); r, c = idx // cols, idx % cols
    return np.stack([origin[0] + c * sp, origin[1] - r * sp, np.zeros(n)], 1)


def lock_icon(color=GREY_B, s=1.0):
    body = RoundedRectangle(width=0.5 * s, height=0.4 * s, corner_radius=0.06 * s, fill_color=color, fill_opacity=1, stroke_width=0)
    shackle = Arc(0, PI, radius=0.16 * s, stroke_color=color, stroke_width=5 * s).next_to(body, UP, buff=-0.02 * s)
    hole = Dot(body.get_center() + UP * 0.03 * s, radius=0.045 * s, color=BG)
    return VGroup(shackle, body, hole)


def token_color(v):
    """Illustrative token colour from a scalar in [0, 1] (patch mean intensity): dark slate -> DINO teal -> white."""
    v = float(np.clip(v * 1.6, 0, 1))
    c0, c1, c2 = np.array(color_to_rgb("#1B2A35")), np.array(color_to_rgb(C_DINO)), np.array([1.0, 1.0, 1.0])
    return rgb_to_hex(c0 + (c1 - c0) * v / 0.7 if v < 0.7 else c1 + (c2 - c1) * (v - 0.7) / 0.3)


def normal_rgb(n):
    n = np.asarray(n, float); L = np.linalg.norm(n)
    if L < 1e-9:
        return np.zeros(3)
    n = n / L
    return (np.array([n[0], -n[1], -n[2]]) + 1) / 2


class S05_Features(ThreeDScene):
    default_camera_config = dict(background_color=BG)

    def add(self, *mobjects, **kw):
        for m in mobjects:
            if getattr(m, "_depth", False):
                ThreeDScene.add(self, m)
            else:
                Scene.add(self, m)
        return self

    def construct(self):
        info = prepare_assets()
        pm = np.array(info["patch_mean"]); cn = np.array(info["cell_normal"]); cf = np.array(info["cell_frac"])
        self.frame.reorient(0, 0, 0)

        # ================================================================== b1
        b = Beat(self, "S05_Features_b1")
        IMC, IMS = np.array([-4.55, 0.1, 0]), 4.3                 # image centre / size
        img = ImageMobject(str(ASSET / "crop512.png")).set_height(IMS).move_to(IMC)
        img_lab = TexText(r"SPEED+ synthetic image, $256\times256$ crop", font_size=26, color=GREY_B).next_to(img, DOWN, buff=0.15)
        b.play(FadeIn(img), FadeIn(img_lab), rt=0.7)
        ps = IMS / 16
        grid = VGroup(*[Line(IMC + np.array([-IMS / 2 + k * ps, -IMS / 2, 0]), IMC + np.array([-IMS / 2 + k * ps, IMS / 2, 0])) for k in range(17)],
                      *[Line(IMC + np.array([-IMS / 2, -IMS / 2 + k * ps, 0]), IMC + np.array([IMS / 2, -IMS / 2 + k * ps, 0])) for k in range(17)])
        grid.set_stroke(C_DINO, 1, opacity=0.8)
        patch_lab = TexText(r"$16\times16$ patches of $16\times16$ px", font_size=26, color=C_DINO).next_to(img, UP, buff=0.15)
        b.play(ShowCreation(grid, lag_ratio=0.05), FadeIn(patch_lab), rt=0.8)

        # frozen DINOv3 lattice (grey) in the middle
        LC = np.array([-0.55, 0.1, 0])
        lat = DotCloud(make_lattice_pts(24 * 16, 24, LC + np.array([-23 * 0.075 / 2, 15 * 0.075 / 2, 0]), 0.075), radius=0.022,
                       color="#6B7682").set_opacity(0.95)
        lat_lab = TexText("DINOv3 (frozen)", font_size=28, color=C_DINO).next_to(lat, UP, buff=0.2)
        lock = lock_icon(s=0.8).next_to(lat, DOWN, buff=0.2)
        b.play(FadeIn(lat), FadeIn(lat_lab), FadeIn(lock), rt=0.5)

        # patches fly through the lattice and land as tokens
        TC, TS = np.array([4.3, 0.1, 0]), 4.3
        ts = TS / 16
        patches = Group()
        for i in range(16):
            for j in range(16):
                p = ImageMobject(str(ASSET / "patches" / f"p{i:02d}_{j:02d}.png")).set_height(ps * 0.98)
                p.move_to(IMC + np.array([-IMS / 2 + (j + 0.5) * ps, IMS / 2 - (i + 0.5) * ps, 0]))
                patches.add(p)
        tokens = VGroup()
        for i in range(16):
            for j in range(16):
                sq = Square(side_length=ts * 0.9, stroke_width=0, fill_color=token_color(pm[i, j]), fill_opacity=1)
                sq.move_to(TC + np.array([-TS / 2 + (j + 0.5) * ts, TS / 2 - (i + 0.5) * ts, 0]))
                tokens.add(sq)
        self.add(patches)
        self.remove(grid); self.add(grid)
        b.play(FadeOut(grid), LaggedStart(*[p.animate.move_to(LC).scale(0.3).set_opacity(0.0) for p in patches], lag_ratio=0.004),
               rt=1.3)
        self.remove(patches)
        b.play(LaggedStart(*[GrowFromPoint(t, LC) for t in tokens], lag_ratio=0.004), rt=1.2)
        tok_lab = TexText(r"last-layer patch tokens: $16\times16$", font_size=26, color=GREY_A).next_to(VGroup(*tokens), UP, buff=0.15)
        tok_ill = TexText("(colours illustrative)", font_size=22, color=GREY_B).next_to(tok_lab, UP, buff=0.06)
        b.play(FadeIn(tok_lab), FadeIn(tok_ill), rt=0.4)

        # 2x2 average pooling -> 8x8 (exact means of the four token values)
        pooled = VGroup()
        TS8 = TS
        t8 = TS8 / 8
        for i in range(8):
            for j in range(8):
                v = pm[2 * i:2 * i + 2, 2 * j:2 * j + 2].mean()
                sq = Square(side_length=t8 * 0.9, stroke_width=0, fill_color=token_color(v), fill_opacity=1)
                sq.move_to(TC + np.array([-TS8 / 2 + (j + 0.5) * t8, TS8 / 2 - (i + 0.5) * t8, 0]))
                pooled.add(sq)
        win = Square(side_length=2 * ts, stroke_color=C_GT, stroke_width=4).move_to(TC + np.array([-TS / 2 + ts, TS / 2 - ts, 0]))
        pool_lab = TexText(r"average pool $2\times2$ $\rightarrow$ $8\times8$", font_size=26, color=C_GT).next_to(VGroup(*tokens), DOWN, buff=0.15)
        b.play(ShowCreation(win), FadeIn(pool_lab), rt=0.4)
        for k in range(3):
            b.play(win.animate.shift(RIGHT * 2 * ts), rt=0.18)
        groups = []
        for i in range(8):
            for j in range(8):
                groups.append(VGroup(*[tokens[(2 * i + di) * 16 + 2 * j + dj] for di in (0, 1) for dj in (0, 1)]))
        # each of the four tokens morphs into the pooled square (their exact mean colour)
        b.play(FadeOut(win), *[Transform(sq, pooled[k].copy()) for k, g in enumerate(groups) for sq in g], rt=1.0)
        self.remove(tokens); self.add(pooled)

        # each pooled token is a 1024-vector: bars rise out of the grid (3-D)
        b.play(FadeOut(Group(img, img_lab, patch_lab, lat, lat_lab, lock, pool_lab, tok_lab, tok_ill)), pooled.animate.move_to(ORIGIN + DOWN * 0.3),
               rt=0.6)
        H = 3.4
        bars = Group()
        for k, sq in enumerate(pooled):
            c = sq.get_center()
            pr = Prism(t8 * 0.62, t8 * 0.62, H, color=sq.get_fill_color(), shading=(0.3, 0.5, 0.2))
            pr.move_to(c + OUT * H / 2)
            bars.add(pr)
        bars._depth = True
        dims = TexText(r"$64\times1024$", font_size=56, color=C_DINO).to_corner(UR, buff=0.5).fix_in_frame()
        each = TexText(r"each token: 1024 numbers", font_size=30, color=GREY_A).next_to(dims, DOWN, buff=0.15).align_to(dims, RIGHT).fix_in_frame()
        self.add(bars)
        for br in bars:
            br.stretch(0.02, 2, about_edge=IN)
        b.play(self.frame.animate.reorient(-25, 58, 0).move_to(OUT * 1.0 + DOWN * 0.2).set_height(7.0),
               *[br.animate.stretch(50, 2, about_edge=IN) for br in bars], rt=1.4)
        b.play(Write(dims), FadeIn(each), self.frame.animate.increment_theta(12 * DEGREES), rt=b.left - 0.05)
        b.done()

        # ================================================================== b2
        b = Beat(self, "S05_Features_b2")
        small8 = pooled.copy().scale(0.33).to_corner(UL, buff=0.35).shift(DOWN * 0.15)
        small_lab = TexText(r"$64\times1024$", font_size=24, color=C_DINO).next_to(small8, DOWN, buff=0.08)
        b.play(FadeOut(bars), FadeOut(dims), FadeOut(each), self.frame.animate.reorient(0, 0, 0).move_to(ORIGIN).set_height(FRAME_HEIGHT),
               ReplacementTransform(pooled, small8), FadeIn(small_lab), rt=0.7)
        # MoGe-2: a normal per pixel (stand-in computed from the box model at the true pose)
        A, NS = np.array([-4.55, -0.35, 0]), 4.1
        img2 = ImageMobject(str(ASSET / "crop512.png")).set_height(NS).move_to(A)
        nm = ImageMobject(str(ASSET / "normals512.png")).set_height(NS).move_to(A + RIGHT * 4.45)
        moge_lab = TexText("MoGe-2 (frozen): a surface normal per pixel", font_size=28, color=C_MOGE).to_edge(UP, buff=0.3).shift(RIGHT * 1.0)
        sch = TexText("schematic: normals of a box stand-in at the true pose", font_size=22, color=GREY_B).next_to(nm, DOWN, buff=0.12)
        b.play(FadeIn(img2), FadeIn(moge_lab), rt=0.5)
        sweep = Rectangle(width=0.06, height=NS, stroke_width=0, fill_color=C_MOGE, fill_opacity=0.9).move_to(nm.get_left())
        nm_cover = Rectangle(width=NS, height=NS, stroke_width=0, fill_color=BG, fill_opacity=1).move_to(nm)
        self.add(nm, nm_cover, sweep)

        def cover_upd(m):
            x0 = sweep.get_center()[0]
            right = nm.get_right()[0]
            w = max(right - x0, 0.001)
            m.stretch_to_fit_width(w).move_to(np.array([x0 + w / 2, nm.get_center()[1], 0]))
        nm_cover.add_updater(cover_upd)
        b.play(sweep.animate.move_to(nm.get_right()), FadeIn(sch), rt=1.2, rate_func=linear)
        nm_cover.clear_updaters(); self.remove(nm_cover, sweep)
        # legend: colour = normal direction, computed on a hemisphere
        leg = VGroup()
        LR, LC_ = 0.7, np.array([4.6, 0.4, 0])
        for u in np.linspace(-1, 1, 15):
            for v in np.linspace(-1, 1, 15):
                if u * u + v * v <= 1:
                    n = np.array([u, -v, -np.sqrt(max(0, 1 - u * u - v * v))])     # camera frame, facing the camera
                    leg.add(Square(side_length=2 * LR / 15, stroke_width=0, fill_color=rgb_to_hex(normal_rgb(n)), fill_opacity=1)
                            .move_to(LC_ + np.array([u * LR, v * LR, 0])))
        leg_lab = TexText("colour = normal", font_size=22, color=GREY_B).next_to(leg, DOWN, buff=0.1)
        b.play(FadeIn(leg), FadeIn(leg_lab), rt=0.6)

        # two masks, consensus (AND)
        M1 = ImageMobject(str(ASSET / "mask_dino.png")).set_height(2.1).move_to(np.array([3.45, -0.2, 0]))
        M2 = ImageMobject(str(ASSET / "mask_moge.png")).set_height(2.1).move_to(np.array([5.75, -0.2, 0]))
        f1 = Square(side_length=2.1, stroke_color=C_DINO, stroke_width=2).move_to(M1)
        f2 = Square(side_length=2.1, stroke_color=C_MOGE, stroke_width=2).move_to(M2)
        l1 = TexText("DINOv3 fg", font_size=22, color=C_DINO).next_to(f1, DOWN, buff=0.08)
        l2 = TexText("MoGe-2 mask", font_size=22, color=C_MOGE).next_to(f2, DOWN, buff=0.08)
        ill = TexText("(masks illustrative)", font_size=20, color=GREY_B).next_to(VGroup(l1, l2), DOWN, buff=0.08)
        b.play(FadeOut(leg), FadeOut(leg_lab), FadeIn(M1), FadeIn(M2), ShowCreation(f1), ShowCreation(f2), FadeIn(l1), FadeIn(l2), FadeIn(ill), rt=1.0)
        mid = (M1.get_center() + M2.get_center()) / 2
        andl = TexText("AND", font_size=36, color=WHITE).move_to(mid + UP * 1.4)
        b.play(M1.animate.move_to(mid), M2.animate.move_to(mid), f1.animate.move_to(mid), f2.animate.move_to(mid),
               FadeOut(l1), FadeOut(l2), FadeIn(andl), rt=0.9)
        MC = ImageMobject(str(ASSET / "mask_cons.png")).set_height(2.1).move_to(mid)
        fc = Square(side_length=2.1, stroke_color=WHITE, stroke_width=2).move_to(mid)
        lc = TexText("consensus mask", font_size=24, color=WHITE).next_to(fc, DOWN, buff=0.1)
        b.play(FadeOut(M1), FadeOut(M2), FadeOut(f1), FadeOut(f2), FadeIn(MC), ShowCreation(fc), FadeIn(lc), FadeOut(ill), rt=0.7)
        # mask applied to the normals: background leakage removed
        nmm = ImageMobject(str(ASSET / "normals_masked512.png")).set_height(NS).move_to(nm)
        mc2 = MC.copy()
        b.play(mc2.animate.move_to(nm).set_height(NS).set_opacity(0.6), rt=0.6)
        b.play(FadeOut(mc2), FadeIn(nmm), FadeOut(nm), rt=0.5)

        # pooling into 16x16 cells: mean masked normal (3 numbers) + mask fraction
        cs = NS / 16
        cells = VGroup()
        for i in range(16):
            for j in range(16):
                fr = cf[i, j]
                col = normal_rgb(cn[i, j]) if fr > 0 else np.zeros(3)
                sq = Square(side_length=cs * 0.92, stroke_width=0, fill_color=rgb_to_hex(col), fill_opacity=0.15 + 0.85 * fr if fr > 0 else 0.08)
                sq.move_to(nm.get_center() + np.array([-NS / 2 + (j + 0.5) * cs, NS / 2 - (i + 0.5) * cs, 0]))
                cells.add(sq)
        cgrid = VGroup(*[Line(nm.get_corner(UL) + RIGHT * k * cs, nm.get_corner(DL) + RIGHT * k * cs) for k in range(17)],
                       *[Line(nm.get_corner(DL) + UP * k * cs, nm.get_corner(DR) + UP * k * cs) for k in range(17)]).set_stroke(GREY_B, 1, 0.6)
        b.play(ShowCreation(cgrid, lag_ratio=0.03), FadeOut(img2), rt=0.7)
        b.play(FadeOut(nmm), FadeIn(cells, lag_ratio=0.003), rt=0.9)
        # one cell -> four numbers
        fr_flat = cf.flatten()
        k_best = int(np.argmax(np.where((fr_flat > 0.4) & (fr_flat < 0.85), fr_flat, 0)))   # a boundary cell
        if fr_flat[k_best] == 0:
            k_best = int(np.argmax(fr_flat))
        i0, j0 = divmod(k_best, 16)
        cell = cells[k_best]
        hl = Square(side_length=cs * 1.15, stroke_color=C_GT, stroke_width=4).move_to(cell)
        vals = list(cn[i0, j0]) + [cf[i0, j0]]
        names = [r"\bar n_x", r"\bar n_y", r"\bar n_z", r"m"]
        BX = np.array([3.3, -0.6, 0])
        bar_g = VGroup()
        for k, (v, nmn) in enumerate(zip(vals, names)):
            x = BX + RIGHT * k * 0.85
            h = 1.3 * v
            r = Rectangle(width=0.5, height=max(abs(h), 0.01), stroke_width=0, fill_color=[C_MOGE, C_MOGE, C_MOGE, WHITE][k], fill_opacity=0.9)
            r.move_to(x + UP * h / 2)
            lab = Tex(nmn, font_size=28).move_to(x + DOWN * 1.65)
            num = DecimalNumber(v, num_decimal_places=2, include_sign=k < 3, font_size=22).move_to(x + UP * (1.6 if True else 0))
            bar_g.add(VGroup(r, lab, num))
        axis = Line(BX + LEFT * 0.4, BX + RIGHT * 2.95, stroke_width=1.5, color=GREY_C)
        link = DashedLine(hl.get_right(), BX + LEFT * 0.45, dash_length=0.08, stroke_width=2, color=C_GT)
        dims2 = TexText(r"$256\times4$", font_size=56, color=C_MOGE).move_to([4.65, 2.35, 0])
        b.play(ShowCreation(hl), FadeOut(Group(MC, fc, lc, andl)), rt=0.4)
        b.play(ShowCreation(link), ShowCreation(axis), LaggedStartMap(FadeIn, bar_g, lag_ratio=0.2), rt=1.0)
        b.play(Write(dims2), rt=b.left - 0.05)
        b.done()

        # ================================================================== b3
        b = Beat(self, "S05_Features_b3")
        b.play(FadeOut(Group(cells, cgrid, hl, link, axis, bar_g, dims2, moge_lab, sch)), small8.animate.set_opacity(0.0),
               FadeOut(small_lab), rt=0.5)
        # both backbones, grey and locked
        sp = 0.07
        d_o = np.array([-6.2, 3.0, 0]); m_o = np.array([-6.2, -0.6, 0])
        dlat = DotCloud(make_lattice_pts(960, 40, d_o, sp), radius=0.028, color="#8A96A3").set_opacity(0.95)
        mlat = DotCloud(make_lattice_pts(1049, 40, m_o, sp), radius=0.028, color="#8A96A3").set_opacity(0.95)
        dl = TexText("DINOv3", font_size=28, color=C_DINO).next_to(dlat, UP, buff=0.1).align_to(dlat, LEFT)
        ml = TexText("MoGe-2", font_size=28, color=C_MOGE).next_to(mlat, UP, buff=0.1).align_to(mlat, LEFT)
        lk1 = lock_icon(s=0.9).next_to(dlat, RIGHT, buff=0.2)
        lk2 = lock_icon(s=0.9).next_to(mlat, RIGHT, buff=0.2)
        b.play(FadeIn(dlat), FadeIn(mlat), FadeIn(dl), FadeIn(ml), rt=0.5)
        never = TexText("never trained", font_size=30, color=GREY_A).next_to(Group(dlat, mlat), DOWN, buff=0.35)
        b.play(FadeIn(lk1, scale=2.0), FadeIn(lk2, scale=2.0), FadeIn(never), rt=0.6)

        # cache: computed once per image, stored
        CC = np.array([1.0, 0.2, 0]); CW, CH = 1.8, 3.0
        body = Rectangle(width=CW, height=CH, stroke_width=0, fill_color="#1C2633", fill_opacity=1).move_to(CC)
        top = Ellipse(width=CW, height=0.45, stroke_color=GREY_B, stroke_width=2, fill_color="#26323F", fill_opacity=1).move_to(CC + UP * CH / 2)
        bot = Arc(PI, PI, radius=1.0, stroke_color=GREY_B, stroke_width=2).stretch_to_fit_width(CW).stretch_to_fit_height(0.225)
        bot.move_to(CC + DOWN * CH / 2 + DOWN * 0.1125)
        sides = VGroup(Line(CC + np.array([-CW / 2, CH / 2, 0]), CC + np.array([-CW / 2, -CH / 2, 0])),
                       Line(CC + np.array([CW / 2, CH / 2, 0]), CC + np.array([CW / 2, -CH / 2, 0]))).set_stroke(GREY_B, 2)
        fill = Rectangle(width=CW - 0.04, height=0.01, stroke_width=0, fill_color=C_PRED, fill_opacity=0.55)
        fill.move_to(CC + DOWN * CH / 2, aligned_edge=DOWN)
        cyl = VGroup(body, fill, sides, bot, top)
        clab = TexText("feature cache", font_size=26, color=GREY_A).next_to(top, UP, buff=0.12)
        once = TexText("once per image", font_size=24, color=C_PRED).next_to(bot, DOWN, buff=0.12)
        b.play(FadeIn(cyl), FadeIn(clab), rt=0.4)
        # sheets fly in from both backbones; the level rises
        sheets = VGroup()
        for k in range(10):
            src = dlat.get_right() if k % 2 == 0 else mlat.get_right()
            sh = Square(side_length=0.35, stroke_width=1, stroke_color=WHITE, fill_color=C_DINO if k % 2 == 0 else C_MOGE, fill_opacity=0.8)
            sh.move_to(src + RIGHT * 1.25)
            sheets.add(sh)
        lvl = ValueTracker(0.01)
        fill.add_updater(lambda m: m.stretch_to_fit_height(max(lvl.get_value(), 0.01)).move_to(CC + DOWN * CH / 2, aligned_edge=DOWN))
        b.play(LaggedStart(*[Succession(FadeIn(s, run_time=0.1), s.animate.move_to(CC + UP * 0.4).scale(0.4).set_opacity(0))
                             for s in sheets], lag_ratio=0.12),
               lvl.animate.set_value(CH * 0.85), FadeIn(once), rt=1.9)
        fill.clear_updaters()

        # head training from the cache: minutes
        CK = np.array([4.9, 0.5, 0]); RK = 1.15
        dial = Circle(radius=RK, stroke_color=GREY_A, stroke_width=4).move_to(CK)
        ticks = VGroup(*[Line(CK + 0.85 * RK * np.array([np.sin(a), np.cos(a), 0]), CK + RK * np.array([np.sin(a), np.cos(a), 0]),
                              stroke_width=3, color=GREY_B) for a in np.arange(12) * TAU / 12])
        hand = Line(CK, CK + 0.85 * RK * UP, stroke_width=6, color=C_FT)
        wedge = AnnularSector(inner_radius=0, outer_radius=RK * 0.8, angle=0.001, start_angle=PI / 2, fill_color=C_FT, fill_opacity=0.25, arc_center=CK)
        mins = TexText("minutes", font_size=40, color=C_FT).next_to(dial, DOWN, buff=0.2)
        hlab = TexText("head training", font_size=26, color=GREY_A).next_to(dial, UP, buff=0.2)
        b.play(FadeIn(dial), FadeIn(ticks), FadeIn(hand), FadeIn(hlab), rt=0.4)
        ang = ValueTracker(0.0)
        hand.add_updater(lambda m: m.put_start_and_end_on(CK, CK + 0.85 * RK * np.array([np.sin(ang.get_value()), np.cos(ang.get_value()), 0])))
        wedge.add_updater(lambda m: m.become(AnnularSector(inner_radius=0, outer_radius=RK * 0.8, angle=-max(ang.get_value(), 0.001),
                                                           start_angle=PI / 2, fill_color=C_FT, fill_opacity=0.25, arc_center=CK)))
        self.add(wedge)
        flow = Arrow(top.get_right() + DOWN * 0.3, dial.get_left() + UP * 0.1, buff=0.15, stroke_width=4, color=C_PRED)
        b.play(ang.animate.set_value(TAU * 0.55), FadeIn(mins), GrowArrow(flow), rt=b.left - 0.05)
        hand.clear_updaters(); wedge.clear_updaters()
        b.done()
