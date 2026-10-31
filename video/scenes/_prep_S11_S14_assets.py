"""Asset preparation for S11_Translation and S14_Spacecraft (run once with the manim311 python).

Reads SPEED+ (read-only) and writes downscaled images, crops and a JSON of exact numbers to
/media/kavinder/hdd2/RAM/video/assets/.  Projection follows src/tfpose/data.py (p_cam = R p + r, OpenCV
distortion model, gt_square_box with a 15% margin); cv2 is not available in this env, so the OpenCV
distortion model is written out explicitly.
Usage:  python prep_S11_S14_assets.py [img names to preview ...]"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io
from PIL import Image, ImageDraw

ROOT = Path("/media/kavinder/hdd2/RFO/speed_pose_moe/speedplusv2")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from tfpose.rotations import quat_to_matrix  # noqa: E402

OUT = Path("/media/kavinder/hdd2/RAM/video/assets")
OUT.mkdir(parents=True, exist_ok=True)
cam = json.loads((ROOT / "camera.json").read_text())
K = np.array(cam["cameraMatrix"], float); DIST = np.array(cam["distCoeffs"], float)
W, H = cam["Nu"], cam["Nv"]
PTS = scipy.io.loadmat(str(ROOT / "src/utils/tangoPoints.mat"))["tango3Dpoints"].T.astype(float)


def project(P_cam, distort=True):
    x, y = P_cam[:, 0] / P_cam[:, 2], P_cam[:, 1] / P_cam[:, 2]
    if distort:
        k1, k2, p1, p2, k3 = DIST; r2 = x * x + y * y; rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        x, y = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.c_[K[0, 0] * x + K[0, 2], K[1, 1] * y + K[1, 2]]


def gt_box(q, r):
    R = quat_to_matrix(np.asarray(q, float)[None])[0]
    uv = project(PTS @ R.T + r)
    x0, y0 = uv.min(0); x1, y1 = uv.max(0)
    side = max(x1 - x0, y1 - y0, 32.0) * 1.3
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1), side, uv


def labels(domain):
    return {e["filename"]: e for e in json.loads((ROOT / domain / "test.json").read_text())}


def crop(img, cx, cy, side, out=512):
    return img.crop((int(round(cx - side / 2)), int(round(cy - side / 2)),
                     int(round(cx + side / 2)), int(round(cy + side / 2)))).resize((out, out), Image.BILINEAR)


def spacehop_u(t, cx, cy, side, alpha=1.6):
    """SPACE-HOP target (paper/sec/3_method.tex), square box b_w = b_h = side, image (W, H)."""
    X, Y, Z = t; fx, fy, ccx, ccy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Ux = (fx * X / Z + ccx - cx) / side
    Uy = (fy * Y / Z + ccy - cy) / side
    Uz = 0.5 * (alpha * side / W + side / H) * Z
    return np.array([Ux, Uy, Uz])


def spacehop_inv(u, cx, cy, side, alpha=1.6):
    fx, fy, ccx, ccy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Z = u[2] / (0.5 * (alpha * side / W + side / H))
    X = (u[0] * side + cx - ccx) * Z / fx
    Y = (u[1] * side + cy - ccy) * Z / fy
    return np.array([X, Y, Z])


def preview(names):
    lb = labels("lightbox")
    for n in names:
        e = lb[n]; cx, cy, s, uv = gt_box(e["q_vbs2tango_true"], np.array(e["r_Vo2To_vbs_true"]))
        im = Image.open(ROOT / "lightbox/images" / n).convert("RGB"); d = ImageDraw.Draw(im)
        d.rectangle([cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2], outline=(255, 210, 100), width=6)
        for p in uv: d.ellipse([p[0] - 8, p[1] - 8, p[0] + 8, p[1] + 8], fill=(80, 200, 240))
        im.resize((960, 600)).save(f"/tmp/claude-1000/frames_C/prev_{n}")


def main():
    data = {"K": K.tolist(), "W": W, "H": H, "alpha": 1.6}
    lb, sl = labels("lightbox"), labels("sunlamp")
    det = {r["image_relpath"]: r for r in csv.DictReader(open(REPO / "outputs/data_manifests/exp124_det_test3.csv"))}
    # ---- S11 example: a real lightbox frame
    n = S11_IMG; e = lb[n]; t = np.array(e["r_Vo2To_vbs_true"]); cx, cy, s, uv = gt_box(e["q_vbs2tango_true"], t)
    u = spacehop_u(t, cx, cy, s); back = spacehop_inv(u, cx, cy, s)
    full = Image.open(ROOT / "lightbox/images" / n).convert("RGB")
    full.resize((960, 600), Image.BILINEAR).save(OUT / "s11_full.png")
    crop(full, cx, cy, s).save(OUT / "s11_crop.png")
    pc = np.array([K[0, 0] * t[0] / t[2] + K[0, 2], K[1, 1] * t[1] / t[2] + K[1, 2]])
    data["s11"] = dict(img=n, t=t.tolist(), box=[cx, cy, s, s], proj_centre=pc.tolist(), u=u.tolist(),
                       t_back=back.tolist(), err=float(np.abs(back - t).max()))
    # ---- S14 looks: lightbox and sunlamp crops
    data["s14"] = {}
    for dom, lab, name in [("lightbox", lb, S14_LB), ("sunlamp", sl, S14_SL)]:
        e = lab[name]; cx, cy, s, _ = gt_box(e["q_vbs2tango_true"], np.array(e["r_Vo2To_vbs_true"]))
        im = Image.open(ROOT / dom / "images" / name).convert("RGB")
        crop(im, cx, cy, s * 1.25).save(OUT / f"s14_{dom}.png")
        data["s14"][dom] = dict(img=name, box=[cx, cy, s])
    # ---- S14 b1: one synthetic training render
    syn = {e["filename"]: e for e in json.loads((ROOT / "synthetic/train.json").read_text())}
    e = syn[S14_SYN]; cx, cy, s, _ = gt_box(e["q_vbs2tango_true"], np.array(e["r_Vo2To_vbs_true"]))
    crop(Image.open(ROOT / "synthetic/images" / S14_SYN).convert("RGB"), cx, cy, s * 1.25).save(OUT / "s14_synthetic.png")
    data["s14"]["synthetic"] = dict(img=S14_SYN, box=[cx, cy, s])
    # ---- S14 b4: detected vs GT box on lightbox frames (exp124 manifest: crop_* is the detected square box)
    data["s14_det"] = []
    for name in S14_DET:
        e = lb[name]; cx, cy, s, _ = gt_box(e["q_vbs2tango_true"], np.array(e["r_Vo2To_vbs_true"]))
        r = det[f"lightbox/images/{name}"]
        im = Image.open(ROOT / "lightbox/images" / name).convert("RGB")
        im.resize((960, 600), Image.BILINEAR).save(OUT / f"s14_det_{name[:-4]}.png")
        data["s14_det"].append(dict(img=name, gt=[cx, cy, s], det=[float(r["crop_cx"]), float(r["crop_cy"]),
                                    float(r["crop_side"])], iou=float(r["det_iou"]), miss=r["det_miss"]))
    (OUT / "s11_s14.json").write_text(json.dumps(data, indent=1))
    print(json.dumps(data, indent=1))


S11_IMG = "img004459.jpg"
S14_LB = "img003031.jpg"
S14_SL = "img001500.jpg"
S14_SYN = "img000001.jpg"
S14_DET = ["img004459.jpg", "img004948.jpg"]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        preview(sys.argv[1:])
    else:
        main()
