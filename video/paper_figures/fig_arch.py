"""Paper figure 2 (main architecture): the general method for any known object, training and test time in one diagram.
Shared frozen backbones (snowflake) -> cached token grids -> per-object trained heads (flame): one tesseract tree Transformer
per branch and a translation block; training uses three views with exact in-plane labels and multi-path targets, test time
uses four rolled views, per-level PoE beam search and orbit voting.
Running example (fixed rules, prep_arch.py): T-LESS object 5. Training views: the first object-5 PBR training crop with
visib_fract > 0.95 and its two cached augmented copies (re-created with the extraction code). Test views: the T-LESS teaser
image (median-error test image of object 5), rolled by the homography of Eq. roll; feature maps are its cached features;
the output overlay is the final T-LESS system's prediction.
Render:  python render.py paper_figures/fig_arch.py FigArch -s -w -r 4200x1700 --video_dir <dir>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pfcommon import *  # noqa: F401,F403,E402

AS = json.loads((ASSET / "assets.json").read_text()); AR = AS["arch"]; TE = AS["teaser"]["tless"]


def labelled(name, h, cap, size=18, color=INK, border=None, up=False):
    m = img(name, h, border=border); l = tt(cap, size, color)
    l.next_to(m, UP if up else DOWN, buff=0.07)
    return Group(m, l)


def path_strip(w=0.34, h=0.3):
    toks = VGroup(*[RoundedRectangle(width=w, height=h, corner_radius=0.05).set_fill(PAPER, 1)
                    .set_stroke(DINO if i == 0 else LV[i - 1] if i > 1 else CH4[2], 2) for i in range(6)]).arrange(RIGHT, buff=0.06)
    labs = VGroup(*[tx(s, 13).move_to(t) for s, t in zip([r"\mathrm{R}", "1", "2", "3", "4", "5"], toks)])
    return VGroup(toks, labs)


class FigArch(PaperScene):
    def construct(self):
        # ------------------------------------------------------------------ inputs: training views (top), test views (bottom)
        a1, a2 = AR["aug"]["1"], AR["aug"]["2"]
        trv = Group(labelled("arch_train_clean", 1.05, r"clean", border=TRAIN),
                    labelled("arch_train_aug1", 1.05, rf"$\theta={a1:.0f}^\circ$", border=TRAIN),
                    labelled("arch_train_aug2", 1.05, rf"$\theta={a2:.0f}^\circ$", border=TRAIN)).arrange(RIGHT, buff=0.12)
        trt = tt(r"\textbf{training:} synthetic crop $+$ 2 augmented copies", 19, TRAIN).next_to(trv, UP, buff=0.12)
        trl = tx(r"q'=\mathrm{Exp}(-\theta d)\otimes q,\;\; t'=R(d,-\theta)\,t", 20, TRAIN).next_to(trv, DOWN, buff=0.12)
        TR = Group(trt, trv, trl).move_to([-0.9, 2.05, 0])
        tev = Group(*[labelled(f"arch_view{k}", 0.8, rf"${90 * k}^\circ$", 17, border=ORBIT) for k in range(4)]).arrange(RIGHT, buff=0.1)
        tet = tt(r"\textbf{test:} crop (GT box or detector),\\ 4 rolled views $\theta_k$, each through the network", 19, ORBIT).next_to(tev, UP, buff=0.12)
        TE_ = Group(tet, tev).move_to([-0.9, -1.75, 0])

        # ------------------------------------------------------------------ frozen backbones, shared by every object
        dino = block(r"DINOv3 ViT-L/16", [r"0.30\,B"], FROZEN, FROZEN_FILL, frozen=True, tsize=21, lsize=18)
        moge = block(r"MoGe-2 ViT-L", [r"0.33\,B, normals"], FROZEN, FROZEN_FILL, frozen=True, tsize=21, lsize=18)
        bb = VGroup(moge, dino).arrange(DOWN, buff=1.3).move_to([3.95, 0.05, 0])
        fz = VGroup(snowflake(0.3), tt(r"\textbf{frozen, shared by all objects}", 19, FROZEN)).arrange(RIGHT, buff=0.1).next_to(bb, UP, buff=0.18)
        fzi = tt(r"features extracted once, cached", 16, MUTED).next_to(bb, DOWN, buff=0.18)
        feat_m = labelled("arch_moge_normals", 1.0, r"normals, masked\\ $256{\times}4$", 15, MUTED, border=MOGE)
        feat_d = labelled("arch_dino_pca", 1.0, r"tokens (PCA)\\ $64{\times}1024$", 15, MUTED, border=DINO)
        feat_m.next_to(moge, RIGHT, buff=0.55); feat_d.next_to(dino, RIGHT, buff=0.55)
        feat_m[0].set_y(moge.get_y()); feat_m[1].next_to(feat_m[0], DOWN, buff=0.07)
        feat_d[0].set_y(dino.get_y()); feat_d[1].next_to(feat_d[0], DOWN, buff=0.07)

        # ------------------------------------------------------------------ per-object trained heads
        def tree(name, col, fill, npar):
            b = box(3.9, 1.35, col, fill, r=0.1, sw=3)
            t = tt(rf"\textbf{{tree Transformer}} ({name})", 19, col)
            s = path_strip(); n = tt(rf"$\mathrm{{path}}=(c;k_1..k_5)$, {npar}\,M", 15, MUTED)
            VGroup(t, s, n).arrange(DOWN, buff=0.1).move_to(b).shift(LEFT * 0.17)
            return VGroup(b, t, s, n, badge(b, flame()))
        tm = tree("MoGe-2", MOGE, MOGE_FILL, "2.1"); td = tree("DINOv3", DINO, DINO_FILL, "2.4")
        X_T = feat_d.get_right()[0] + 0.6 + 1.95
        tm.move_to([X_T, moge.get_y(), 0]); td.move_to([X_T, dino.get_y(), 0])
        trn = block(r"translation block", [r"query $\to$ tree tokens, MLP $\to u\to t$"], TRANS, TRANS_FILL, tsize=18, lsize=15, w=3.9, icon="fire")
        trn.next_to(td, DOWN, buff=0.42)
        mem = DashedLine(td.get_bottom() + LEFT * 0.9, trn.get_top() + LEFT * 0.9, dash_length=0.06, stroke_width=1.6).set_color(TRANS)
        hg = VGroup(tm, td, trn); hd = box(hg.get_width() + 0.32, hg.get_height() + 0.75, FIRE, "#FFF8F2", frozen=True, sw=1.8).move_to(hg).shift(UP * 0.2)
        hdl = VGroup(flame(0.3), tt(r"\textbf{trained per object} (4.85\,M with both branches)", 18, FIRE)).arrange(RIGHT, buff=0.1).next_to(hd, DOWN, buff=0.1)
        opt = tt(r"(MoGe-2 branch optional, chosen on validation)", 14, MUTED).next_to(tm, UP, buff=0.07)

        # ------------------------------------------------------------------ training objective (top right)
        loss = block(r"training loss (per level $\ell$)",
                     [r"multi-path soft targets: $\mathcal{L}_{\mathrm{sym}}=\frac{1}{|\mathcal{S}'|}\sum_{s}\mathcal{L}_{\mathrm{GT}}(q\,s)$",
                      r"$+\;\|u-u^*\|^2$ (translation)",
                      r"optional: orbit-teacher distillation"], TRAIN, "#FDF1E0", tsize=19, lsize=16, buff=0.08, icon=False)
        loss.next_to(hd, UP, buff=0.45)

        # ------------------------------------------------------------------ test time: PoE beam search, orbit voting, output
        poe = VGroup(Circle(radius=0.27).set_fill(PAPER, 1).set_stroke(ORBIT, 3), tx(r"+", 30, ORBIT))
        poe.move_to([hd.get_right()[0] + 0.85, (tm.get_y() + td.get_y()) / 2, 0])
        poel = tt(r"per-level PoE:\\ $\log p_D+\log p_M$", 16, ORBIT)
        beam = block(r"beam search", [r"8 leaves per view"], ORBIT, ORBIT_FILL, tsize=18, lsize=15, icon=False)
        beam.next_to(poe, RIGHT, buff=0.45)
        orb = block(r"orbit voting", [r"map back: 32 candidates", r"$s(c)=\sum_k\log p_k(\mathrm{path}(\cdot))$", r"$\hat q=\arg\max_c s(c)$"],
                    ORBIT, ORBIT_FILL, tsize=19, lsize=15, buff=0.08, icon=False)
        orb.next_to(beam, RIGHT, buff=0.45)
        poel.next_to(VGroup(poe, beam), DOWN, buff=0.2).align_to(poe, LEFT).shift(LEFT * 0.2)
        out = labelled("teaser_tless", 1.7, rf"$\hat q,\hat t$: $E_R={TE['err']:.1f}^\circ$", 18, INK, border=ORBIT)
        out.next_to(orb, RIGHT, buff=0.5)
        tl = tt(r"$\hat t$: per view, rolled back, averaged", 14, MUTED).next_to(out, DOWN, buff=0.08)
        leg_o = VGroup(VGroup(Line(ORIGIN, RIGHT * 0.35, stroke_width=4).set_color(OV_GT), tt("GT", 15)).arrange(RIGHT, buff=0.06),
                       VGroup(Line(ORIGIN, RIGHT * 0.35, stroke_width=4).set_color(OV_PRED), tt("prediction", 15)).arrange(RIGHT, buff=0.06)).arrange(RIGHT, buff=0.2)
        leg_o.next_to(tl, DOWN, buff=0.08)

        # ------------------------------------------------------------------ arrows
        X_IN = bb.get_left()[0]; XB = X_IN - 0.45
        bus = Line([XB, moge.get_y(), 0], [XB, dino.get_y(), 0], stroke_width=4).set_color(INK)
        ar = VGroup(
            bus, arrow([XB, moge.get_y(), 0], moge.get_left(), INK, sw=2.6, buff=0.02), arrow([XB, dino.get_y(), 0], dino.get_left(), INK, sw=2.6, buff=0.02),
            arrow(trv.get_right() + RIGHT * 0.05, [XB, moge.get_y() + 0.0, 0], TRAIN, sw=2.6, buff=0.03),
            arrow(tev.get_right() + RIGHT * 0.05, [XB, dino.get_y(), 0], ORBIT, sw=2.6, buff=0.03),
            arrow(moge.get_right(), feat_m[0].get_left(), INK, sw=2.4), arrow(dino.get_right(), feat_d[0].get_left(), INK, sw=2.4),
            arrow(feat_m[0].get_right(), tm.get_left(), INK, sw=2.4), arrow(feat_d[0].get_right(), td.get_left(), INK, sw=2.4),
            arrow(hd.get_top(), loss.get_bottom(), TRAIN, sw=2.6, buff=0.03),
            arrow(tm.get_right() + DOWN * 0.3, poe.get_left() + UP * 0.1, ORBIT, sw=2.4),
            arrow(td.get_right() + DOWN * 0.3, poe.get_left() + DOWN * 0.1, ORBIT, sw=2.4),
            arrow(poe.get_right(), beam.get_left(), ORBIT, sw=2.4), arrow(beam.get_right(), orb.get_left(), ORBIT, sw=2.4),
            arrow(orb.get_right(), out[0].get_left(), ORBIT, sw=2.4),
            Line(trn.get_right() + RIGHT * 0.05, [out[0].get_x(), trn.get_y(), 0], stroke_width=4).set_color(ORBIT),
            arrow([out[0].get_x(), trn.get_y(), 0], [out[0].get_x(), leg_o.get_bottom()[1], 0], ORBIT, sw=2.4, buff=0.06),
        )

        # ------------------------------------------------------------------ legend
        leg = VGroup(VGroup(snowflake(0.3), tt("frozen", 17)).arrange(RIGHT, buff=0.08),
                     VGroup(flame(0.3), tt("trained", 17)).arrange(RIGHT, buff=0.08),
                     VGroup(Line(ORIGIN, RIGHT * 0.5, stroke_width=5).set_color(TRAIN), tt("training only", 17)).arrange(RIGHT, buff=0.08),
                     VGroup(Line(ORIGIN, RIGHT * 0.5, stroke_width=5).set_color(ORBIT), tt("test time only", 17)).arrange(RIGHT, buff=0.08),
                     VGroup(Line(ORIGIN, RIGHT * 0.5, stroke_width=5).set_color(INK), tt("both", 17)).arrange(RIGHT, buff=0.08)).arrange(RIGHT, buff=0.4)
        allm = [hd, TR, TE_, bb, fz, fzi, feat_m, feat_d, tm, td, trn, mem, hdl, opt, loss, poe, poel, beam, orb, out, tl, leg_o, ar]
        leg.next_to(Group(*allm), DOWN, buff=0.3)
        allm.append(leg)
        self.add(*allm)
        fit(self, allm, pad=0.3)
