"""Paper figure 1 (teaser): one frozen design for different known objects, rotation as a coarse-to-fine path, and orbit
voting against look-alike flips. Example images follow fixed rules (assets.json): the median-error test image of the
first-listed object of the overview rule per benchmark, and the fixed-rule SPEED+ orbit example; all are predictions of the
finalised systems (rotation and translation predicted, ground-truth crops).
Render:  python render.py paper_figures/fig_teaser.py FigTeaser -s -w -r 4000x1500 --video_dir <dir>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pfcommon import *  # noqa: F401,F403,E402
set_text_scale(1.45)                           # larger print text (same figure footprint)

AS = json.loads((ASSET / "assets.json").read_text()); TE = AS["teaser"]; O = AS["orbit"]
Q = quat([0.62, -0.35, 0.70], 2.25); CH, PATH = encode(Q, 5); NAMES = ["w", "x", "y", "z"]


class FigTeaser(PaperScene):
    def construct(self):
        H = 1.45
        # ---------------------------------------------------------------- (a) one frozen design, several objects
        ex = Group()
        for key, nm, im in (("tless", "T-LESS", "teaser_tless"), ("ycbv", "YCB-V", "teaser_ycbv"), ("symsolt", "SYMSOL-T", "teaser_symsolt"),
                            ("speedplus", "SPEED+", "orbit_final")):
            m = img(im, H, border=FROZEN)
            l = tt(rf"{nm}: {TE[key]['err']:.1f}$^\circ$", 19).next_to(m, DOWN, buff=0.08)
            ex.add(Group(m, l))
        ex.arrange_in_grid(2, 2, buff=0.22)
        fz = block("frozen, shared", [r"DINOv3-L (0.30\,B)", r"MoGe-2 (0.33\,B)"], FROZEN, FROZEN_FILL, frozen=True, tsize=22, lsize=19)
        fz.next_to(ex, RIGHT, buff=0.55)
        heads = VGroup()
        for i in range(4):
            heads.add(box(1.75, 0.5, DINO, DINO_FILL, r=0.07, sw=2.2))
        heads.arrange(DOWN, buff=0.12).next_to(fz, RIGHT, buff=0.55)
        hl = VGroup(*[tt(r"2.7--4.85\,M", 17).move_to(h) for h in heads])
        hlab = VGroup(flame(0.32), tt(r"trained heads,\\ one per object", 18, DINO, bold=True)).arrange(RIGHT, buff=0.08).next_to(heads, UP, buff=0.12)
        aa = VGroup(arrow(ex.get_right(), fz.get_left(), INK), *[arrow(fz.get_right(), h.get_left(), DINO, sw=1.6, buff=0.05) for h in heads])
        A = Group(ex, fz, heads, hl, hlab, aa)
        ta = tt("frozen backbones, small heads", 26, bold=True)

        # ---------------------------------------------------------------- (b) rotation as a coarse-to-fine path (2-D sketch of the octree)
        sq = VGroup(); dig = VGroup(); S0 = 1.5
        for l in range(4):
            s = Square(S0).set_stroke(LV[l] if l else CH4[CH], 3).set_fill(PAPER, 1)
            grid = VGroup(Line(s.get_top(), s.get_bottom(), stroke_width=1), Line(s.get_left(), s.get_right(), stroke_width=1)).set_color("#B8BEC8")
            k = PATH[l]; qx, qy = (k & 1), (k >> 1) & 1                     # the cell's (u_0, u_1) bits; u_2 not drawn
            sub = Square(S0 / 2).set_stroke(LV[l + 1], 3).set_fill(LV[l + 1], 0.3).move_to(s.get_center() + (S0 / 4) * np.array([2 * qx - 1, 2 * qy - 1, 0]))
            sq.add(VGroup(s, grid, sub))
            dig.add(tx(NAMES[CH] if l == 0 else str(PATH[l - 1]), 26, CH4[CH] if l == 0 else LV[l]))
        sq.arrange(RIGHT, buff=0.55)
        zl = VGroup(*[DashedLine(sq[i][2].get_corner(UR), sq[i + 1][0].get_corner(UL), dash_length=0.05, stroke_width=1.1).set_color(MUTED) for i in range(3)]
                    + [DashedLine(sq[i][2].get_corner(DR), sq[i + 1][0].get_corner(DL), dash_length=0.05, stroke_width=1.1).set_color(MUTED) for i in range(3)])
        for i, d in enumerate(dig):
            d.next_to(sq[i], DOWN, buff=0.12)
        more = tx(r"\cdots", 30, MUTED).next_to(sq, RIGHT, buff=0.2)
        pt = tx(r"\mathrm{path}(q)=(" + NAMES[CH] + r";\," + ",".join(map(str, PATH)) + r")", 26)
        info = tt(r"4 charts $\times$ $8^5$ $=$ 131{,}072 leaves; mean leaf error $2.51^\circ$\\ (2-D sketch of the 3-D cells)", 19, MUTED)
        B = VGroup(VGroup(sq, zl, dig, more), pt, info).arrange(DOWN, buff=0.3)
        tb = tt("rotation as a coarse-to-fine path", 26, bold=True)

        # ---------------------------------------------------------------- (c) orbit voting
        views = Group(*[img(f"orbit_view{k}", 1.15, border=["#4361EE", "#7209B7", "#F77F00", "#0F9D8A"][k]) for k in range(4)]).arrange_in_grid(2, 2, buff=0.08)
        vl = tt(r"4 rolled views", 18, MUTED).next_to(views, DOWN, buff=0.08)
        g1 = img("orbit_greedy", 1.8, border=FLIP); g1l = tt(rf"1 view: {O['err_greedy']:.0f}$^\circ$ (flip)", 19, FLIP).next_to(g1, DOWN, buff=0.08)
        g2 = img("orbit_final", 1.8, border=ORBIT); g2l = tt(rf"orbit voting: {O['err_final']:.1f}$^\circ$", 19, ORBIT, bold=True).next_to(g2, DOWN, buff=0.08)
        C = Group(Group(g1, g1l), Group(views, vl), Group(g2, g2l)).arrange(RIGHT, buff=0.45)
        ca = VGroup(arrow(g1.get_right(), views.get_left(), MUTED, sw=2), arrow(views.get_right(), g2.get_left(), ORBIT, sw=2.4))
        cap = tt(r"one example image; candidates are scored across all four views", 19, MUTED).next_to(C, DOWN, buff=0.25)
        C = Group(C, ca, cap)
        tc = tt("orbit voting selects the consistent mode", 26, bold=True)

        row = Group(A, B, C).arrange(RIGHT, buff=0.9, aligned_edge=DOWN)
        B.align_to(A, DOWN).shift(UP * 0.35)
        ys = row.get_top()[1] + 0.45
        for t_, p_ in ((ta, A), (tb, B), (tc, C)):
            t_.move_to([0, ys, 0]).align_to(p_, LEFT)
        L = [letter(c).next_to(t_, LEFT, buff=0.12) for c, t_ in zip("abc", (ta, tb, tc))]
        leg = VGroup(VGroup(Line(ORIGIN, RIGHT * 0.4, stroke_width=4).set_color(OV_GT), tt("GT", 18)).arrange(RIGHT, buff=0.08),
                     VGroup(Line(ORIGIN, RIGHT * 0.4, stroke_width=4).set_color(OV_PRED), tt(r"prediction (rotation and translation)", 18)).arrange(RIGHT, buff=0.08),
                     tt(r"$E_R$: median-error test image (SPEED+: the example of (c))", 18, MUTED)).arrange(RIGHT, buff=0.35)
        leg.next_to(row, DOWN, buff=0.35).align_to(A, LEFT)
        allm = [row, ta, tb, tc, *L, leg]
        self.add(*allm)
        fit(self, allm, pad=0.3)
