"""Paper figure: orbit voting on a real SPEED+ lightbox image (the fixed-rule example of
paper/figures/qualitative/orbit_voting_speedplus: greedy wrong by >150 deg, orbit within 10 deg, median of that group).
Crops, overlays, the 32 candidates' orbit scores s(c) (Eq. orbit) and their errors come from assets.json (prep_assets.py),
i.e. from the saved predictions of the finalised SPEED+ system (distilled PoE students, pre-registered product rule).
Render:  python render.py paper_figures/fig_orbit.py FigOrbit -s -w -r 3600x1400 --video_dir <dir>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pfcommon import *  # noqa: F401,F403,E402

O = json.loads((ASSET / "assets.json").read_text())["orbit"]
VIEW = ["#4361EE", "#7209B7", "#F77F00", "#0F9D8A"]
ok = lambda e: e < 30


class FigOrbit(PaperScene):
    def construct(self):
        H = 1.75
        # ---------------------------------------------------------------- (a) rolled views and (b) their mapped-back top hypothesis
        cols = Group()
        for k in range(4):
            v = img(f"orbit_view{k}", H, border=VIEW[k])
            vl = tx(rf"\theta_{k + 1}={90 * k}^\circ", 24, VIEW[k]).next_to(v, UP, buff=0.1)
            h = img(f"orbit_top{k}", H, border=VIEW[k])
            e = O["top_err"][k]
            el = tt(rf"top-1: {e:.0f}$^\circ$ " + (r"\checkmark" if ok(e) else r"flip"), 22, ORBIT if ok(e) else FLIP)
            col = Group(vl, v, h, el)
            h.next_to(v, DOWN, buff=0.75); el.next_to(h, DOWN, buff=0.1)
            cols.add(col)
        cols.arrange(RIGHT, buff=0.28)
        arrs = VGroup(*[arrow(c[1].get_bottom(), c[2].get_top(), VIEW[k], buff=0.05, sw=2.2) for k, c in enumerate(cols)])
        mid = tt(r"beam search (8) in each view, mapped back: $c=\mathrm{Exp}(+\theta_k d)\otimes c_k$", 20, INK)
        bg = BackgroundRectangle(mid, fill_color=PAPER, fill_opacity=1, buff=0.06)
        midg = VGroup(bg, mid).move_to((cols[0][1].get_bottom() + cols[3][2].get_top()) / 2 * [0, 1, 0] + [cols.get_x(), 0, 0])
        ta = tt("rolled views and each view's best hypothesis", 26, bold=True).next_to(cols, UP, buff=0.3).align_to(cols, LEFT)
        A = Group(cols, arrs, midg)

        # ---------------------------------------------------------------- (c) orbit score of all 32 candidates
        s = np.array(O["score"]); d = s - s.max(); e = np.array(O["cand_err"]); win = O["winner"]
        W, Hc = 6.6, 3.0; ymin = float(d.min()); sx = W / 32
        axes = VGroup(Line(ORIGIN, RIGHT * W, stroke_width=2).set_color(INK))
        bars = VGroup()
        for i in range(32):
            h = max(-d[i] / -ymin * Hc, 0.03)
            b = Rectangle(width=sx * 0.72, height=h).set_stroke(width=0).set_fill(ORBIT if ok(e[i]) else FLIP, 0.9 if i != win else 1.0)
            b.move_to(RIGHT * (sx * (i + 0.5)), aligned_edge=UP)
            bars.add(b)
        grp = VGroup(*[VGroup(Line(RIGHT * sx * 8 * k + DOWN * (Hc + 0.12), RIGHT * sx * 8 * (k + 1) - RIGHT * 0.04 + DOWN * (Hc + 0.12), stroke_width=5).set_color(VIEW[k]),
                              tx(rf"{90 * k}^\circ", 20, VIEW[k]).move_to(RIGHT * sx * (8 * k + 4) + DOWN * (Hc + 0.36))) for k in range(4)])
        gl = tt("source view of the candidate", 18, MUTED).move_to(RIGHT * W / 2 + DOWN * (Hc + 0.7))
        yt = VGroup(*[VGroup(Line(LEFT * 0.08, ORIGIN, stroke_width=1.5).set_color(INK), tx(str(v), 18, MUTED).next_to(ORIGIN, LEFT, buff=0.12)).shift(DOWN * (-v / -ymin * Hc))
                      for v in (0, -20, -40)])
        yl = tx(r"s(c)-\max_{c'} s(c')", 22, INK).rotate(PI / 2).next_to(yt, LEFT, buff=0.15)
        star = Triangle().set_height(0.2).set_fill(GT, 1).set_stroke(width=0).rotate(PI).next_to(bars[win], UP, buff=0.06)
        wl = tt(rf"selected ({e[win]:.1f}$^\circ$)", 18, GT).next_to(star, UP, buff=0.06)
        key = VGroup(VGroup(Square(0.2).set_fill(ORBIT, 1).set_stroke(width=0), tt(r"$<30^\circ$ from GT", 18)).arrange(RIGHT, buff=0.08),
                     VGroup(Square(0.2).set_fill(FLIP, 1).set_stroke(width=0), tt(r"flip ($>150^\circ$)", 18)).arrange(RIGHT, buff=0.08)).arrange(RIGHT, buff=0.3)
        key.arrange(DOWN, buff=0.12, aligned_edge=LEFT).move_to(RIGHT * (W - 1.2) + DOWN * (Hc * 0.78))
        eq = tx(r"s(c)=\sum_{k=1}^{4}\log p_k\big(\mathrm{path}(\mathrm{Exp}(-\theta_k d)\otimes c)\big)", 24)
        C = VGroup(axes, bars, grp, gl, yt, yl, star, wl, key)
        eq.next_to(C, UP, buff=0.25).align_to(C, LEFT)
        C.add(eq)
        C.next_to(A, RIGHT, buff=0.8).align_to(A, DOWN).shift(UP * 0.35)
        tc = tt("each candidate is scored in all four views", 26, bold=True).next_to(C, UP, buff=0.3).align_to(C, LEFT)

        # ---------------------------------------------------------------- (d) result
        fin = img("orbit_final", 2.1, border=GT)
        fl = tt(rf"orbit voting: {O['err_final']:.1f}$^\circ$", 22, ORBIT, bold=True).next_to(fin, DOWN, buff=0.1)
        gr = img("orbit_greedy", 1.3, border=FROZEN)
        grl = tt(rf"1 view: {O['err_greedy']:.0f}$^\circ$", 20, FLIP).next_to(gr, DOWN, buff=0.08)
        D = Group(Group(fin, fl), Group(gr, grl)).arrange(DOWN, buff=0.35)
        D.next_to(C, RIGHT, buff=0.6).align_to(A, DOWN)
        legend = VGroup(VGroup(Line(ORIGIN, RIGHT * 0.4, stroke_width=4).set_color(OV_GT), tt("GT", 18)).arrange(RIGHT, buff=0.08),
                        VGroup(Line(ORIGIN, RIGHT * 0.4, stroke_width=4).set_color(OV_PRED), tt("prediction", 18)).arrange(RIGHT, buff=0.08)).arrange(RIGHT, buff=0.3)
        legend.next_to(D, DOWN, buff=0.2)
        td = tt("result", 26, bold=True).next_to(D, UP, buff=0.3).align_to(D, LEFT)

        y = max(ta.get_y(), tc.get_y(), td.get_y())
        for t_ in (ta, tc, td):
            t_.set_y(y)
        L = [letter(c).next_to(t_, LEFT, buff=0.12) for c, t_ in zip("abc", (ta, tc, td))]
        allm = [A, ta, C, tc, D, legend, td, *L]
        self.add(*allm)
        fit(self, allm, pad=0.3)
