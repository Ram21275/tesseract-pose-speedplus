"""S06: the tree Transformer head, geodesic soft targets and the per-level product of experts.
Computed: the example rotation's chart/path (encode), every cell centre (decode via `centre`), the geodesic distance
delta from q to each candidate centre and the exact soft targets softmax(-(delta/tau_l)^2) with the code's TAU_DEG
(src/tfpose/phase3.py; cross-checked against phase3.soft_targets to 2e-8). Attention weights and the DINO/MoGe
probabilities of b4 are illustrative and labelled so."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa: F401,F403,E402


def Tx(*a, color=None, **k):
    """Tex with a colour (the color kwarg is ignored by this manimgl build)."""
    t = Tex(*a, **k)
    return t.set_color(color) if color is not None else t

SC = "S06_TreeHead"
TAU = [39.27, 39.27, 19.95, 10.01, 5.01, 2.51]          # phase3.TAU_DEG (code values, used for every computation)
TAU_TXT = ["39.3", "39.3", "20.0", "10.0", "5.0", "2.5"]  # the same widths as printed in the paper
Q = np.array([0.6605, 0.6182, -0.1147, 0.4104]); Q = Q / np.linalg.norm(Q)   # example rotation
CHART, PATH = encode(Q, 5)
NAMES = ["w", "x", "y", "z"]


def level_targets(q):
    """Per level: candidate centres' geodesic distances delta (deg) and the exact soft targets."""
    c, p = encode(q, 5)
    out = []
    cands = [centre(k, []) for k in range(4)]
    for l in range(6):
        if l > 0:
            cands = [centre(c, p[:l - 1] + [k]) for k in range(8)]
        d = np.array([geodesic_deg(q, x) for x in cands])
        t = np.exp(-(d / TAU[l]) ** 2); t = t / t.sum()
        out.append((d, t))
    return out


LT = level_targets(Q)
TRUE = [CHART] + PATH
assert all(int(np.argmax(LT[l][1])) == TRUE[l] for l in range(6))


def label(s, size=24, color=GREY_A):
    return Text(s, font_size=size).set_color(color)


def bars(vals, colors, x0, base_y, width=0.5, step=0.72, scale=3.4, minh=0.015):
    g = VGroup()
    for i, v in enumerate(vals):
        h = max(abs(v) * scale, minh)
        r = Rectangle(width=width, height=h).set_stroke(width=0).set_fill(colors[i % len(colors)], 0.95)
        if v >= 0:
            r.move_to([x0 + i * step, base_y, 0], aligned_edge=DOWN)
        else:
            r.move_to([x0 + i * step, base_y, 0], aligned_edge=UP)
        g.add(r)
    return g


class S06_TreeHead(Scene):
    default_camera_config = dict(background_color=BG)

    def construct(self):
        rng = np.random.default_rng(6)
        # ------------------------------------------------ layout
        GRID_C = np.array([-5.0, -0.35, 0])
        cell = 0.3
        mem = VGroup()
        inten = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                blob = np.exp(-((i - 3.6) ** 2 + (j - 4.2) ** 2) / 6.0)
                inten[i, j] = 0.25 + 0.6 * blob + 0.15 * rng.random()
                sq = Square(side_length=cell).set_stroke(C_DINO, 0.6, 0.5)
                sq.set_fill(interpolate_color(BG, C_DINO, inten[i, j]), 1.0)
                sq.move_to(GRID_C + np.array([(j - 3.5) * (cell + 0.05), (3.5 - i) * (cell + 0.05), 0]))
                mem.add(sq)
        mem_lab = label("frozen DINOv3 tokens  8×8", 22, C_DINO).next_to(mem, UP, buff=0.25)
        mem_lab2 = label("64 × 1024", 20, GREY_B).next_to(mem, DOWN, buff=0.2)

        COL_X = -1.75
        tok_y = [2.45 - 0.8 * k for k in range(6)]

        def token(txt, k, color):
            r = RoundedRectangle(width=1.55, height=0.56, corner_radius=0.12).set_stroke(color, 2).set_fill(color, 0.18)
            r.move_to([COL_X, tok_y[k], 0])
            t = Tx(txt, font_size=26).move_to(r)
            return VGroup(r, t)

        BX0, BASE = 0.95, -2.35

        # ------------------------------------------------ b1: encoder over memory, root token, chart probabilities
        b = Beat(self, f"{SC}_b1")
        b.play(LaggedStartMap(FadeIn, mem, lag_ratio=0.015, scale=0.6), FadeIn(mem_lab), FadeIn(mem_lab2), frac=0.17)
        enc_lab = label("1 encoder layer", 20, GREY_A).next_to(mem_lab2, DOWN, buff=0.15)
        # encoder pass: a brightness wave sweeping the memory (every token mixes with every other)
        wave = [sq.animate.set_fill(interpolate_color(BG, WHITE, 0.35 + 0.5 * inten[k // 8, k % 8]))
                for k, sq in enumerate(mem)]
        back = [sq.animate.set_fill(interpolate_color(BG, C_DINO, inten[k // 8, k % 8])) for k, sq in enumerate(mem)]
        b.play(LaggedStart(*wave, lag_ratio=0.01), FadeIn(enc_lab), frac=0.1)
        b.play(LaggedStart(*back, lag_ratio=0.01), frac=0.07)
        dec_lab = label("causal decoder", 22, C_PRED).move_to([COL_X, 3.2, 0])
        root = token(r"\text{root}", 0, C_PRED)
        b.play(FadeIn(dec_lab), GrowFromCenter(root), frac=0.08)
        # cross-attention lines (illustrative weights)
        w = inten.flatten() ** 3; w = w / w.max()

        def att_lines(tok, seed):
            r2 = np.random.default_rng(seed)
            ww = np.clip(w * (0.6 + 0.6 * r2.random(64)), 0, 1)
            return VGroup(*[Line(tok[0].get_left(), sq.get_right() if True else sq.get_center(), stroke_width=1.6,
                                 color=C_PRED).set_stroke(opacity=0.08 + 0.85 * ww[k]) for k, sq in enumerate(mem)])

        al = att_lines(root, 1)
        att_lab = label("cross-attention (weights illustrative)", 18, GREY_B).next_to(mem, RIGHT, buff=0.15).shift(DOWN * 1.95)
        b.play(ShowCreation(al, lag_ratio=0.01), FadeIn(att_lab), frac=0.17)
        # chart output: exact root soft target for this q
        d0, t0 = LT[0]
        cb = bars(t0, CHART_COLORS, BX0 + 1.1, BASE, width=0.85, step=1.25)
        cnames = VGroup(*[Tx(NAMES[i], font_size=34, color=CHART_COLORS[i]).next_to(cb[i], DOWN, buff=0.12)
                          .set_y(BASE - 0.28) for i in range(4)])
        cvals = VGroup(*[DecimalNumber(t0[i], num_decimal_places=3, font_size=24).next_to(cb[i], UP, buff=0.1) for i in range(4)])
        axis = Line([BX0 - 0.3, BASE, 0], [BX0 + 5.6, BASE, 0], stroke_width=2, color=GREY_B)
        out_lab = Tx(r"\text{chart } c\text{: 4 logits}", font_size=32).move_to([BX0 + 2.95, 2.75, 0])
        tgt_lab = label("bars: exact soft target for this q", 18, GREY_B).next_to(out_lab, DOWN, buff=0.12)
        flow = Arrow(root.get_right(), [BX0 - 0.35, tok_y[0], 0], buff=0.1, stroke_width=3, color=C_PRED)
        b.play(GrowArrow(flow), ShowCreation(axis), FadeIn(out_lab), frac=0.1)
        b.play(LaggedStart(*[GrowFromEdge(r, DOWN) for r in cb], lag_ratio=0.15), FadeIn(cnames), FadeIn(tgt_lab), frac=0.16)
        b.play(FadeIn(cvals), cb[CHART].animate.set_stroke(WHITE, 3), frac=0.08)
        b.done()

        # ------------------------------------------------ b2: autoregressive unrolling along the real path
        b = Beat(self, f"{SC}_b2")
        path_tex = Tx(r"\text{path} = (" + NAMES[CHART] + r";\ " + r",\ ".join(str(k) for k in PATH) + ")", font_size=38)
        path_tex.move_to([BX0 + 2.95, 3.45, 0])
        # sub-mobjects: "path=(" ... we reveal pieces progressively by colouring
        path_tex.set_opacity(0.0)
        self.add(path_tex)
        chosen = [cb[CHART]]
        b.play(FadeOut(cvals), FadeOut(tgt_lab), al.animate.set_stroke(opacity=0.03), frac=0.06)
        prev_bars, prev_names = cb, cnames
        hist = VGroup()
        per = (0.86 * b.T) / 5
        prefix = []
        for l in range(1, 6):
            prefix_txt = NAMES[CHART] + (";" + ",".join(str(k) for k in prefix) if prefix else "")
            tk = token(r"\ell{=}" + str(l) + r"\ \ c(" + prefix_txt + ")", l, LEVEL_COLORS[l])
            tk[1].scale(0.8 if l < 4 else 0.7)
            # chosen bar flies into the token slot
            ghost = chosen[-1].copy()
            new_al = att_lines(tk, 10 + l)
            d, t = LT[l]
            nb = bars(t, [LEVEL_COLORS[l]] * 8, BX0, BASE, width=0.5, step=0.72)
            nnames = VGroup(*[Tx(str(k), font_size=28).next_to(nb[k], DOWN, buff=0.1).set_y(BASE - 0.25) for k in range(8)])
            new_out = Tx(r"\text{level } \ell=" + str(l) + r"\text{: 8 children}", font_size=32).move_to(out_lab)
            b.play(ReplacementTransform(ghost, tk[0]), FadeIn(tk[1]), Transform(out_lab, new_out),
                   FadeOut(prev_bars), FadeOut(prev_names), flow.animate.put_start_and_end_on(tk.get_right(), [BX0 - 0.35, tk.get_y(), 0]),
                   rt=per * 0.35)
            b.play(ShowCreation(new_al, lag_ratio=0.0), rt=per * 0.2)
            b.play(LaggedStart(*[GrowFromEdge(r, DOWN) for r in nb], lag_ratio=0.06), FadeIn(nnames),
                   new_al.animate.set_stroke(opacity=0.03), rt=per * 0.3)
            k = PATH[l - 1]
            b.play(nb[k].animate.set_stroke(WHITE, 3), nnames[k].animate.set_color(LEVEL_COLORS[l]), rt=per * 0.15)
            hist.add(new_al)
            chosen.append(nb[k]); prefix.append(k)
            prev_bars, prev_names = nb, nnames
        b.play(path_tex.animate.set_opacity(1.0), frac=0.06)
        b.done()

        # ------------------------------------------------ b3: soft targets, tau halves level by level
        b = Beat(self, f"{SC}_b3")
        keep = VGroup(path_tex)
        b.play(*[FadeOut(m) for m in self.mobjects if m is not path_tex], path_tex.animate.scale(0.8).to_corner(UR, buff=0.3),
               frac=0.06)
        eq = Tx(r"\text{target} \;\propto\; \exp\!\big(-(\delta/\tau_\ell)^2\big)", font_size=48).to_corner(UL, buff=0.35)
        dl = label("δ = angle between true q and a child's centre", 20, GREY_B).next_to(eq, DOWN, aligned_edge=LEFT, buff=0.15)
        b.play(Write(eq), frac=0.1)
        b.play(FadeIn(dl), frac=0.04)
        # log-scale delta axis: halving tau is a constant shift left
        AX_L, AX_R, AX_Y, AX_H = -6.4, 1.2, -2.6, 3.6
        lmin, lmax = np.log2(1.0), np.log2(200.0)

        def X(dg):
            return AX_L + (np.log2(max(dg, 1.0)) - lmin) / (lmax - lmin) * (AX_R - AX_L)

        ax = Line([AX_L, AX_Y, 0], [AX_R + 0.2, AX_Y, 0], color=GREY_B, stroke_width=2)
        ticks = VGroup()
        for v in [2.5, 5, 10, 20, 40, 80, 160]:
            ticks.add(Line([X(v), AX_Y - 0.08, 0], [X(v), AX_Y + 0.08, 0], color=GREY_B, stroke_width=2))
            tt = Tx(f"{v:g}", font_size=24).move_to([X(v), AX_Y - 0.32, 0])
            deg = Tx(r"^\circ", font_size=24).next_to(tt, RIGHT, buff=0.02).align_to(tt, UP)
            ticks.add(tt, deg)
        ax_lab = Tx(r"\delta\ (\text{log scale})", font_size=28).move_to([(AX_L + AX_R) / 2, AX_Y - 0.75, 0])
        yax = Line([AX_L, AX_Y, 0], [AX_L, AX_Y + AX_H + 0.2, 0], color=GREY_B, stroke_width=2)
        b.play(ShowCreation(ax), ShowCreation(yax), FadeIn(ticks), FadeIn(ax_lab), frac=0.07)

        def curve(l):
            xs = np.linspace(1.0, 200.0, 400)
            pts = [[X(x), AX_Y + AX_H * np.exp(-(x / TAU[l]) ** 2), 0] for x in np.geomspace(1.0, 200.0, 300)]
            return VMobject().set_points_smoothly(pts).set_stroke(LEVEL_COLORS[l], 4)

        def stems(l):
            d, t = LT[l]
            g = VGroup()
            for k, dk in enumerate(d):
                y = AX_Y + AX_H * np.exp(-(dk / TAU[l]) ** 2)
                col = C_GT if k == TRUE[l] else LEVEL_COLORS[l]
                g.add(VGroup(Line([X(dk), AX_Y, 0], [X(dk), y, 0], stroke_width=2, color=col).set_stroke(opacity=0.7),
                             Dot([X(dk), y, 0], radius=0.07, fill_color=col)))
            return g

        def tau_mark(l):
            x = X(TAU[l])
            ln = DashedLine([x, AX_Y, 0], [x, AX_Y + AX_H * np.exp(-1), 0], color=WHITE, stroke_width=2)
            tx = Tx(r"\tau_{" + str(l) + "}=" + TAU_TXT[l] + r"^\circ", font_size=30, color=LEVEL_COLORS[l])
            tx.next_to([x, AX_Y + AX_H * np.exp(-1), 0], UR, buff=0.12)
            return VGroup(ln, tx)

        # right: normalised targets
        RB0, RBASE = 3.0, -2.6

        def rbars(l):
            d, t = LT[l]
            n = len(t)
            g = bars(t, [LEVEL_COLORS[l]] * n, RB0 + (0.0 if n == 8 else 0.9), RBASE, width=0.42, step=0.5 if n == 8 else 0.85,
                     scale=3.6)
            g[TRUE[l]].set_stroke(C_GT, 3)
            nm = VGroup(*[(Tx(NAMES[k], font_size=24) if n == 4 else Tx(str(k), font_size=24)).next_to(g[k], DOWN, buff=0.08)
                          .set_y(RBASE - 0.22) for k in range(n)])
            return VGroup(g, nm)

        rax = Line([RB0 - 0.35, RBASE, 0], [RB0 + 3.9, RBASE, 0], color=GREY_B, stroke_width=2)
        rlab = label("normalised target", 20, GREY_B).move_to([RB0 + 1.75, 1.6, 0])
        lvl_lab = Tx(r"\text{root (chart)}", font_size=34, color=LEVEL_COLORS[0]).move_to([RB0 + 1.75, 2.15, 0])
        cv, st, tm, rb = curve(0), stems(0), tau_mark(0), rbars(0)
        b.play(ShowCreation(cv), FadeIn(st, lag_ratio=0.1), ShowCreation(rax), FadeIn(rlab), FadeIn(lvl_lab), frac=0.09)
        b.play(FadeIn(tm), LaggedStart(*[GrowFromEdge(r, DOWN) for r in rb[0]], lag_ratio=0.1), FadeIn(rb[1]), frac=0.07)
        per = 0.47 * b.T / 5
        for l in range(1, 6):
            nl = Tx(r"\text{level } \ell=" + str(l), font_size=34, color=LEVEL_COLORS[l]).move_to(lvl_lab)
            ncv, nst, ntm, nrb = curve(l), stems(l), tau_mark(l), rbars(l)
            b.play(Transform(cv, ncv), Transform(tm, ntm), FadeOut(st), FadeIn(nst), Transform(lvl_lab, nl),
                   FadeOut(rb), FadeIn(nrb), rt=per * 0.75)
            b.wait(per * 0.25)
            st, rb = nst, nrb
        # all six widths together: the same bump marching left by one octave per level
        ghosts = VGroup(*[curve(l).set_stroke(opacity=0.45, width=2.5) for l in range(6)])
        halv = label("τ halves with the cell size", 24, WHITE).move_to([X(10), AX_Y + AX_H + 0.35, 0])
        b.play(FadeIn(ghosts, lag_ratio=0.2), FadeIn(halv), frac=0.08)
        b.done()

        # ------------------------------------------------ b4: per-level product of experts (illustrative numbers)
        b = Beat(self, f"{SC}_b4")
        b.play(*[FadeOut(m) for m in self.mobjects], frac=0.05)
        poe = Tx(r"\log p^{(\ell)}\propto\log p^{(\ell)}_{\text{DINO}}+\log p^{(\ell)}_{\text{MoGe}}", font_size=46)
        poe.to_edge(UP, buff=0.3)
        ill = label("illustrative probabilities, one level", 20, GREY_B).next_to(poe, DOWN, buff=0.12)
        pd = np.array([0.05, 0.30, 0.08, 0.35, 0.04, 0.06, 0.07, 0.05])
        pm = np.array([0.04, 0.10, 0.06, 0.45, 0.05, 0.12, 0.10, 0.08])
        pf = pd * pm; pf = pf / pf.sum()
        Z = -0.2                     # zero line of every chart
        PS, LS = 2.8, 0.42          # probability scale, log scale
        W, ST = 0.32, 0.44
        XD, XM, XF = -5.75, -1.7, 2.35

        def zl(x0):
            return Line([x0 - 0.3, Z, 0], [x0 + 7 * ST + 0.3, Z, 0], color=GREY_B, stroke_width=1.5)

        def ids(x0, y):
            return VGroup(*[Tx(str(k), font_size=20, color=GREY_B).move_to([x0 + k * ST, y, 0]) for k in range(8)])

        bd = bars(pd, [C_DINO], XD, Z, width=W, step=ST, scale=PS)
        bm = bars(pm, [C_MOGE], XM, Z, width=W, step=ST, scale=PS)
        ld = label("DINOv3 tree", 24, C_DINO).move_to([XD + 3.5 * ST, 2.45, 0])
        lm = label("MoGe-2 tree", 24, C_MOGE).move_to([XM + 3.5 * ST, 2.45, 0])
        lf = label("fused", 24, C_POE).move_to([XF + 3.5 * ST, 2.45, 0])
        b.play(Write(poe), FadeIn(ill), frac=0.12)
        b.play(LaggedStart(*[GrowFromEdge(r, DOWN) for r in bd], lag_ratio=0.05),
               LaggedStart(*[GrowFromEdge(r, DOWN) for r in bm], lag_ratio=0.05), FadeIn(ld), FadeIn(lm),
               ShowCreation(zl(XD)), ShowCreation(zl(XM)), FadeIn(ids(XD, Z - 0.2)), FadeIn(ids(XM, Z - 0.2)), frac=0.12)
        pl = Tx("p", font_size=30).move_to([XD - 0.6, Z + 0.8, 0])
        self.add(pl)
        # to log-probabilities: bars flip below the zero line
        lbd = bars(np.log(pd), [C_DINO], XD, Z - 0.35, width=W, step=ST, scale=LS)
        lbm = bars(np.log(pm), [C_MOGE], XM, Z - 0.35, width=W, step=ST, scale=LS)
        logl = Tx(r"\log p", font_size=30).move_to([XD - 0.65, Z - 1.2, 0])
        b.play(Transform(bd, lbd), Transform(bm, lbm), Transform(pl, logl), frac=0.12)
        # add: per child, the MoGe log bar is stacked under the DINO log bar
        fd = bars(np.log(pd), [C_DINO], XF, Z - 0.35, width=W, step=ST, scale=LS)
        fm = VGroup()
        for k in range(8):
            r = Rectangle(width=W, height=abs(np.log(pm[k])) * LS).set_stroke(width=0).set_fill(C_MOGE, 0.95)
            r.move_to(fd[k].get_bottom(), aligned_edge=UP); fm.add(r)
        plus = Tx("+", font_size=44).move_to([(XM + 7 * ST + XF) / 2, Z - 1.0, 0])
        b.play(TransformFromCopy(bd, fd), TransformFromCopy(bm, fm), FadeIn(plus), FadeIn(lf),
               FadeIn(ids(XF, Z - 0.2)), frac=0.14)
        # the sum, one purple bar per child; then exponentiate and renormalise
        sums = np.log(pd) + np.log(pm)
        sb = bars(sums, [C_POE], XF, Z - 0.35, width=W, step=ST, scale=LS)
        b.play(*[ReplacementTransform(VGroup(fd[k], fm[k]), sb[k]) for k in range(8)], frac=0.1)
        fb = bars(pf, [C_POE], XF, Z, width=W, step=ST, scale=PS)
        ren = Tx(r"\text{exp, renormalise}", font_size=26, color=C_POE).move_to([XF + 3.5 * ST, Z - 0.75, 0])
        b.play(ReplacementTransform(sb, fb), ShowCreation(zl(XF)), FadeIn(ren), frac=0.13)
        kf = int(np.argmax(pf))
        nums = VGroup(*[DecimalNumber(pf[k], num_decimal_places=2, font_size=20).next_to(fb[k], UP, buff=0.06)
                        for k in range(8) if pf[k] > 0.05])
        nd = DecimalNumber(pd[kf], num_decimal_places=2, font_size=20).next_to(bd[kf], DOWN, buff=0.06)
        b.play(fb[kf].animate.set_stroke(WHITE, 3), FadeIn(nums), frac=0.1)
        # beam search continues on the fused distribution: the winning child is expanded
        nxt = bars(np.array([0.08, 0.12, 0.1, 0.11, 0.18, 0.15, 0.13, 0.13]), [C_POE], XF + 1.6, Z - 3.45, width=0.18, step=0.26,
                   scale=4.0)
        nxt.set_opacity(0.55)
        arr = Arrow(fb[kf].get_bottom() + DOWN * 0.95, nxt.get_top() + UP * 0.05, buff=0.05, stroke_width=3, color=C_POE)
        nl = label("next level: fuse again", 18, GREY_B).next_to(nxt, LEFT, buff=0.25)
        b.play(GrowArrow(arr), LaggedStart(*[GrowFromEdge(r, DOWN) for r in nxt], lag_ratio=0.05), FadeIn(nl), frac=0.12)
        b.done()
