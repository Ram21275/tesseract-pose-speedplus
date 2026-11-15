"""Paper figure: the tree Transformer, beam search over the tesseract tree, and multi-path symmetric targets.
(a) the architecture as built in src/tfpose/phase3.py (HierTransformer): frozen 8x8 tokens -> LN + linear + 2-D positions ->
    1 encoder layer -> memory; causal 2-layer decoder over [ROOT, cell_1..cell_5], each token = level embedding + Fourier(cell
    centre); root head 4 logits, child head 8 logits.
(b) beam search (src/tfpose/fusion.py fused_beam): 4 charts, then every kept node expands to 8 children and the best 8 are kept.
(c) multi-path targets (Sec. 3.4, Eq. symloss) for a real T-LESS training-frame label (object 5, 2-fold symmetry); the two
    equivalent labels and their paths come from assets.json (computed with the project encoder).
Render:  python render.py paper_figures/fig_tree.py FigTree -s -w -r 3600x1400 --video_dir <dir>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pfcommon import *  # noqa: F401,F403,E402
set_text_scale(1.45)                           # larger print text (same figure footprint)

AS = json.loads((ASSET / "assets.json").read_text())
MP = AS["multipath"]; NAMES = ["w", "x", "y", "z"]


def token(label, color, fill, w=0.95, h=0.62, size=20):
    b = box(w, h, color, fill, r=0.08, sw=2.5); return VGroup(b, tx(label, size).move_to(b))


class FigTree(PaperScene):
    def construct(self):
        # ---------------------------------------------------------------- (a) tree Transformer
        grid = VGroup(*[Square(0.19).set_fill(FROZEN_FILL, 1).set_stroke(FROZEN, 1.2) for _ in range(64)]).arrange_in_grid(8, 8, buff=0.02)
        gl = tt(r"frozen tokens\\ $8{\times}8{\times}1024$", 20, MUTED).next_to(grid, DOWN, buff=0.15)
        gl = VGroup(snowflake(0.3), gl).arrange(RIGHT, buff=0.08).next_to(grid, DOWN, buff=0.15)
        enc = block("encoder", [r"LN, linear $\to 256$", "2-D positions, 1 layer"], DINO, DINO_FILL, lsize=19, tsize=22, icon="fire")
        enc.next_to(grid, RIGHT, buff=0.55)
        mem = VGroup(*[Square(0.16).set_fill(DINO_FILL, 1).set_stroke(DINO, 1.2) for _ in range(8)]).arrange(RIGHT, buff=0.03)
        meml = tt("memory (64 tokens)", 19, DINO)
        seq = [token(r"\mathrm{ROOT}", DINO, PAPER)] + [token(rf"\mathrm{{cell}}_{l}", LV[l - 1] if l > 1 else CH4[2], PAPER) for l in range(1, 6)]
        seq = VGroup(*seq).arrange(RIGHT, buff=0.32)
        dec = box(seq.get_width() + 0.5, 1.45, DINO, DINO_FILL)
        decg = VGroup(dec, seq.move_to(dec).shift(DOWN * 0.15), badge(dec, flame()))
        decg.next_to(enc, RIGHT, buff=0.75).align_to(enc, UP).shift(UP * 0.1)
        dect = tt(r"causal decoder, 2 layers (4 heads, FFN 512)", 18, DINO).move_to(dec).align_to(dec, UP).shift(DOWN * 0.1)
        heads = VGroup(); outs = VGroup()
        for i, t in enumerate(seq):
            lab = r"4 \text{ charts}" if i == 0 else r"8 \text{ children}"
            hb = token(lab, INK, PAPER, w=1.12, h=0.44, size=14).next_to(dec, UP, buff=0.42).set_x(t.get_x())
            heads.add(hb); outs.add(arrow(dec.get_top() * [0, 1, 0] + [t.get_x(), 0, 0], hb.get_bottom(), INK, buff=0.04, sw=2.2))
        ex = MP["path"][0]
        picks = VGroup(*[tx(d, 24, LV[i] if i else CH4[MP["chart"][0]]).next_to(heads[i], UP, buff=0.1)
                         for i, d in enumerate([NAMES[MP["chart"][0]]] + [str(k) for k in ex])])
        fb = VGroup()
        for i in range(5):                                         # decision i is embedded as token i+1
            a = picks[i].get_right() + RIGHT * 0.04
            b = heads[i + 1].get_left() + UP * 0.12
            fb.add(arrow(a, b, MUTED, buff=0.03, sw=1.6, arc=-1.0))
        fbl = tt(r"next token $=$ level embedding $+$ Fourier(chosen cell centre)", 18, MUTED).next_to(picks, UP, buff=0.2).align_to(decg, LEFT)
        mem.next_to(dec, DOWN, buff=0.85).set_x(dec.get_x() - 1.2); meml.next_to(mem, DOWN, buff=0.1)
        xatt = VGroup(*[DashedLine(mem.get_top() + UP * 0.03, [t.get_x(), dec.get_bottom()[1], 0], dash_length=0.06, stroke_width=1.1).set_color(DINO)
                        for t in seq])
        xl = tt("cross-attention", 18, DINO).next_to(mem, RIGHT, buff=0.35)
        a1 = arrow(grid.get_right(), enc.get_left(), INK); a2 = arrow(enc.get_bottom(), mem.get_left() + LEFT * 0.05, INK, arc=0.6)
        size_l = tt(r"2.4\,M trained parameters per branch", 20, bold=True).next_to(meml, DOWN, buff=0.3).align_to(decg, RIGHT)
        A = VGroup(grid, gl, enc, mem, meml, a1, a2, decg, dect, heads, outs, picks, fb, fbl, xatt, xl, size_l)
        ta = tt("tree Transformer: autoregressive path decoding", 26, bold=True).next_to(A, UP, buff=0.35).align_to(grid, LEFT)

        # ---------------------------------------------------------------- (b) beam search over the tree
        rng = np.random.default_rng(3)                             # fixed layout only (which nodes are kept is schematic)
        cols = VGroup(); kept_cols = []; X0 = A.get_right()[0] + 1.6; dy = 0.088
        n_prev = 4; kept = list(range(4))
        lev_x = [X0 + 1.0 * l for l in range(6)]
        nodes = []
        for l in range(6):
            n = 4 if l == 0 else 8 * len(kept)
            ys = (np.arange(n) - (n - 1) / 2) * (dy if n > 8 else 0.42)
            keep = list(range(n)) if l == 0 else sorted(rng.choice(n, 8, replace=False).tolist())
            col = VGroup()
            for j in range(n):
                on = j in keep
                d = Dot(np.array([lev_x[l], ys[j], 0]), radius=0.05 if n > 8 else 0.08)
                d.set_fill(LV[l] if (on and l) else (CH4[j] if l == 0 else "#CDD2DA"), 1)
                col.add(d)
            nodes.append((col, keep, n)); cols.add(col); kept = keep
        edges = VGroup()
        for l in range(1, 6):
            pc, pk, _ = nodes[l - 1]; cc, ck, n = nodes[l]
            for j in range(n):
                par = pk[j // 8]
                on = j in ck
                edges.add(Line(pc[par].get_center(), cc[j].get_center(), stroke_width=1.6 if on else 0.4).set_color(LV[l] if on else "#DADDE3"))
        B = VGroup(edges, cols)
        B.move_to([X0 + 2.0, A.get_y() - 0.1, 0]); B.set_height(A.get_height() * 0.92)
        lx = VGroup(*[tx(str(l), 22, MUTED).next_to(B, DOWN, buff=0.15).set_x(cols[l].get_x()) for l in range(6)])
        lx.add(tx(r"\text{level } \ell:", 22, MUTED).next_to(lx, LEFT, buff=0.2))
        nl = VGroup(*[tt(t, 20, MUTED).next_to(B, UP, buff=0.15).set_x(cols[l].get_x()) for l, t in enumerate(["4", "32", "64", "64", "64", "64"])])
        nlab = tt("expanded:", 20, MUTED).next_to(nl, LEFT, buff=0.2)
        bl = tt(r"keep the 8 most likely partial paths\\ per level (beam 8): 8 leaves per view", 19, MUTED)
        B.add(lx, nl, nlab)
        bl.next_to(B, DOWN, buff=0.25)
        tb = tt("beam search", 26, bold=True).next_to(B, UP, buff=0.3).align_to(B, LEFT)
        kk = VGroup(VGroup(Dot(radius=0.06).set_fill(LV[3], 1), tt("kept", 18)).arrange(RIGHT, buff=0.08),
                    VGroup(Dot(radius=0.06).set_fill("#CDD2DA", 1), tt("pruned", 18)).arrange(RIGHT, buff=0.08)).arrange(RIGHT, buff=0.3).next_to(tb, RIGHT, buff=0.4)
        B.add(bl)

        # ---------------------------------------------------------------- (c) multi-path symmetric targets
        im = img("teaser_tless", 1.75, border=FROZEN)
        iml = tt(r"T-LESS object 5\\ 2-fold symmetry", 18, MUTED).next_to(im, DOWN, buff=0.1)
        p1 = MP["path"][0]; p2 = MP["path"][1]; c1 = NAMES[MP["chart"][0]]; c2 = NAMES[MP["chart"][1]]
        def ptex(c, p, col):
            return tx(r"(" + c + r";\," + ",".join(map(str, p)) + ")", 26, col)
        r1 = ptex(c1, p1, PRED); r2 = ptex(c2, p2, PRED)
        q1 = tx(r"q", 26); q2 = tx(r"q\,s", 26)
        A0 = VGroup(tt(r"\textbf{single label}", 21), VGroup(q1.copy(), tx(r"\to", 24), r1.copy()).arrange(RIGHT, buff=0.15),
                    tt(r"the other, identical-looking\\ view may carry $q\,s$: contradictory", 17, FLIP)).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        A2 = VGroup(tt(r"\textbf{multi-path (ours)}", 21),
                    VGroup(q1, tx(r"\to", 24), r1, tx(r"\tfrac12", 24, MUTED)).arrange(RIGHT, buff=0.15),
                    VGroup(q2, tx(r"\to", 24), r2, tx(r"\tfrac12", 24, MUTED)).arrange(RIGHT, buff=0.15),
                    tt(r"both paths supervised; the tree\\ splits its mass between them", 17, ORBIT)).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        eq = tx(r"\mathcal{L}_{\mathrm{sym}}=\tfrac{1}{|\mathcal{S}'|}\textstyle\sum_{s\in\mathcal{S}'}\mathcal{L}_{\mathrm{GT}}(q\,s)", 26)
        right = VGroup(A0, A2, eq).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        C = Group(Group(im, iml), right).arrange(RIGHT, buff=0.35)
        C.next_to(B, RIGHT, buff=0.9).align_to(A, DOWN).shift(UP * 0.1)
        tc = tt("symmetric objects: all equivalent labels", 26, bold=True).next_to(C, UP, buff=0.3).align_to(C, LEFT)

        y = max(ta.get_y(), tb.get_y(), tc.get_y())
        for t_ in (ta, tb, tc):
            t_.set_y(y)
        kk.arrange(RIGHT, buff=0.3).next_to(B, UP, buff=0.1).align_to(B, RIGHT).shift(UP * 0.35)
        la, lb, lc = [letter(s).next_to(t_, LEFT, buff=0.12) for s, t_ in zip("abc", (ta, tb, tc))]
        A.add(dect)
        allm = [A, ta, la, B, tb, lb, kk, C, tc, lc]
        self.add(*allm)
        fit(self, allm, pad=0.3)
