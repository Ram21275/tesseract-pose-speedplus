import manim_compat  # noqa: F401
from manimlib import *


class Smoke(ThreeDScene):
    def construct(self):
        eq = Tex(r"q' = \mathrm{Exp}(-\theta\, d)\otimes q", font_size=60).to_edge(UP)
        eq.fix_in_frame()
        cube = Cube(side_length=2, color=BLUE, opacity=0.5)
        self.frame.set_euler_angles(theta=-30 * DEGREES, phi=70 * DEGREES)
        self.play(Write(eq), ShowCreation(cube), run_time=1)
        self.play(Rotate(cube, PI / 2, axis=OUT), self.frame.animate.increment_theta(40 * DEGREES), run_time=1)
