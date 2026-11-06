"""Trusted mathematical animation primitives for source-grounded explainers."""
from __future__ import annotations

import json
import os
import re
import hashlib
import subprocess
import tempfile
from pathlib import Path

from manim import *
import numpy as np

BG = "#000000"
WHITE_SOFT = "#F5F5F5"
RED_VAR = "#ED6A76"
GREEN_VAR = "#7ED68C"
BLUE_VAR = "#61B9E7"
YELLOW_VAR = "#F3D65E"
PURPLE_VAR = "#B98ADE"
VARIABLE_COLORS = {
    "R_1": RED_VAR, "R_2": GREEN_VAR, "R_3": BLUE_VAR,
    "R_a": GREEN_VAR, "R_b": RED_VAR, "R_c": BLUE_VAR,
    "R_A": GREEN_VAR, "R_B": RED_VAR, "R_C": BLUE_VAR,
    "x": GREEN_VAR, "y": RED_VAR, "z": BLUE_VAR,
}


_MATH_COMMANDS = {
    "frac", "dfrac", "tfrac", "sqrt", "left", "right", "sum", "prod", "int",
    "sin", "cos", "tan", "log", "ln", "lim", "cdot", "times", "parallel",
    "Delta", "Omega", "alpha", "beta", "gamma", "theta", "pi", "infty",
    "vec", "hat", "bar", "overline", "underbrace", "overbrace", "text",
    "begin", "end", "bmatrix", "pmatrix", "mathbf", "mathbb", "mathrm",
    "leq", "geq", "neq", "approx", "to", "longrightarrow", "longleftrightarrow",
    "Rightarrow", "Leftrightarrow", "partial", "nabla", "in", "subset",
    "cdots", "vdots", "ldots", "quad", "qquad", "leftarrow", "rightarrow",
}


def math_tex(source: str, size: int = 44) -> SVGMobject:
    text = source.strip().strip("$")
    unknown = set(re.findall(r"\\([A-Za-z]+)", text)) - _MATH_COMMANDS
    if unknown:
        raise ValueError(f"Unsupported TeX command: {sorted(unknown)[0]}")
    for token, color in sorted(VARIABLE_COLORS.items(), key=lambda pair: -len(pair[0])):
        pattern = rf"(?<![A-Za-z\\]){re.escape(token)}(?![A-Za-z])"
        text = re.sub(pattern, lambda match: rf"\textcolor[HTML]{{{color[1:]}}}{{{match.group()}}}", text)
    document = (
        r"\documentclass[border=2pt]{standalone}" + "\n"
        r"\usepackage{amsmath,amssymb,xcolor}" + "\n"
        r"\begin{document}" + "\n"
        r"\color{white}$\displaystyle " + text + r"$" + "\n"
        r"\end{document}" + "\n"
    )
    cache = Path(tempfile.gettempdir()) / "vidframe-math-svg"
    cache.mkdir(exist_ok=True)
    key = hashlib.sha256(document.encode()).hexdigest()
    svg = cache / f"{key}.svg"
    if not svg.is_file():
        tex = cache / f"{key}.tex"
        tex.write_text(document, encoding="utf-8")
        latex = subprocess.run(["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error",
                                "-output-directory", str(cache), str(tex)],
                               capture_output=True, text=True)
        if latex.returncode:
            raise ValueError("Could not typeset equation: " + latex.stdout[-300:])
        converted = subprocess.run(["pdftocairo", "-svg", str(cache / f"{key}.pdf"), str(svg)],
                                   capture_output=True, text=True)
        if converted.returncode or not svg.is_file():
            raise RuntimeError("Could not convert equation to SVG: " + converted.stderr[-300:])
    result = SVGMobject(str(svg))
    result.scale(size / 80)
    return result


def fit(mob: Mobject, width: float = 12.1, height: float = 5.9) -> Mobject:
    if mob.width > width:
        mob.scale_to_fit_width(width)
    if mob.height > height:
        mob.scale_to_fit_height(height)
    return mob


def serif(text: str, size: int = 31, color: str = WHITE_SOFT) -> Text:
    return Text(text, font="DejaVu Serif", font_size=size, color=color)


def resistor(start, end, label: str, color: str = WHITE_SOFT, side: float = 1, offset=ORIGIN) -> VGroup:
    a, b = np.array(start), np.array(end)
    direction = (b - a) / np.linalg.norm(b - a)
    normal = np.array([-direction[1], direction[0], 0])
    center, half = (a + b) / 2, 0.43
    lead_a = Line(a, center - direction * half, color=WHITE_SOFT, stroke_width=3)
    lead_b = Line(center + direction * half, b, color=WHITE_SOFT, stroke_width=3)
    points = [center - direction * half]
    for i in range(1, 8):
        points.append(center + direction * (-half + i * 2 * half / 8) + normal * (0.12 if i % 2 else -0.12))
    points.append(center + direction * half)
    zigzag = VMobject(color=color, stroke_width=3).set_points_as_corners(points)
    tag = math_tex(label, 22).move_to(center + normal * side * 0.63 + offset)
    return VGroup(lead_a, zigzag, lead_b, tag)


def circuit(kind: str, active: set[str] | None = None) -> VGroup:
    active = active or set()
    a, b, c = UP * 1.75, LEFT * 1.85 + DOWN * 1.35, RIGHT * 1.85 + DOWN * 1.35
    nodes = VGroup(*[Dot(point, radius=0.075, color=WHITE_SOFT) for point in (a, b, c)])
    names = VGroup(math_tex("A", 23).move_to(a + UP * 0.38),
                   math_tex("B", 23).move_to(b + LEFT * 0.38),
                   math_tex("C", 23).move_to(c + RIGHT * 0.38))
    if kind == "delta":
        edges = [resistor(a, b, "R_3", BLUE_VAR, -1), resistor(b, c, "R_1", RED_VAR, -1),
                 resistor(c, a, "R_2", GREEN_VAR, -1)]
        keys = ["R_3", "R_1", "R_2"]
    else:
        o = DOWN * 0.15
        edges = [resistor(a, o, "R_a", GREEN_VAR, 1, LEFT * 0.12),
                 resistor(b, o, "R_b", RED_VAR, 1),
                 resistor(c, o, "R_c", BLUE_VAR, 1)]
        keys = ["R_a", "R_b", "R_c"]
        nodes.add(Dot(o, radius=0.075, color=WHITE_SOFT))
    for key, edge in zip(keys, edges):
        if active and key not in active:
            edge.set_opacity(0.25)
    return VGroup(*edges, nodes, names)


class ExplainerScene(Scene):
    def construct(self):
        payload = json.loads(Path(os.environ["VIDFRAME_EXPLAINER_PLAN"]).read_text())
        self.camera.background_color = BG
        for scene in payload["scenes"]:
            duration = max(2.0, float(scene["duration"]))
            blocks = self.visual_blocks(scene) or [(serif(scene["title"], 36), "write")]
            reveal = min(1.35, max(0.1, (duration - 0.55) / (len(blocks) * 2)))
            hold = max(0, (duration - 0.55 - reveal * len(blocks)) / len(blocks))
            shown = []
            for mob, action in blocks:
                fit(mob)
                if action == "write":
                    self.play(Write(mob), run_time=reveal)
                elif action == "create":
                    self.play(Create(mob), run_time=reveal)
                else:
                    self.play(FadeIn(mob), run_time=reveal)
                shown.append(mob)
                if hold > 0:
                    self.wait(hold)
            self.play(*[FadeOut(mob) for mob in shown], run_time=0.55)

    def visual_blocks(self, scene: dict) -> list[tuple[Mobject, str]]:
        mode = scene.get("visual_mode") or scene.get("kind")
        if mode in {"circuit_compare", "circuit_path"}:
            return self.circuit_blocks(scene, mode)
        if mode in {"formula_triplet", "equation_chain"}:
            return self.equation_blocks(scene)
        if mode == "matrix":
            return self.matrix_blocks(scene)
        if mode == "vector_field":
            return self.vector_blocks(scene)
        if mode == "graph":
            return self.graph_blocks(scene)
        if mode == "equation":
            return self.equation_blocks(scene)
        if mode == "diagram":
            return self.generic_diagram_blocks(scene)
        return self.label_blocks(scene)

    def heading(self, text: str, size: int = 31) -> Text:
        return fit(serif(text, size)).to_edge(UP, buff=0.38)

    def circuit_pair(self) -> tuple[VGroup, VGroup, MathTex]:
        delta = circuit("delta").scale(0.72).move_to(LEFT * 3.65 + DOWN * 0.30)
        star = circuit("star").scale(0.72).move_to(RIGHT * 3.65 + DOWN * 0.30)
        sign = math_tex(r"\longleftrightarrow", 26).set_color(YELLOW_VAR).move_to(DOWN * 0.30)
        return delta, star, sign

    def circuit_blocks(self, scene: dict, mode: str) -> list[tuple[Mobject, str]]:
        title = self.heading(scene["title"])
        delta, star, sign = self.circuit_pair()
        names = VGroup(serif("Delta", 25).next_to(delta, UP, buff=0.15),
                       serif("Star", 25).next_to(star, UP, buff=0.15))
        blocks: list[tuple[Mobject, str]] = [(title, "write"), (delta, "fade"), (star, "fade"), (sign, "write"), (names, "fade")]
        equations = [str(e) for e in scene.get("equations") or [] if str(e).strip()]
        if scene.get("equation") and scene["equation"] not in equations:
            equations.insert(0, scene["equation"])
        if mode == "circuit_path" and not equations:
            equations = [r"R_{AB}^{\Delta}=R_3\parallel(R_1+R_2)", r"R_{AB}^{Y}=R_a+R_b"]
        if mode == "formula_triplet" and not equations:
            equations = [r"R_a=\frac{R_2R_3}{R_1+R_2+R_3}",
                         r"R_b=\frac{R_1R_3}{R_1+R_2+R_3}",
                         r"R_c=\frac{R_1R_2}{R_1+R_2+R_3}"]
        if equations:
            VGroup(delta, star, sign, names).shift(UP * 0.42)
            rows = VGroup(*[math_tex(eq, 31) for eq in equations[:3]])
            rows.arrange(DOWN, buff=0.15).move_to(DOWN * 2.0)
            fit(rows, 11.6, 1.15)
            blocks.append((rows, "write"))
        return blocks

    def equation_blocks(self, scene: dict) -> list[tuple[Mobject, str]]:
        raw = scene.get("equations") or ([scene["equation"]] if scene.get("equation") else [])
        raw = [str(x) for x in raw if str(x).strip()][:4]
        if not raw:
            return self.label_blocks(scene)
        equations = VGroup(*[math_tex(eq, 47) for eq in raw]).arrange(DOWN, buff=0.55).move_to(ORIGIN)
        fit(equations, 11.8, 4.4)
        blocks = [(self.heading(scene["title"]), "write")]
        blocks.extend((mob, "write") for mob in equations)
        caption = scene.get("caption") or (scene.get("items") or [""])[0]
        if caption:
            if len(equations) == 1:
                equations.shift(UP * 0.45)
                brace = Brace(equations, DOWN, color=WHITE_SOFT)
                label = fit(serif(str(caption), 25, WHITE_SOFT), 11, 0.75).next_to(brace, DOWN, buff=0.1)
                blocks.extend([(brace, "create"), (label, "fade")])
            else:
                blocks.append((fit(serif(str(caption), 24, WHITE_SOFT), 11, 0.7).move_to(DOWN * 2.3), "fade"))
        return blocks

    def matrix_blocks(self, scene: dict) -> list[tuple[Mobject, str]]:
        raw = scene.get("equations") or ([scene.get("equation")] if scene.get("equation") else [])
        rows = [math_tex(str(eq), 45) for eq in raw[:3] if eq]
        if not rows:
            return self.label_blocks(scene)
        group = VGroup(*rows).arrange(DOWN, buff=0.55).move_to(ORIGIN)
        fit(group, 12, 5)
        return [(self.heading(scene["title"]), "write"), *[(row, "write") for row in group]]

    def vector_blocks(self, scene: dict) -> list[tuple[Mobject, str]]:
        title = self.heading(scene["title"], 42)
        grid = NumberPlane(x_range=[-6, 6, 1], y_range=[-3, 3, 1],
                           background_line_style={"stroke_color": "#243038", "stroke_width": 1, "stroke_opacity": 0.6})
        arrows = VGroup()
        for x in range(-5, 6):
            for y in range(-2, 3):
                if x == 0 and y == 0:
                    continue
                end = np.array([x, y, 0])
                arrows.add(Arrow(end * 0.77, end, buff=0, stroke_width=2.2,
                                 max_tip_length_to_length_ratio=0.25,
                                 color=PURPLE_VAR if x < 0 else BLUE_VAR))
        origin = Dot(ORIGIN, radius=0.09, color=YELLOW_VAR)
        return [(grid, "create"), (arrows, "fade"), (origin, "fade"), (title, "write")]

    def graph_blocks(self, scene: dict) -> list[tuple[Mobject, str]]:
        points = scene.get("points") or []
        if len(points) < 2:
            return self.label_blocks(scene)
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        xlo, xhi, ylo, yhi = min(xs), max(xs), min(ys), max(ys)
        if xlo == xhi: xhi += 1
        if ylo == yhi: yhi += 1
        axes = Axes(x_range=[xlo, xhi, (xhi-xlo)/5], y_range=[ylo, yhi, (yhi-ylo)/5],
                    x_length=9.5, y_length=4.5, tips=True, axis_config={"color": WHITE_SOFT}).move_to(DOWN * 0.2)
        curve = VMobject(color=YELLOW_VAR, stroke_width=4).set_points_as_corners([axes.c2p(x, y) for x, y in points])
        dots = VGroup(*[Dot(axes.c2p(x, y), color=BLUE_VAR, radius=0.045) for x, y in points])
        return [(self.heading(scene["title"]), "write"), (axes, "create"), (curve, "create"), (dots, "fade")]

    def generic_diagram_blocks(self, scene: dict) -> list[tuple[Mobject, str]]:
        labels = (scene.get("items") or [])[:5]
        if not labels:
            return self.label_blocks(scene)
        words = VGroup(*[fit(serif(str(label), 25), 2.1, 0.9) for label in labels])
        words.arrange(RIGHT, buff=0.85).move_to(ORIGIN)
        arrows = VGroup(*[Arrow(words[i].get_right(), words[i+1].get_left(), buff=0.08,
                                color=YELLOW_VAR, stroke_width=3) for i in range(len(words)-1)])
        return [(self.heading(scene["title"]), "write"), *[(word, "write") for word in words], (arrows, "create")]

    def label_blocks(self, scene: dict) -> list[tuple[Mobject, str]]:
        labels = scene.get("items") or [scene.get("narration", "").split(".")[0][:100]]
        labels = [str(item) for item in labels[:5] if str(item).strip()]
        rows = VGroup(*[fit(serif(label, 32, [WHITE_SOFT, GREEN_VAR, RED_VAR, BLUE_VAR, YELLOW_VAR][i]), 10.8, 0.9)
                        for i, label in enumerate(labels)])
        rows.arrange(DOWN, buff=0.55).move_to(ORIGIN)
        return [(self.heading(scene["title"]), "write"), *[(row, "write") for row in rows]]
