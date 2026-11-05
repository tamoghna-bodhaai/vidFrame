#!/usr/bin/env python3
"""Generate a synthetic demo video with distinct visual scenes and legible text."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT, FPS = 960, 540, 24
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = FONT_DIR / name
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def mono(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_DIR / ("DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def rounded(draw: ImageDraw.ImageDraw, box, radius: int, fill, outline=None, width: int = 1) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text(draw, xy, value, fnt, fill, anchor=None) -> None:
    draw.text(xy, value, font=fnt, fill=fill, anchor=anchor)


def slide_title(seconds: float) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#0b0c0e")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, WIDTH, 6], fill="#ffb64d")
    draw.ellipse([690, -160, 1120, 270], fill="#17130c")
    draw.ellipse([720, -130, 1090, 240], outline="#342813", width=2)
    text(draw, (72, 108), "VIDFRAME / DEMO REEL", font(15, True), "#ffb64d")
    text(draw, (70, 152), "SEE WHAT THE", font(64, True), "#f1f2f4")
    text(draw, (70, 226), "VIDEO", font(64, True), "#ffb64d")
    text(draw, (70, 300), "ACTUALLY SHOWS.", font(64, True), "#f1f2f4")
    text(draw, (73, 402), "Frame-by-frame visual analysis, compiled into a timestamped report.", font(18), "#9098a2")
    for index, label in enumerate(["EXTRACT", "READ", "COMPILE"]):
        x = 73 + index * 150
        rounded(draw, [x, 458, x + 126, 490], 7, "#15181d", "#2b3037")
        text(draw, (x + 63, 474), label, mono(10, True), "#aab1ba", anchor="mm")
    text(draw, (886, 504), f"{seconds:04.1f}s", mono(12), "#5d646e", anchor="rs")
    return image


def slide_chart(seconds: float) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f2efe8")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, WIDTH, 74], fill="#14171b")
    text(draw, (36, 24), "NORTHWIND / Q1 REVIEW", font(18, True), "#f4f0e7")
    text(draw, (900, 28), "LIVE DASHBOARD", mono(11), "#ffb64d", anchor="ra")

    rounded(draw, [30, 100, 640, 500], 16, "#fffdf8", "#ded8ca")
    text(draw, (58, 126), "Quarterly revenue", font(25, True), "#181a1d")
    text(draw, (58, 162), "USD thousands, unaudited", font(13), "#857f73")

    values = [("JAN", 42), ("FEB", 61), ("MAR", 55), ("APR", 78), ("MAY", 96)]
    base_y = 432
    for index, (label, value) in enumerate(values):
        x = 78 + index * 108
        height = int(value * 2.35)
        color = "#ffb64d" if index == len(values) - 1 else "#22272e"
        rounded(draw, [x, base_y - height, x + 62, base_y], 7, color)
        text(draw, (x + 31, base_y - height - 20), str(value), mono(12, True), "#3c3f44" if index < 4 else "#b77714", anchor="mm")
        text(draw, (x + 31, base_y + 18), label, mono(10, True), "#8e887c", anchor="mm")
    draw.line([68, base_y + 2, 616, base_y + 2], fill="#dad4c7", width=1)
    text(draw, (58, 468), "May is up 53% year over year", font(12), "#6c675d")

    rounded(draw, [664, 100, 930, 500], 16, "#171a1e", "#2b3038")
    text(draw, (688, 126), "KEY METRICS", mono(10, True), "#ffb64d")
    metrics = [("REVENUE", "$96.4k", "+53%"), ("ACTIVE USERS", "12,840", "+18%"), ("CHURN", "1.8%", "-0.4%")]
    for index, (label, value, delta) in enumerate(metrics):
        y = 166 + index * 92
        text(draw, (688, y), label, mono(9, True), "#78808a")
        text(draw, (688, y + 18), value, font(27, True), "#f2f3f5")
        text(draw, (894, y + 24), delta, mono(11, True), "#6fe3c2", anchor="ra")
        draw.line([688, y + 74, 906, y + 74], fill="#272c33", width=1)
    text(draw, (688, 452), "Source: finance_snapshot.csv", mono(8), "#5e6670")

    draw.rectangle([0, 520, WIDTH, HEIGHT], fill="#e4dfd3")
    text(draw, (30, 529), "CONFIDENTIAL — INTERNAL REVIEW", mono(8, True), "#817b6e")
    text(draw, (930, 529), f"slide 02 · {seconds:04.1f}s", mono(8), "#817b6e", anchor="ra")
    return image


def slide_ui(seconds: float) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#0d0f12")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 214, HEIGHT], fill="#12151a")
    draw.line([214, 0, 214, HEIGHT], fill="#242931", width=1)
    draw.rectangle([0, 0, 5, HEIGHT], fill="#ffb64d")
    text(draw, (26, 30), "vidFrame", font(20, True), "#f2f4f6")
    text(draw, (26, 60), "visual analysis", mono(8), "#69717c")
    for index, label in enumerate(["Overview", "Analyses", "Frames", "Settings"]):
        y = 106 + index * 38
        active = index == 1
        rounded(draw, [16, y, 198, y + 30], 7, "#221b0f" if active else "#12151a")
        text(draw, (32, y + 15), label, font(11, active), "#ffcb78" if active else "#7d858f", anchor="lm")
    text(draw, (26, 492), "2.4 GB stored", mono(8), "#5a626c")

    text(draw, (248, 36), "New analysis", font(23, True), "#f1f3f5")
    text(draw, (248, 68), "Upload a video, sample every second, read every frame.", font(11), "#7f8792")
    rounded(draw, [248, 104, 704, 386], 14, "#12151a", "#2c323a")
    draw.rounded_rectangle([282, 138, 670, 352], radius=10, outline="#3a414b", width=1)
    text(draw, (476, 172), "product-demo.mp4", font(15, True), "#e8ebee", anchor="ma")
    text(draw, (476, 196), "00:18 · 960×540 · 24 fps", mono(9), "#6b737e", anchor="ma")
    rounded(draw, [318, 234, 634, 242], 4, "#242a31")
    rounded(draw, [318, 234, 512, 242], 4, "#ffb64d")
    text(draw, (318, 258), "Analyzing frames · 68%", mono(9), "#a3aab3")
    text(draw, (634, 258), "12 / 18", mono(9), "#6b737e", anchor="ra")
    rounded(draw, [392, 294, 560, 330], 7, "#ffb64d")
    text(draw, (476, 312), "Open live analysis", font(10, True), "#1a1206", anchor="mm")

    text(draw, (730, 104), "RECENT", mono(9, True), "#5f6771")
    items = [("Meetup keynote", "Ready · 4:12", "#6fe3c2"), ("Dashboard tour", "Analyzing · 68%", "#ffb64d"), ("Code walkthrough", "Ready · 2:04", "#6fe3c2")]
    for index, (name, meta, color) in enumerate(items):
        y = 132 + index * 78
        rounded(draw, [730, y, 928, y + 64], 9, "#12151a", "#252b32")
        draw.ellipse([742, y + 14, 750, y + 22], fill=color)
        text(draw, (760, y + 13), name, font(10, True), "#dfe3e7")
        text(draw, (760, y + 33), meta, mono(8), "#69717b")
        rounded(draw, [742, y + 46, 916, y + 51], 2, "#232930")
        rounded(draw, [742, y + 46, 850 if index != 1 else 800, y + 51], 2, color)

    rounded(draw, [730, 396, 928, 500], 12, "#12151a", "#252b32")
    text(draw, (746, 412), "VISUAL TIMELINE", mono(8, True), "#5f6771")
    for index, (stamp, label) in enumerate([("0:04", "Chart"), ("0:08", "Upload UI")]):
        y = 434 + index * 26
        text(draw, (746, y), stamp, mono(8), "#ffb64d")
        text(draw, (782, y), label, font(9), "#9ba2ab")

    text(draw, (930, 524), f"frame {seconds:04.1f}s", mono(8), "#4f5660", anchor="rs")
    return image


def slide_code(seconds: float) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#0c0f14")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, WIDTH, 44], fill="#12161c")
    draw.rectangle([0, 0, 150, 44], fill="#171c23")
    text(draw, (18, 15), "analyze.py", mono(10, True), "#d8dde3")
    text(draw, (168, 15), "pipeline.py", mono(10), "#5f6874")
    text(draw, (168 + 120, 15), "report.py", mono(10), "#5f6874")
    draw.line([0, 44, WIDTH, 44], fill="#242a32", width=1)

    lines = [
        ("1", "import cv2"),
        ("2", "from vlm import VisionModel"),
        ("3", ""),
        ("4", "def analyze(video_path: str):"),
        ("5", "    frames = extract_frames(video_path, fps=1)"),
        ("6", "    unique = drop_near_duplicates(frames)"),
        ("7", ""),
        ("8", "    notes = []"),
        ("9", "    for batch in chunk(unique, size=4):"),
        ("10", "        notes += vlm.describe(batch)"),
        ("11", ""),
        ("12", "    return compile_report(notes, timestamps=True)"),
    ]
    colors = {
        "import": "#c792ea", "from": "#c792ea", "def": "#82aaff", "return": "#c792ea",
        "for": "#c792ea", "in": "#82aaff", "str": "#ffcb6b", "fps": "#f78c6c",
        "size": "#f78c6c", "True": "#f78c6c", "timestamps": "#f78c6c",
    }
    y = 78
    for number, code in lines:
        text(draw, (24, y), number, mono(10), "#3f4855")
        x = 64
        for token in code.split(" "):
            if not token:
                x += 7
                continue
            color = "#c5cbd3"
            stripped = token.strip("()[]:,")
            if stripped in colors:
                color = colors[stripped]
            elif stripped.startswith('"') or stripped.startswith("'"):
                color = "#c3e88d"
            elif stripped.endswith("("):
                color = "#82aaff"
            elif stripped.startswith("#"):
                color = "#5b6470"
            draw.text((x, y), token, font=mono(12), fill=color)
            x += int(draw.textlength(token + " ", font=mono(12)))
        y += 27

    draw.rectangle([0, 424, WIDTH, HEIGHT], fill="#0a0d11")
    draw.line([0, 424, WIDTH, 424], fill="#242a32", width=1)
    text(draw, (24, 440), "TERMINAL", mono(8, True), "#5f6874")
    text(draw, (24, 462), "$ python -m app.cli analyze product-demo.mp4 --fps 1", mono(10), "#8b939e")
    text(draw, (24, 484), "✓ 18 frames · 12 unique · report.md written in 41s", mono(10), "#6fe3c2")
    draw.rectangle([24, 512, 32, 524], fill="#ffb64d")
    text(draw, (926, 524), f"{seconds:04.1f}s", mono(8), "#3f4855", anchor="rs")
    return image


def slide_summary(seconds: float) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#0b0c0e")
    draw = ImageDraw.Draw(image)
    draw.ellipse([-220, 280, 420, 900], fill="#14110b")
    draw.ellipse([600, -240, 1180, 340], fill="#101317")
    text(draw, (72, 92), "COMPILED VISUAL REPORT", mono(11, True), "#ffb64d")
    text(draw, (70, 132), "A transcript says", font(38, True), "#6f7680")
    text(draw, (70, 182), "what was said.", font(38, True), "#f0f1f3")
    text(draw, (70, 242), "This says what was shown.", font(38, True), "#ffb64d")

    chips = [("0:00", "Title"), ("0:04", "Revenue chart"), ("0:08", "Upload UI"), ("0:12", "Code"), ("0:16", "Summary")]
    x = 70
    for stamp, label in chips:
        width = 76 + len(label) * 7
        rounded(draw, [x, 330, x + width, 376], 8, "#14171b", "#2a3038")
        text(draw, (x + 12, 342), stamp, mono(8), "#ffb64d")
        text(draw, (x + 12, 357), label, font(9), "#b9bfc7")
        x += width + 9

    text(draw, (72, 420), "18 frames · 5 scenes · 41 on-screen text blocks · 3m 12s to read", mono(10), "#78808a")
    rounded(draw, [72, 452, 286, 490], 8, "#ffb64d")
    text(draw, (179, 471), "vidFrame / done", font(11, True), "#1a1206", anchor="mm")
    text(draw, (928, 520), f"{seconds:04.1f}s", mono(8), "#434a54", anchor="rs")
    return image


def make_video(path: Path, duration: float = 20.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter (mp4v)")
    total = int(duration * FPS)
    scenes = [slide_title, slide_chart, slide_ui, slide_code, slide_summary]
    for index in range(total):
        seconds = index / FPS
        scene_index = min(len(scenes) - 1, int(seconds / (duration / len(scenes))))
        frame = scenes[scene_index](seconds)
        writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    writer.release()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a synthetic demo video for vidFrame")
    parser.add_argument("--out", default="demo.mp4")
    parser.add_argument("--duration", type=float, default=20.0)
    args = parser.parse_args()
    path = make_video(Path(args.out), args.duration)
    print(f"wrote {path} ({path.stat().st_size / 1_048_576:.1f} MB)")


if __name__ == "__main__":
    main()
