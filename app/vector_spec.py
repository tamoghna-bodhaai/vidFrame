"""Validated SVG primitives for source-grounded STEM diagrams.

The vision model supplies geometry, not SVG. Only known shapes and finite normalized
coordinates are accepted; no arbitrary paths, CSS, URLs, or raw markup pass through.
"""
from __future__ import annotations

import math
from html import escape
from pathlib import Path

_SHAPES = {"line", "arrow", "circle", "ellipse", "rect", "polyline", "polygon", "arc", "label", "lens", "resistor", "capacitor", "battery"}


def _coord(value: object) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError("coordinate outside normalized source frame")
    return number


def render_spec(diagram: dict, path: Path) -> bool:
    spec = diagram.get("render_spec")
    if not isinstance(spec, dict):
        return False
    try:
        confidence = float(spec.get("confidence") or 0)
        if not math.isfinite(confidence) or confidence < .8 or confidence > 1:
            return False
        elements = spec.get("elements")
        if not isinstance(elements, list) or not 2 <= len(elements) <= 60:
            return False
        body = []
        for element in elements:
            if not isinstance(element, dict) or element.get("shape") not in _SHAPES:
                return False
            shape = element["shape"]
            x = 45 + 710 * _coord(element.get("x"))
            y = 45 + 450 * _coord(element.get("y"))
            x2 = 45 + 710 * _coord(element.get("x2", element.get("x")))
            y2 = 45 + 450 * _coord(element.get("y2", element.get("y")))
            label = str(element.get("text") or "").strip()
            if len(label) > 90 or any(ord(ch) < 32 for ch in label):
                return False
            safe = escape(label)
            stroke = '#244a68'
            if shape == 'label':
                if not label: return False
                body.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" font-size="18" fill="#17334b">{safe}</text>')
            elif shape in {'line', 'arrow'}:
                body.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="3"'+(' marker-end="url(#arrow)"' if shape == 'arrow' else '')+'/>')
            elif shape in {'circle', 'ellipse'}:
                rx = abs(x2-x)/2; ry = abs(y2-y)/2
                if rx < 3 or (shape == 'ellipse' and ry < 3): return False
                cx = (x+x2)/2; cy = (y+y2)/2
                if shape == 'circle':
                    body.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{min(rx, max(ry, rx)):.1f}" fill="none" stroke="{stroke}" stroke-width="3"/>')
                else:
                    body.append(f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" fill="none" stroke="{stroke}" stroke-width="3"/>')
            elif shape == 'rect':
                body.append(f'<rect x="{min(x,x2):.1f}" y="{min(y,y2):.1f}" width="{abs(x2-x):.1f}" height="{abs(y2-y):.1f}" rx="6" fill="none" stroke="{stroke}" stroke-width="3"/>')
            elif shape in {'polyline', 'polygon', 'arc'}:
                points = element.get('points')
                if not isinstance(points, list) or not 2 <= len(points) <= 100: return False
                pairs = []
                for point in points:
                    if not isinstance(point, list) or len(point) != 2: return False
                    pairs.append(f'{45+710*_coord(point[0]):.1f},{45+450*_coord(point[1]):.1f}')
                if shape == 'polygon':
                    body.append(f'<polygon points="{" ".join(pairs)}" fill="none" stroke="{stroke}" stroke-width="3"/>')
                else:
                    body.append(f'<polyline points="{" ".join(pairs)}" fill="none" stroke="{stroke}" stroke-width="3"/>')
            elif shape == 'lens':
                cx=(x+x2)/2; top=min(y,y2); bottom=max(y,y2)
                if bottom-top < 20:return False
                body.append(f'<path d="M{cx:.1f} {top:.1f} Q{cx+25:.1f} {(top+bottom)/2:.1f} {cx:.1f} {bottom:.1f} Q{cx-25:.1f} {(top+bottom)/2:.1f} {cx:.1f} {top:.1f}" fill="#e8f6fa" stroke="{stroke}" stroke-width="3"/>')
            elif shape in {'resistor','capacitor','battery'}:
                if abs(x2-x) < 30 or abs(y2-y) > 25:return False
                mid=(x+x2)/2
                if shape=='resistor':
                    body.append(f'<rect x="{mid-25:.1f}" y="{y-12:.1f}" width="50" height="24" fill="white" stroke="{stroke}" stroke-width="3"/>')
                else:
                    gap=12 if shape=='capacitor' else 17
                    body.append(f'<line x1="{mid-gap:.1f}" y1="{y-18:.1f}" x2="{mid-gap:.1f}" y2="{y+18:.1f}" stroke="{stroke}" stroke-width="3"/>')
                    body.append(f'<line x1="{mid+gap:.1f}" y1="{y-(18 if shape=="capacitor" else 27):.1f}" x2="{mid+gap:.1f}" y2="{y+(18 if shape=="capacitor" else 27):.1f}" stroke="{stroke}" stroke-width="3"/>')
                body.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{mid-(25 if shape=="resistor" else gap):.1f}" y2="{y:.1f}" stroke="{stroke}" stroke-width="3"/>')
                body.append(f'<line x1="{mid+(25 if shape=="resistor" else gap):.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" stroke="{stroke}" stroke-width="3"/>')
            if label and shape != 'label':
                body.append(f'<text x="{(x+x2)/2:.1f}" y="{min(y,y2)-10:.1f}" text-anchor="middle" font-size="16" fill="#17334b">{safe}</text>')
        svg = '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="540" viewBox="0 0 800 540"><rect width="800" height="540" fill="white"/><defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0 L9 4.5 L0 9" fill="none" stroke="#244a68"/></marker></defs>'+''.join(body)+'</svg>'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(svg, encoding='utf-8')
        return True
    except (TypeError, ValueError, OverflowError):
        return False
