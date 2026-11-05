from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from .utils import as_float, as_list, as_text, fmt_ts


def coerce_bbox(value: Any) -> list[float] | None:
    """Normalize a bounding box to [x, y, width, height] in 0..1 coordinates."""
    box: list[float] | None = None
    if isinstance(value, dict):
        try:
            box = [float(value["x"]), float(value["y"]), float(value["width"]), float(value["height"])]
        except (KeyError, TypeError, ValueError):
            return None
    elif isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            box = [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
        except (TypeError, ValueError):
            return None
    if box is None:
        return None
    if max(box) > 1.5:
        box = [part / 100.0 for part in box]
    x, y, width, height = (max(0.0, min(1.0, part)) for part in box)
    if width < 0.02 or height < 0.02 or x >= 1.0 or y >= 1.0:
        return None
    width = min(width, 1.0 - x)
    height = min(height, 1.0 - y)
    if width < 0.02 or height < 0.02:
        return None
    return [round(x, 4), round(y, 4), round(width, 4), round(height, 4)]


def coerce_diagram(raw: Any, *, present: bool | None = None) -> dict[str, Any] | None:
    """Normalize diagram observations and preserve extraction provenance."""
    if not isinstance(raw, dict):
        return None
    is_present = raw.get("present") if present is None else present
    if is_present is False:
        return None
    confidence = as_float(raw.get("confidence"), 0.65)
    if confidence > 1:
        confidence = confidence / 100.0
    diagram = {
        "present": True,
        "type": as_text(raw.get("type") or raw.get("kind"), 80) or "diagram",
        "title": as_text(raw.get("title") or raw.get("name"), 160),
        "description": as_text(raw.get("description") or raw.get("summary"), 2000),
        "nodes": as_list(raw.get("nodes") or raw.get("elements"), 40),
        "edges": as_list(raw.get("edges") or raw.get("connections") or raw.get("flows"), 40),
        "labels": as_list(raw.get("labels") or raw.get("text"), 60),
        "values": as_list(raw.get("values") or raw.get("data"), 60),
        "bbox": coerce_bbox(raw.get("bbox") or raw.get("box")),
        "confidence": max(0.0, min(1.0, confidence)),
    }
    if isinstance(raw.get("render_spec"), dict):
        diagram["render_spec"] = raw["render_spec"]
    if raw.get("image"):
        diagram["image"] = str(raw["image"])
    if raw.get("analysis_model"):
        diagram["analysis_model"] = str(raw["analysis_model"])
    if raw.get("detail_analyzed"):
        diagram["detail_analyzed"] = True
    if raw.get("error"):
        diagram["error"] = str(raw["error"])[:600]
    return diagram


def crop_diagram(frame_path: Path, bbox: list[float] | None, out_path: Path, padding: float = 0.02) -> bool:
    """Crop the detected diagram region (or the whole frame) into a standalone JPEG."""
    image = cv2.imread(str(frame_path))
    if image is None:
        return False
    height, width = image.shape[:2]
    if bbox:
        x, y, box_w, box_h = bbox
        left = max(0, int((x - padding) * width))
        top = max(0, int((y - padding) * height))
        right = min(width, int((x + box_w + padding) * width))
        bottom = min(height, int((y + box_h + padding) * height))
        if right - left >= 8 and bottom - top >= 8:
            image = image[top:bottom, left:right]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90]))


def collect_diagrams(frames: list[dict[str, Any]], observations: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Return unique diagrams in chronological order, using observations as source of truth."""
    diagrams: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for frame in frames:
        observation = observations.get(frame.get("index"))
        if not observation:
            continue
        diagram = coerce_diagram(
            observation.get("diagram"),
            present=bool(observation.get("has_diagram")),
        )
        if not diagram or diagram.get("type", "").lower() not in {"chart", "graph", "flowchart", "architecture", "map", "schematic"}:
            continue
        entry = dict(diagram)
        entry["time"] = fmt_ts(frame.get("time"))
        entry["time_seconds"] = frame.get("time")
        entry["frame_index"] = frame.get("index")
        entry["scene_index"] = frame.get("scene", 0)
        signature = (
            entry.get("type", "").lower(),
            entry.get("title", "").lower(),
            entry.get("description", "")[:120].lower(),
            "|".join(entry.get("values", []))[:200],
        )
        if signature in seen:
            continue
        seen.add(signature)
        diagrams.append(entry)
    return diagrams


def diagram_digest(diagrams: list[dict[str, Any]]) -> str:
    """Compact diagram facts included in the report compilation prompt."""
    lines: list[str] = []
    for diagram in diagrams:
        parts = [
            f"[DIAGRAM {diagram.get('time', '')}]",
            f"type={diagram.get('type', 'diagram')}",
            f"title={diagram.get('title') or 'untitled'}",
        ]
        if diagram.get("description"):
            parts.append(f"description={diagram['description']}")
        if diagram.get("nodes"):
            parts.append("elements=" + ", ".join(diagram["nodes"]))
        if diagram.get("edges"):
            parts.append("connections=" + ", ".join(diagram["edges"]))
        if diagram.get("labels"):
            parts.append("labels=" + ", ".join(diagram["labels"]))
        if diagram.get("values"):
            parts.append("values=" + ", ".join(diagram["values"]))
        if diagram.get("image"):
            parts.append(f"image={diagram['image']}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def merge_diagram_details(base: dict[str, Any] | None, detail: dict[str, Any] | None) -> dict[str, Any]:
    """Keep crop/location facts from the first pass and richer content from the second pass."""
    merged = dict(base or {})
    if detail:
        for key, value in detail.items():
            if value in (None, "", [], {}):
                continue
            merged[key] = value
    merged["present"] = True
    return merged
