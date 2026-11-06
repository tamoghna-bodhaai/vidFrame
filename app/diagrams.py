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
    if raw.get("image"):
        diagram["image"] = str(raw["image"])
    if raw.get("analysis_model"):
        diagram["analysis_model"] = str(raw["analysis_model"])
    if raw.get("detail_analyzed"):
        diagram["detail_analyzed"] = True
    if raw.get("diagram_schema_version") == 1:
        diagram["diagram_schema_version"] = 1
    if raw.get("error"):
        diagram["error"] = str(raw["error"])[:600]
    return diagram


def crop_diagram(frame_path: Path, bbox: list[float] | None, out_path: Path, padding: float = 0.02, *, capture=None, seconds: float = 0) -> bool:
    """Crop the detected diagram region (or the whole frame) into a standalone JPEG."""
    image = None
    if capture is not None and capture.isOpened():
        capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, image = capture.read()
        if not ok:
            image = None
    if image is None:
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
    for frame in frames:
        observation = observations.get(frame.get("index"))
        if not observation:
            continue
        diagram = coerce_diagram(observation.get("diagram"), present=bool(observation.get("has_diagram")))
        if not diagram:
            continue
        entry = dict(diagram)
        entry.update(time=fmt_ts(frame.get("time")), time_seconds=frame.get("time"),
                     frame_index=frame.get("index"), scene_index=frame.get("scene", 0),
                     id=f"diagram-f{int(frame['index']):05d}")
        diagrams.append(entry)
    # Text similarity is not evidence of identical geometry or labels. Pixel-based
    # deduplication happens only after source assets have been recovered.
    return sorted(diagrams, key=lambda item: float(item.get("time_seconds") or 0))


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
    merged.pop("semantic_spec", None)
    merged.pop("render_spec", None)
    merged.pop("generated_image", None)
    merged["present"] = True
    return merged


def prepare_source_assets(job_dir: Path, frames: list[dict], diagrams: list[dict]) -> tuple[list[dict], list[dict]]:
    """Recover native-resolution crops; retain every distinct observed pixel state.

    Exact pixel deduplication is deliberately conservative: tiny new labels matter.
    Missing videos fall back to existing crops, then stored frames. No model calls.
    """
    import hashlib

    by_index = {frame['index']: frame for frame in frames}
    sources = sorted(job_dir.glob('source.*'))
    capture = cv2.VideoCapture(str(sources[0])) if sources else None
    kept, audit, seen = [], [], {}
    directory = job_dir / 'diagrams'
    directory.mkdir(exist_ok=True)
    (job_dir / "diagrams.zip").unlink(missing_ok=True)
    try:
        for source in diagrams:
            diagram = dict(source)
            index = int(diagram['frame_index'])
            diagram_id = diagram.get('id') or f'diagram-f{index:05d}'
            target = directory / f'f{index:05d}.png'
            image = None
            origin = 'original_video'
            if capture is not None and capture.isOpened():
                capture.set(cv2.CAP_PROP_POS_MSEC, float(diagram.get('time_seconds') or 0) * 1000)
                ok, image = capture.read()
                if not ok:
                    image = None
            if image is None:
                origin = 'stored_crop'
                existing = (job_dir / str(diagram.get('image') or '')).resolve()
                if job_dir.resolve() in existing.parents and existing.is_file():
                    image = cv2.imread(str(existing))
                if image is None:
                    origin = 'sampled_frame'
                    frame = by_index.get(index, {})
                    image = cv2.imread(str(job_dir / 'frames' / frame.get('file', 'missing')))
            if image is None:
                audit.append({'id': diagram_id, 'status': 'missing', 'reason': 'No readable video, crop, or frame'})
                continue
            bbox = coerce_bbox(diagram.get('bbox'))
            if origin != 'stored_crop' and bbox:
                x, y, w, h = bbox
                height, width = image.shape[:2]
                image = image[max(0, int((y-.02)*height)):min(height, int((y+h+.02)*height)),
                              max(0, int((x-.02)*width)):min(width, int((x+w+.02)*width))]
            digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
            # Preserve reappearances in different topics/scenes for contextual placement.
            key = (diagram.get('scene_index', 0), digest)
            if key in seen:
                audit.append({'id': diagram_id, 'status': 'duplicate', 'duplicate_of': seen[key]})
                continue
            if not cv2.imwrite(str(target), image):
                audit.append({'id': diagram_id, 'status': 'missing', 'reason': 'Crop write failed'})
                continue
            seen[key] = diagram_id
            diagram.pop('generated_image', None)
            diagram.pop('semantic_spec', None)
            diagram.update(id=diagram_id, display_image=f'diagrams/{target.name}',
                           image=f'diagrams/{target.name}', asset_kind='source_crop',
                           source_resolution=origin, crop_scope='region' if bbox else 'full_frame')
            kept.append(diagram)
            audit.append({'id': diagram_id, 'status': 'included', 'image': diagram['image'],
                          'source_resolution': origin, 'crop_scope': diagram['crop_scope']})
    finally:
        if capture is not None:
            capture.release()
    return kept, audit
