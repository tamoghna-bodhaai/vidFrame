from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .control import CancelledError


class VideoError(Exception):
    pass


@dataclass(slots=True)
class VideoMeta:
    duration: float
    width: int
    height: int
    fps: float
    total_frames: int


@dataclass(slots=True)
class ExtractionResult:
    meta: VideoMeta
    sample_fps: float
    sample_every: int
    scene_count: int
    frames: list[dict]
    cancelled: bool = False


def probe_video(path: Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError("Could not open the video. The format may not be supported.")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()

    if not total or not fps or fps <= 0 or fps > 1000:
        raise VideoError("Could not read valid timing metadata from this video.")
    duration = total / fps
    if duration <= 0:
        raise VideoError("Video has zero duration.")
    if width <= 0 or height <= 0:
        raise VideoError("Could not read video dimensions.")
    return VideoMeta(duration=duration, width=width, height=height, fps=fps, total_frames=total)


def dhash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _brightness(gray: np.ndarray) -> float:
    return float(gray.mean() / 255.0)


def _resize(frame: np.ndarray, max_side: int) -> np.ndarray:
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return frame
    scale = max_side / float(longest)
    return cv2.resize(
        frame,
        (max(2, int(round(width * scale))), max(2, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def plan_sampling(meta: VideoMeta, sample_fps: float, max_frames: int) -> tuple[float, int]:
    sample_fps = max(0.05, min(float(sample_fps), meta.fps))
    sample_every = max(1, int(round(meta.fps / sample_fps)))
    estimated = math.ceil(meta.total_frames / sample_every)
    if estimated > max_frames:
        sample_every = max(sample_every, math.ceil(meta.total_frames / max_frames))
        estimated = math.ceil(meta.total_frames / sample_every)
    effective_fps = meta.fps / sample_every
    return effective_fps, sample_every


def extract_frames(
    path: Path,
    out_dir: Path,
    sample_fps: float,
    max_frames: int,
    max_side: int,
    jpeg_quality: int,
    progress: Callable[[float, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    duplicate_distance: int = 2,
    scene_distance: int = 14,
) -> ExtractionResult:
    """Sample frames sequentially (accurate timestamps, no keyframe seeking)."""
    meta = probe_video(path)
    effective_fps, sample_every = plan_sampling(meta, sample_fps, max_frames)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError("Could not open the video for extraction.")

    frames: list[dict] = []
    scene_index = -1
    last_dhash: int | None = None
    last_kept_dhash: int | None = None
    last_kept_index: int | None = None
    last_report = 0.0
    frame_index = 0
    sample_index = 0
    scene_count = 0

    try:
        while True:
            if should_cancel and should_cancel():
                return ExtractionResult(meta, effective_fps, sample_every, scene_count, frames, True)
            grabbed = cap.grab()
            if not grabbed:
                break
            if frame_index % sample_every != 0:
                frame_index += 1
                continue

            ok, frame = cap.retrieve()
            if not ok or frame is None:
                frame_index += 1
                continue

            timestamp = frame_index / meta.fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fingerprint = dhash(gray)

            duplicate = False
            duplicate_of = None
            if last_kept_dhash is not None and hamming(fingerprint, last_kept_dhash) <= duplicate_distance:
                duplicate = True
                duplicate_of = last_kept_index
            else:
                if last_dhash is None or hamming(fingerprint, last_dhash) > scene_distance:
                    scene_index += 1
                    scene_count = scene_index + 1
                last_kept_dhash = fingerprint
                last_kept_index = sample_index
            last_dhash = fingerprint

            resized = _resize(frame, max_side)
            filename = f"f{sample_index:05d}.jpg"
            target = out_dir / filename
            cv2.imwrite(
                str(target),
                resized,
                [int(cv2.IMWRITE_JPEG_QUALITY), max(40, min(95, jpeg_quality))],
            )
            height, width = resized.shape[:2]
            frames.append(
                {
                    "index": sample_index,
                    "time": round(timestamp, 3),
                    "file": filename,
                    "width": width,
                    "height": height,
                    "dhash": f"{fingerprint:016x}",
                    "duplicate": duplicate,
                    "duplicate_of": duplicate_of,
                    "scene": max(0, scene_index),
                    "brightness": round(_brightness(gray), 3),
                    "bytes": target.stat().st_size if target.exists() else 0,
                }
            )

            sample_index += 1
            frame_index += 1

            if progress and (time.monotonic() - last_report > 0.7 or sample_index % 50 == 0):
                last_report = time.monotonic()
                progress(min(1.0, frame_index / max(1, meta.total_frames)), sample_index)

        if progress:
            progress(1.0, sample_index)
    finally:
        cap.release()

    return ExtractionResult(meta, effective_fps, sample_every, scene_count, frames, False)


def analyze_image_stats(path: Path) -> dict:
    """Cheap deterministic image facts used by the mock provider and UI."""
    image = cv2.imread(str(path))
    if image is None:
        return {"brightness": 0.0, "detail": 0.0, "dominant": "unknown"}
    small = cv2.resize(image, (64, 36), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue = float(np.median(hsv[:, :, 0]))
    sat = float(hsv[:, :, 1].mean())
    val = float(hsv[:, :, 2].mean())
    edges = cv2.Canny(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 60, 160)
    detail = float((edges > 0).mean())
    if sat < 30:
        dominant = "muted grey" if val < 140 else "bright neutral"
    elif hue < 12 or hue > 165:
        dominant = "red"
    elif hue < 35:
        dominant = "orange/amber"
    elif hue < 75:
        dominant = "green"
    elif hue < 130:
        dominant = "blue"
    else:
        dominant = "violet"
    return {
        "brightness": round(val / 255.0, 3),
        "detail": round(detail, 4),
        "dominant": dominant,
        "saturation": round(sat / 255.0, 3),
    }
