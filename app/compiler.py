from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable

from .diagrams import coerce_diagram, collect_diagrams, diagram_digest
from .control import JobControlError
from .prompts import COMPILE_SYSTEM, compile_prompt
from .utils import (
    as_list,
    as_text,
    extract_json,
    extract_json_meta,
    fmt_duration,
    fmt_ts,
    now_iso,
    parse_ts,
)
from .vlm import VLMClient

MAX_TIMELINE = 24


def _norm_time(value: Any, default: str = "0:00") -> str:
    if isinstance(value, (int, float)):
        return fmt_ts(value)
    text = str(value or "").strip()
    if not text:
        return default
    match = re.match(r"^(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?$", text)
    if match:
        parts = [int(part) for part in match.groups() if part is not None]
        seconds = parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0] * 3600 + parts[1] * 60 + parts[2]
        return fmt_ts(seconds)
    match = re.match(r"^(\d+(?:\.\d+)?)\s*(s|sec|secs|m|min|h)$", text, re.IGNORECASE)
    if match:
        amount = float(match.group(1))
        unit = match.group(2).lower()
        factor = {"s": 1, "sec": 1, "secs": 1, "m": 60, "min": 60, "h": 3600}[unit]
        return fmt_ts(amount * factor)
    return text


def _screen_text_from(obs: dict[str, Any]) -> list[str]:
    return as_list(obs.get("screen_text"), 60)


def build_digest(frames: list[dict], observations: dict[int, dict]) -> str:
    lines: list[str] = []
    previous_scene: int | None = None
    last_texts: set[str] = set()
    for frame in frames:
        index = frame["index"]
        obs = observations.get(index, {})
        timestamp = fmt_ts(frame["time"])
        scene = frame.get("scene", 0)
        if scene != previous_scene:
            lines.append(f"\n--- scene {scene + 1} (from {timestamp}) ---")
            previous_scene = scene
            last_texts = set()

        if frame.get("duplicate") and obs.get("status") == "duplicate":
            lines.append(f"[{timestamp}] (visual unchanged since {fmt_ts(frame.get('duplicate_time'))})")
            continue
        if obs.get("status") == "error" and not obs.get("description"):
            lines.append(f"[{timestamp}] (frame unavailable: {obs.get('error', 'analysis failed')})")
            continue

        summary = as_text(obs.get("summary") or obs.get("scene") or obs.get("description"), 600)
        description = as_text(obs.get("description"), 2400)
        if description and description != summary:
            text = f"[{timestamp}] {summary} :: {description}"
        else:
            text = f"[{timestamp}] {summary}"

        details: list[str] = []
        setting = as_text(obs.get("setting"), 200)
        if setting:
            details.append(f"setting={setting}")
        people = as_list(obs.get("people"), 12)
        if people:
            details.append("people=" + ", ".join(people))
        objects = as_list(obs.get("objects"), 15)
        if objects:
            details.append("objects=" + ", ".join(objects))
        actions = as_list(obs.get("actions"), 12)
        if actions:
            details.append("actions=" + ", ".join(actions))
        visual_notes = as_text(obs.get("visual_notes"), 600)
        if visual_notes:
            details.append(f"notes={visual_notes}")

        texts = [text for text in _screen_text_from(obs) if text not in last_texts]
        if texts:
            details.append("on-screen text=" + " | ".join(texts))
            last_texts.update(texts)
        diagram = coerce_diagram(obs.get("diagram"), present=bool(obs.get("has_diagram")))
        if diagram:
            details.append(diagram_digest([{**diagram, "time": timestamp}]))
        if details:
            text += " | " + " | ".join(details)
        lines.append(text)

    return "\n".join(lines).strip()


def _group_scenes(frames: list[dict], observations: dict[int, dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    current: list[dict] = []
    current_scene: int | None = None
    for frame in frames:
        if current and frame.get("scene") != current_scene:
            groups.append(current)
            current = []
        current.append(frame)
        current_scene = frame.get("scene")
    if current:
        groups.append(current)
    return groups


def _merge_groups(groups: list[list[dict]], target: int) -> list[list[dict]]:
    if len(groups) <= target:
        return groups
    bucket_size = max(1, len(groups) / target)
    merged: list[list[dict]] = []
    for position in range(target):
        start = int(round(position * bucket_size))
        end = int(round((position + 1) * bucket_size))
        bucket = [frame for group in groups[start:end] for frame in group]
        if bucket:
            merged.append(bucket)
    return merged


def _representative(group: list[dict], observations: dict[int, dict]) -> dict[str, Any] | None:
    for frame in group:
        obs = observations.get(frame["index"])
        if obs and obs.get("status") in {"ok", "partial"} and (obs.get("description") or obs.get("scene")):
            return obs
    for frame in group:
        obs = observations.get(frame["index"])
        if obs and obs.get("description"):
            return obs
    return None


def _group_texts(group: list[dict], observations: dict[int, dict]) -> list[str]:
    texts: list[str] = []
    for frame in group:
        for text in _screen_text_from(observations.get(frame["index"], {})):
            if text not in texts:
                texts.append(text)
    return texts[:40]


def _fallback_note(
    frames: list[dict],
    observations: dict[int, dict],
    meta: dict[str, Any],
    reason: str = "",
) -> dict[str, Any]:
    groups = _merge_groups(_group_scenes(frames, observations), MAX_TIMELINE)
    sample_fps = float(meta.get("sample_fps") or 1.0) or 1.0
    sample_interval = meta.get("sample_interval") or 1.0 / sample_fps
    diagrams = collect_diagrams(frames, observations)
    timeline: list[dict[str, Any]] = []
    key_moments: list[dict[str, Any]] = []
    on_screen_text: list[dict[str, Any]] = []
    people: Counter[str] = Counter()
    objects: Counter[str] = Counter()
    descriptions: list[str] = []

    for group in groups:
        first, last = group[0], group[-1]
        rep = _representative(group, observations) or {}
        start = fmt_ts(first["time"])
        end = fmt_ts(last["time"] + sample_interval)
        title = as_text(rep.get("scene"), 90) or f"Visual segment at {start}"
        summary = as_text(rep.get("description") or rep.get("summary"), 1400) or "No visual description available."
        visuals = []
        for value in as_list(rep.get("visual_notes"), 6) + as_list(rep.get("objects"), 8):
            if value not in visuals:
                visuals.append(value)
        texts = _group_texts(group, observations)
        group_people: list[str] = []
        for frame in group:
            obs = observations.get(frame["index"], {})
            for person in as_list(obs.get("people"), 8):
                people[person] += 1
                if person not in group_people:
                    group_people.append(person)
            for obj in as_list(obs.get("objects"), 10):
                objects[obj] += 1
        timeline.append(
            {
                "start": start,
                "end": end,
                "title": title,
                "summary": summary,
                "visuals": visuals,
                "screen_text": texts,
                "people": group_people[:8],
            }
        )
        descriptions.append(summary)
        for text in texts:
            on_screen_text.append({"time": start, "text": text})

    seen_text: set[str] = set()
    deduped_text: list[dict[str, Any]] = []
    for entry in on_screen_text:
        key = entry["text"].strip().lower()
        if key and key not in seen_text:
            seen_text.add(key)
            deduped_text.append(entry)

    for entry in timeline[:12]:
        key_moments.append(
            {
                "time": entry["start"],
                "label": entry["title"][:80],
                "description": entry["summary"][:240],
            }
        )

    unique_descriptions: list[str] = []
    for text in descriptions:
        if text and text not in unique_descriptions:
            unique_descriptions.append(text)

    overview_parts = [
        f"The video runs {fmt_duration(meta.get('duration'))} and was sampled at approximately "
        f"{meta.get('sample_fps', 1):.2f} frame(s) per second, giving {len(frames)} visual samples.",
        f"The footage moves through {len(timeline)} distinct visual segment(s).",
    ]
    if unique_descriptions:
        overview_parts.append("Key visual content: " + " ".join(unique_descriptions[:4]))
    if diagrams:
        overview_parts.append(
            f"The video contains {len(diagrams)} distinct diagram(s); their extracted structure and "
            "cropped images are listed in the Diagrams section."
        )
    caveat = "This report was assembled directly from frame observations without a final LLM editing pass."
    if reason:
        caveat += f" Compilation fallback reason: {reason[:300]}"
    caveats = [caveat, "Visual analysis cannot recover audio, dialogue, or off-screen events."]

    return {
        "title": as_text(meta.get("title"), 120) or "Visual report",
        "summary": " ".join(unique_descriptions[:3]) or "No visual content was described for this video.",
        "overview": "\n\n".join(overview_parts),
        "key_moments": key_moments,
        "timeline": timeline,
        "diagrams": diagrams,
        "recurring": [item for item, _ in objects.most_common(10)]
        + [item for item, _ in people.most_common(6) if item not in objects],
        "on_screen_text": deduped_text[:80],
        "caveats": caveats,
        "generated_at": now_iso(),
        "schema_version": 1,
        "model": meta.get("model"),
        "provider": meta.get("provider"),
        "stats": {
            "duration": meta.get("duration"),
            "frames_sampled": len(frames),
            "frames_analyzed": sum(
                1 for obs in observations.values() if obs.get("status") in {"ok", "partial"}
            ),
            "frames_skipped": sum(1 for obs in observations.values() if obs.get("status") == "duplicate"),
            "diagrams": len(diagrams),
            "sample_fps": meta.get("sample_fps"),
        },
    }


def _mock_note(frames: list[dict], observations: dict[int, dict], meta: dict[str, Any]) -> dict[str, Any]:
    note = _fallback_note(frames, observations, meta)
    if not note["title"].startswith("[Mock]"):
        note["title"] = f"[Mock] {note['title']}"
    note["summary"] = (
        "Mock analysis output. The extraction, batching, storage, and compilation pipeline ran "
        "end to end, but the frame descriptions were produced by the deterministic mock provider."
    )
    note["caveats"] = [
        "Generated with VIDFRAME_VLM_MODE=mock; descriptions are synthetic and do not reflect actual video content.",
        "Visual analysis cannot recover audio, dialogue, or off-screen events.",
    ]
    return note


def normalize_note(
    payload: dict[str, Any],
    frames: list[dict],
    observations: dict[int, dict],
    meta: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return _fallback_note(frames, observations, meta, "model returned a non-object response")

    fallback = _fallback_note(frames, observations, meta)
    timeline_raw = payload.get("timeline")
    timeline: list[dict[str, Any]] = []
    if isinstance(timeline_raw, list):
        for entry in timeline_raw[:MAX_TIMELINE]:
            if not isinstance(entry, dict):
                continue
            summary = as_text(entry.get("summary") or entry.get("description"), 1800)
            if not summary:
                continue
            timeline.append(
                {
                    "start": _norm_time(entry.get("start"), "0:00"),
                    "end": _norm_time(entry.get("end"), "0:00"),
                    "title": as_text(entry.get("title"), 120) or "Visual segment",
                    "summary": summary,
                    "visuals": as_list(entry.get("visuals"), 12),
                    "screen_text": as_list(entry.get("screen_text"), 30),
                    "people": as_list(entry.get("people"), 10),
                }
            )
    if not timeline:
        timeline = fallback["timeline"]

    # If the model's timeline stops early (often after a truncated response), append
    # deterministic entries from the observations so the report covers the whole video.
    duration = float(meta.get("duration") or 0)
    interval = float(meta.get("sample_fps") or 1.0) or 1.0
    covered_end = None
    for entry in timeline:
        end = parse_ts(entry.get("end")) or parse_ts(entry.get("start"))
        if end is not None:
            covered_end = end if covered_end is None else max(covered_end, end)
    if duration and covered_end is not None and duration - covered_end > max(2.0, 2.0 / interval):
        existing = {(entry.get("start"), entry.get("title")) for entry in timeline}
        for entry in fallback["timeline"]:
            start = parse_ts(entry.get("start"))
            if start is None or start < covered_end - 1.0 / interval:
                continue
            if (entry.get("start"), entry.get("title")) in existing:
                continue
            timeline.append(entry)
            if len(timeline) >= MAX_TIMELINE:
                break
        timeline.sort(key=lambda entry: parse_ts(entry.get("start")) or 0)

    # A model often ends its final entry on the last sampled frame rather than the
    # video duration. Close a small trailing gap so the timeline covers the whole clip.
    if duration and timeline:
        last = max(timeline, key=lambda entry: parse_ts(entry.get("start")) or 0)
        last_start = parse_ts(last.get("start")) or 0
        last_end = parse_ts(last.get("end")) or last_start
        if last_start <= last_end < duration and duration - last_end <= max(2.0, 2.0 / interval):
            last["end"] = fmt_ts(duration)

    key_moments: list[dict[str, Any]] = []
    raw_moments = payload.get("key_moments")
    if isinstance(raw_moments, list):
        for entry in raw_moments[:16]:
            if not isinstance(entry, dict):
                continue
            key_moments.append(
                {
                    "time": _norm_time(entry.get("time") or entry.get("timestamp"), "0:00"),
                    "label": as_text(entry.get("label") or entry.get("title"), 120),
                    "description": as_text(entry.get("description"), 500),
                }
            )
    if not key_moments:
        key_moments = fallback["key_moments"]

    screen_text: list[dict[str, Any]] = []
    raw_text = payload.get("on_screen_text")
    if isinstance(raw_text, list):
        for entry in raw_text[:120]:
            if isinstance(entry, dict):
                text = as_text(entry.get("text") or entry.get("content"), 500)
                if text:
                    screen_text.append({"time": _norm_time(entry.get("time"), "0:00"), "text": text})
            elif isinstance(entry, str) and entry.strip():
                screen_text.append({"time": "0:00", "text": entry.strip()[:500]})
    if not screen_text:
        seen: set[str] = set()
        for entry in timeline:
            for text in entry.get("screen_text", []):
                low = text.strip().lower()
                if low and low not in seen:
                    seen.add(low)
                    screen_text.append({"time": entry["start"], "text": text})
    screen_text = screen_text[:120]

    summary = as_text(payload.get("summary"), 2500) or fallback["summary"]
    overview = as_text(payload.get("overview"), 6000) or fallback["overview"]
    caveats = as_list(payload.get("caveats"), 12)
    caveats.append("Visual-only analysis: audio, dialogue, and off-screen events are not captured.")
    caveats = [caveat for caveat in dict.fromkeys(caveats) if caveat]

    diagrams = collect_diagrams(frames, observations)
    if not diagrams:
        raw_diagrams = payload.get("diagrams")
        if isinstance(raw_diagrams, list):
            for entry in raw_diagrams[:MAX_TIMELINE]:
                if not isinstance(entry, dict):
                    continue
                diagram = coerce_diagram(entry, present=True)
                if not diagram:
                    continue
                diagram["time"] = _norm_time(entry.get("time"), "0:00")
                if entry.get("image"):
                    diagram["image"] = str(entry["image"])
                diagrams.append(diagram)

    return {
        "title": as_text(payload.get("title"), 140) or fallback["title"],
        "summary": summary,
        "overview": overview,
        "key_moments": key_moments,
        "timeline": timeline,
        "diagrams": diagrams,
        "recurring": as_list(payload.get("recurring"), 15) or fallback["recurring"],
        "on_screen_text": screen_text,
        "caveats": caveats[:12],
        "generated_at": now_iso(),
        "schema_version": 1,
        "model": meta.get("model"),
        "provider": meta.get("provider"),
        "stats": {
            "duration": meta.get("duration"),
            "frames_sampled": len(frames),
            "frames_analyzed": sum(1 for obs in observations.values() if obs.get("status") == "ok"),
            "frames_skipped": sum(1 for obs in observations.values() if obs.get("status") == "duplicate"),
            "diagrams": len(diagrams),
            "sample_fps": meta.get("sample_fps"),
        },
    }


def _split_digest(digest: str, chunk_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in digest.splitlines():
        line_size = len(line) + 1
        if current and size + line_size > chunk_chars:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += line_size
    if current:
        chunks.append("\n".join(current))
    return chunks or [digest]


async def _reduce_notes(client: VLMClient, notes: str, chunk_chars: int) -> str:
    rounds = 0
    while len(notes) > chunk_chars * 1.3 and rounds < 2:
        rounds += 1
        partials: list[str] = []
        for chunk in _split_digest(notes, int(chunk_chars * 0.9)):
            partials.append(await client.chunk_notes(chunk))
        notes = "\n\n".join(partials)
    return notes


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def render_markdown(note: dict[str, Any], meta: dict[str, Any]) -> str:
    lines: list[str] = [f"# {note['title']}", ""]
    lines.append(f"> {note['summary']}")
    lines.append("")
    lines.append(
        "**Duration:** {duration} &nbsp;·&nbsp; **Sampled:** {frames} frames @ {fps:.2f} fps "
        "&nbsp;·&nbsp; **Model:** `{model}`".format(
            duration=fmt_duration(meta.get("duration")),
            frames=meta.get("frames_sampled", 0),
            fps=meta.get("sample_fps", 1.0) or 1.0,
            model=meta.get("model") or "unknown",
        )
    )
    generated = meta.get("generated_at") or note.get("generated_at") or ""
    if generated:
        lines.append(f"**Generated:** {generated} &nbsp;·&nbsp; **Mode:** visual-only")
    lines.append("")

    lines += ["## Overview", "", note.get("overview") or note.get("summary") or "", ""]

    key_moments = note.get("key_moments") or []
    if key_moments:
        lines += ["## Key moments", "", "| Time | Moment | Detail |", "| --- | --- | --- |"]
        for moment in key_moments:
            lines.append(
                f"| {_escape_cell(moment.get('time', ''))} | {_escape_cell(moment.get('label', ''))} "
                f"| {_escape_cell(moment.get('description', ''))} |"
            )
        lines.append("")

    timeline = note.get("timeline") or []
    if timeline:
        lines += ["## Visual timeline", ""]
        for entry in timeline:
            start = entry.get("start", "0:00")
            end = entry.get("end", "")
            heading = f"### {start}{(' – ' + end) if end and end != start else ''} · {entry.get('title', 'Visual segment')}"
            lines += [heading, "", entry.get("summary", ""), ""]
            visuals = entry.get("visuals") or []
            if visuals:
                lines.append("**Visuals:** " + "; ".join(visuals))
                lines.append("")
            people = entry.get("people") or []
            if people:
                lines.append("**People:** " + "; ".join(people))
                lines.append("")
            texts = entry.get("screen_text") or []
            if texts:
                lines.append("**On screen:**")
                for text in texts:
                    lines.append(f"- `{text.replace('`', '')}`")
                lines.append("")

    diagrams = note.get("diagrams") or []
    if diagrams:
        lines += ["## Diagrams", ""]
        job_id = meta.get("job_id")
        for diagram in diagrams:
            start = diagram.get("time", "")
            title = diagram.get("title") or "Diagram"
            lines += [f"### {start} · {title}".strip(" ·"), ""]
            lines.append(f"**Type:** {diagram.get('type') or 'diagram'}")
            lines.append("")
            if diagram.get("description"):
                lines += [diagram["description"], ""]
            for label, key in [
                ("Elements", "nodes"),
                ("Connections", "edges"),
                ("Labels", "labels"),
                ("Values", "values"),
            ]:
                values = diagram.get(key) or []
                if values:
                    lines.append(f"**{label}:** " + "; ".join(values))
                    lines.append("")
            image = diagram.get("image")
            if image and job_id:
                url = f"/api/jobs/{job_id}/{str(image).lstrip('/')}"
                lines += [f"![{title}]({url})", ""]

    recurring = note.get("recurring") or []
    if recurring:
        lines += ["## Recurring visual elements", ""]
        lines += [f"- {item}" for item in recurring]
        lines.append("")

    screen_text = note.get("on_screen_text") or []
    if screen_text:
        lines += ["## On-screen text index", ""]
        for entry in screen_text:
            text = str(entry.get("text", "")).replace("`", "'").strip()
            if text:
                lines.append(f"- **{entry.get('time', '')}** — `{text}`")
        lines.append("")

    caveats = note.get("caveats") or []
    lines += ["## Method & limitations", ""]
    lines.append(
        "vidFrame samples the video at a fixed interval with OpenCV, skips near-identical consecutive "
        "frames, and asks a vision-language model to describe each sampled frame. This report is "
        "compiled from those frame observations."
    )
    lines.append("")
    for caveat in caveats:
        lines.append(f"- {caveat}")
    lines += ["", "---", "", "*Generated by vidFrame · visual-only video analysis*", ""]
    return "\n".join(lines)


async def compile_report(
    frames: list[dict],
    observations: dict[int, dict],
    meta: dict[str, Any],
    client: VLMClient,
    chunk_chars: int = 12000,
    on_step: Callable[[float, str], None] | None = None,
) -> tuple[dict[str, Any], str]:
    def step(value: float, message: str) -> None:
        if on_step:
            on_step(value, message)

    should_mock = client.config.mock
    digest = build_digest(frames, observations)
    meta = dict(meta)
    meta["generated_at"] = now_iso()

    if should_mock:
        step(0.6, "Assembling mock report")
        note = _mock_note(frames, observations, meta)
        return note, render_markdown(note, meta)

    note_payload: dict[str, Any] | None = None
    reason = ""
    text = ""
    try:
        if len(digest) <= chunk_chars:
            step(0.15, "Compiling report from frame observations")
            text = await client.complete(
                COMPILE_SYSTEM,
                compile_prompt(
                    fmt_duration(meta.get("duration")),
                    float(meta.get("sample_fps") or 1.0),
                    digest,
                ),
                max_tokens=8192,
            )
        else:
            chunks = _split_digest(digest, chunk_chars)
            step(0.05, f"Summarizing {len(chunks)} observation chunks")
            partials: list[str] = []
            for position, chunk in enumerate(chunks):
                partials.append(await client.chunk_notes(chunk))
                step(0.05 + 0.45 * (position + 1) / len(chunks), f"Summarized chunk {position + 1}/{len(chunks)}")
            notes = await _reduce_notes(client, "\n\n".join(partials), chunk_chars)
            step(0.65, "Synthesizing the final report")
            text = await client.synthesize(
                fmt_duration(meta.get("duration")),
                float(meta.get("sample_fps") or 1.0),
                notes,
                max_tokens=8192,
            )

        try:
            parsed, was_repaired = extract_json_meta(text)
            if isinstance(parsed, dict):
                note_payload = parsed
                if was_repaired:
                    step(0.82, "Completing a truncated report response")
                    try:
                        repaired_text = await client.repair_json(text)
                        repaired_parsed = extract_json(repaired_text)
                        if isinstance(repaired_parsed, dict):
                            before = len(parsed.get("timeline") or [])
                            after = len(repaired_parsed.get("timeline") or [])
                            if after >= before and repaired_parsed.get("summary"):
                                note_payload = repaired_parsed
                    except Exception:  # noqa: BLE001 - partial report is still usable
                        pass
            else:
                reason = "model returned JSON that was not an object"
        except ValueError as parse_error:
            step(0.82, "Repairing a truncated report response")
            repaired = await client.repair_json(text)
            parsed = extract_json(repaired)
            if isinstance(parsed, dict):
                note_payload = parsed
            else:
                reason = f"{parse_error}; repair returned a non-object"
    except JobControlError:
        raise
    except Exception as exc:  # noqa: BLE001 - always produce a usable report
        reason = f"{type(exc).__name__}: {exc}"

    if note_payload:
        note = normalize_note(note_payload, frames, observations, meta)
    else:
        step(0.9, "Using deterministic report assembler")
        note = _fallback_note(frames, observations, meta, reason)
    return note, render_markdown(note, meta)
