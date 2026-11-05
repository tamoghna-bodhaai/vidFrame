from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fmt_ts(seconds: float | int | None) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    if seconds is None:
        return "0:00"
    seconds = max(0, int(round(float(seconds))))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def parse_ts(value: Any) -> float | None:
    """Parse M:SS, H:MM:SS, or numeric seconds into seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return None
    match = re.match(r"^(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?$", text)
    if match:
        parts = [int(part) for part in match.groups() if part is not None]
        return float(parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0] * 3600 + parts[1] * 60 + parts[2])
    try:
        return float(text)
    except ValueError:
        return None


def fmt_duration(seconds: float | int | None) -> str:
    if not seconds:
        return "0s"
    seconds = int(round(float(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def json_dumps(value: Any, **kwargs: Any) -> str:
    return json.dumps(value, ensure_ascii=False, **kwargs)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _escape_control_chars(text: str) -> str:
    out: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                out.append(char)
                escaped = False
                continue
            if char == "\\":
                out.append(char)
                escaped = True
                continue
            if char == '"':
                in_string = False
                out.append(char)
                continue
            if char == "\n":
                out.append("\\n")
                continue
            if char == "\r":
                out.append("\\r")
                continue
            if char == "\t":
                out.append("\\t")
                continue
            if ord(char) < 0x20:
                out.append(f"\\u{ord(char):04x}")
                continue
            out.append(char)
        else:
            if char == '"':
                in_string = True
            out.append(char)
    if in_string:
        if escaped:
            out.append("\\\\")
        out.append('"')
    return "".join(out)


def _balance_json(text: str) -> str:
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]" and stack:
            opening = stack[-1]
            if (opening == "{" and char == "}") or (opening == "[" and char == "]"):
                stack.pop()
    if in_string:
        if escaped:
            text += "\\\\"
        text += '"'
    for opening in reversed(stack):
        text += "}" if opening == "{" else "]"
    return text


def _strip_trailing_commas(text: str) -> str:
    out: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue
        if char == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead] in " \t\r\n":
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "}]":
                index += 1
                continue
        out.append(char)
        index += 1
    return "".join(out)


def _repair_json(text: str) -> str:
    candidate = _escape_control_chars(text)
    candidate = _balance_json(candidate)
    candidate = _strip_trailing_commas(candidate)
    return candidate


def extract_json_meta(text: str) -> tuple[Any, bool]:
    """Extract JSON and report whether structural repair was required."""
    cleaned = strip_code_fences(text)
    starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON found in response")
    start = min(starts)
    payload = cleaned[start:]

    try:
        value, _ = json.JSONDecoder().raw_decode(payload)
        return value, False
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(payload), False
    except json.JSONDecodeError:
        pass

    repaired = _repair_json(payload)
    try:
        return json.loads(repaired), True
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in response: {exc}") from exc


def extract_json(text: str) -> Any:
    """Best-effort extraction of a JSON object/array from model output.

    Handles code fences, leading prose, literal control characters inside strings,
    trailing commas, and responses truncated mid-string or mid-structure.
    """
    return extract_json_meta(text)[0]


def as_list(value: Any, limit: int = 24) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [part.strip() for part in re.split(r"[;\n]|(?<=\.)\s+(?=[A-Z])", value) if part.strip()]
    if not isinstance(value, (list, tuple)):
        value = [str(value)]
    out: list[str] = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("text") or item.get("name") or json_dumps(item)
        text = str(item).strip()
        if text and text.lower() not in {"none", "null", "n/a", "[]"}:
            out.append(text[:500])
    return out[:limit]


def as_text(value: Any, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(as_list(value, 40))[:limit]
    return str(value).strip()[:limit]


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
