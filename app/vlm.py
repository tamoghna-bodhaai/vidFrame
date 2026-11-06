from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .config import settings
from .diagrams import coerce_diagram
from .prompts import (
    CHUNK_INSTRUCTIONS,
    CHUNK_SYSTEM,
    COMPILE_SYSTEM,
    DIAGRAM_SYSTEM,
    FRAME_SYSTEM,
    REPAIR_INSTRUCTIONS,
    REPAIR_SYSTEM,
    compile_prompt,
    diagram_prompt,
    frame_prompt,
)
from .utils import as_float, as_list, as_text, extract_json, fmt_ts, strip_code_fences
from .video import analyze_image_stats


_REQUEST_SEMAPHORE: asyncio.Semaphore | None = None
_REQUEST_SEMAPHORE_LOOP: asyncio.AbstractEventLoop | None = None
_REQUEST_SEMAPHORE_LIMIT = 0


def _request_semaphore() -> asyncio.Semaphore:
    """Limit total in-flight model requests across every concurrent job."""
    global _REQUEST_SEMAPHORE, _REQUEST_SEMAPHORE_LOOP, _REQUEST_SEMAPHORE_LIMIT
    loop = asyncio.get_running_loop()
    limit = max(1, settings.max_concurrent_requests)
    if (
        _REQUEST_SEMAPHORE is None
        or _REQUEST_SEMAPHORE_LOOP is not loop
        or _REQUEST_SEMAPHORE_LIMIT != limit
    ):
        _REQUEST_SEMAPHORE = asyncio.Semaphore(limit)
        _REQUEST_SEMAPHORE_LOOP = loop
        _REQUEST_SEMAPHORE_LIMIT = limit
    return _REQUEST_SEMAPHORE


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


class VLMError(Exception):
    def __init__(self, message: str, status: int | None = None, retryable: bool = True):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class VLMResponseError(VLMError):
    def __init__(self, message: str, raw: str):
        super().__init__(message, retryable=True)
        self.raw = raw


@dataclass(slots=True)
class VLMConfig:
    model: str
    job_id: str = ""
    cost_stage: str = ""
    provider: str = field(default_factory=lambda: settings.vlm_provider)
    base_url: str = ""
    api_key: str = ""
    mode: str = field(default_factory=lambda: settings.vlm_mode)
    timeout: float = field(default_factory=lambda: settings.request_timeout)
    max_concurrency: int = field(default_factory=lambda: settings.max_concurrency)

    def __post_init__(self) -> None:
        self.provider = (self.provider or settings.vlm_provider).lower()
        if self.provider not in {"local", "openrouter"}:
            self.provider = "local"
        if self.provider == "openrouter":
            self.base_url = self.base_url or settings.openrouter_base_url
            self.api_key = self.api_key or settings.openrouter_api_key
        else:
            self.base_url = self.base_url or settings.vlm_base_url
            self.api_key = self.api_key or settings.vlm_api_key

    @property
    def mock(self) -> bool:
        return self.mode == "mock"

    @property
    def configured(self) -> bool:
        return self.mock or self.provider != "openrouter" or bool(self.api_key)

    def headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = settings.openrouter_referer
            headers["X-Title"] = "vidFrame"
        return headers

    def _require_credentials(self) -> None:
        if self.provider == "openrouter" and not self.api_key:
            raise VLMError(
                "OpenRouter API key is not configured. Set VIDFRAME_OPENROUTER_API_KEY "
                "(or OPENROUTER_API_KEY) on the server.",
                retryable=False,
            )


def _content_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


def _coerce_observation(item: dict[str, Any], fallback_index: int, model: str) -> dict[str, Any]:
    try:
        index = int(item.get("index", fallback_index))
    except (TypeError, ValueError):
        index = fallback_index
    confidence = as_float(item.get("confidence"), 0.7)
    if confidence > 1:
        confidence = confidence / 100.0
    raw_diagram = item.get("diagram")
    diagram = coerce_diagram(raw_diagram) if raw_diagram else None
    has_diagram = bool(diagram) or _truthy(item.get("has_diagram"))
    if has_diagram and not diagram:
        diagram = coerce_diagram({}, present=True)
    equations = []
    for raw in item.get("equations") or []:
        if not isinstance(raw, dict):
            continue
        equations.append({"raw": as_text(raw.get("raw") or raw.get("text"), 500), "latex": as_text(raw.get("latex"), 500), "confidence": max(0, min(1, as_float(raw.get("confidence"), confidence))), "bbox": raw.get("bbox")})
    return {
        "index": index,
        "status": "ok",
        "scene": as_text(item.get("scene"), 300),
        "summary": as_text(item.get("summary"), 500),
        "description": as_text(item.get("description"), 2500),
        "setting": as_text(item.get("setting"), 300),
        "people": as_list(item.get("people")),
        "objects": as_list(item.get("objects"), 30),
        "actions": as_list(item.get("actions"), 20),
        "screen_text": as_list(item.get("screen_text"), 40),
        "visual_notes": as_text(item.get("visual_notes"), 1500),
        "has_diagram": has_diagram,
        "diagram": diagram,
        "equations": equations,
        "confidence": max(0.0, min(1.0, confidence)),
        "model": model,
        "error": None,
    }


def _parse_frame_response(text: str, requested: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    payload = extract_json(text)
    items: Any = payload.get("frames") if isinstance(payload, dict) else payload
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list) or not items:
        raise VLMResponseError("model did not return a frames list", text)

    requested_indices = [int(item["index"]) for item in requested]
    by_index: dict[int, dict[str, Any]] = {}
    extras: list[dict[str, Any]] = []
    for position, raw in enumerate(items):
        if not isinstance(raw, dict):
            continue
        fallback = requested_indices[position] if position < len(requested_indices) else requested_indices[-1]
        observation = _coerce_observation(raw, fallback, model)
        if observation["index"] in requested_indices and observation["index"] not in by_index:
            by_index[observation["index"]] = observation
        else:
            extras.append(observation)

    results: list[dict[str, Any]] = []
    for requested_index in requested_indices:
        if requested_index in by_index:
            results.append(by_index[requested_index])
        elif extras:
            observation = extras.pop(0)
            observation["index"] = requested_index
            results.append(observation)
        else:
            results.append(
                {
                    "index": requested_index,
                    "status": "error",
                    "model": model,
                    "error": "Model response did not include this frame",
                }
            )
    return results


class VLMClient:
    def __init__(self, config: VLMConfig):
        self.config = config
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "VLMClient":
        if not self.config.mock:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout, connect=20.0),
                headers=self.config.headers(),
            )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client:
            await self._client.aclose()

    def _url(self, path: str) -> str:
        return f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"

    async def list_models(self) -> list[str]:
        if self.config.mock:
            return ["mock-vision"]
        self.config._require_credentials()
        assert self._client is not None
        response = await self._client.get(self._url("models"), timeout=12.0)
        response.raise_for_status()
        payload = response.json()
        return [str(item.get("id")) for item in payload.get("data", []) if item.get("id")]

    async def analyze_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not batch:
            return []
        if self.config.mock:
            await asyncio.sleep(0.05)
            return [self._mock_observation(item) for item in batch]

        assert self._client is not None
        last_error: Exception | None = None
        last_raw = ""

        for attempt in range(3):
            try:
                text = await self._chat(
                    self._frame_messages(batch),
                    max_tokens=min(8192, 800 + 620 * len(batch)),
                )
                return _parse_frame_response(text, batch, self.config.model)
            except VLMResponseError as exc:
                last_error, last_raw = exc, exc.raw
            except (httpx.HTTPError, VLMError, ValueError) as exc:
                last_error = exc
                if isinstance(exc, VLMError) and not exc.retryable:
                    break
            if attempt < 2:
                await asyncio.sleep(0.8 * (2**attempt))

        split_worthy = isinstance(last_error, VLMResponseError) or (
            isinstance(last_error, VLMError) and last_error.status in {400, 413, 422}
        )
        if split_worthy and len(batch) > 1:
            midpoint = len(batch) // 2
            left = await self.analyze_batch(batch[:midpoint])
            right = await self.analyze_batch(batch[midpoint:])
            return left + right

        if last_raw and len(batch) == 1:
            raw = strip_code_fences(last_raw)
            if raw:
                return [
                    {
                        "index": batch[0]["index"],
                        "status": "partial",
                        "scene": raw[:180],
                        "summary": raw[:300],
                        "description": raw[:1800],
                        "setting": "",
                        "people": [],
                        "objects": [],
                        "actions": [],
                        "screen_text": [],
                        "visual_notes": "Model returned unstructured text.",
                        "confidence": 0.3,
                        "model": self.config.model,
                        "error": "Response was not valid JSON",
                    }
                ]

        message = str(last_error) if last_error else "Unknown VLM failure"
        return [
            {
                "index": item["index"],
                "status": "error",
                "model": self.config.model,
                "error": message[:600],
            }
            for item in batch
        ]

    async def analyze_diagram(
        self,
        item: dict[str, Any],
        image_b64: str,
        observation: dict[str, Any],
        diagram: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Extract diagram geometry from a crop with a vision model."""
        if self.config.mock:
            await asyncio.sleep(0.02)
            return self._mock_diagram(item, diagram)

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": diagram_prompt(
                    fmt_ts(item.get("time", 0)),
                    as_text(observation.get("description") or observation.get("summary"), 1200),
                    diagram,
                ),
            },
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]
        text = await self._chat(
            [
                {"role": "system", "content": DIAGRAM_SYSTEM},
                {"role": "user", "content": content},
            ],
            max_tokens=2600,
        )
        parsed = extract_json(text)
        if not isinstance(parsed, dict):
            raise VLMResponseError("diagram model did not return a JSON object", text)
        detail = coerce_diagram(parsed)
        if not detail:
            if parsed.get("present") is False:
                return None
            raise VLMResponseError("diagram model returned no diagram details", text)
        detail["analysis_model"] = self.config.model
        return detail

    async def _chat(self, messages: list[dict[str, Any]], max_tokens: int = 4096, temperature: float = 0.1) -> str:
        self.config._require_credentials()
        assert self._client is not None
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        try:
            async with _request_semaphore():
                response = await self._client.post(self._url("chat/completions"), json=payload)
        except httpx.TimeoutException as exc:
            raise VLMError(f"VLM request timed out after {self.config.timeout:.0f}s") from exc
        except httpx.HTTPError as exc:
            raise VLMError(f"VLM connection failed: {exc}") from exc

        if response.status_code >= 400:
            body = response.text[:500].replace("\n", " ")
            retryable = response.status_code in {408, 409, 429, 500, 502, 503, 504}
            raise VLMError(f"VLM HTTP {response.status_code}: {body}", status=response.status_code, retryable=retryable)
        try:
            data = response.json()
            text = _content_text(data["choices"][0]["message"])
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise VLMError(f"Unexpected VLM response shape: {exc}") from exc
        if not text.strip():
            raise VLMError("VLM returned an empty response")
        if self.config.job_id:
            usage = data.get("usage") or {}
            def count(value):
                try:
                    return int(value) if value is not None else None
                except (TypeError, ValueError):
                    return None
            def charge(value):
                try:
                    return float(value) if value is not None else None
                except (TypeError, ValueError):
                    return None
            from . import db
            db.add_model_cost(self.config.job_id, self.config.cost_stage, self.config.provider,
                              self.config.model, count(usage.get("prompt_tokens")),
                              count(usage.get("completion_tokens")), charge(usage.get("cost")))
        return text

    def _frame_messages(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": frame_prompt(batch)}]
        for item in batch:
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"[frame {item['index']} at {item['timestamp_label']}] "
                        f"(scene {item.get('scene', 0)})"
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{item['image_b64']}"},
                }
            )
        return [
            {"role": "system", "content": FRAME_SYSTEM},
            {"role": "user", "content": content},
        ]

    def _mock_observation(self, item: dict[str, Any]) -> dict[str, Any]:
        stats = analyze_image_stats(Path(item["path"]))
        rng = random.Random(1000 + int(item["index"]))
        timestamp = fmt_ts(item.get("time", 0))
        dominant = stats.get("dominant", "neutral")
        detail = stats.get("detail", 0)
        brightness = stats.get("brightness", 0.5)

        if brightness < 0.12:
            scene = "A near-black frame with almost no visible detail"
            description = (
                "The frame is almost completely black, with only faint noise visible. "
                "No subjects, objects, or legible text can be identified."
            )
            screen_text: list[str] = []
            people: list[str] = []
            objects: list[str] = []
            actions: list[str] = []
        elif detail > 0.10:
            scene = f"A detailed {dominant} interface or document fills the frame"
            description = (
                f"A dense composition in a {dominant} palette with high visual detail, likely a screen "
                "capture containing text, charts, or interface elements. The layout is structured in "
                f"rectangular regions, and the overall exposure is {'bright' if brightness > 0.55 else 'dim'}."
            )
            screen_text = [f"MOCK UI · {dominant.upper()}", f"FRAME {item['index']:04d}"]
            people = []
            objects = ["Interface", "Text blocks"]
            actions = ["Displaying information"]
        else:
            scene = f"A simple {dominant} scene with a central subject"
            description = (
                f"A low-detail frame dominated by {dominant} tones, with a central subject and an "
                f"uncluttered background. Lighting is {'bright' if brightness > 0.55 else 'soft and dim'}, "
                "and the composition resembles a talking-head or product shot."
            )
            screen_text = []
            people = ["One presenter (mock)"] if brightness >= 0.12 else []
            objects = ["Product"]
            actions = ["Standing"]

        has_diagram = detail > 0.12 and brightness >= 0.12
        diagram = None
        if has_diagram:
            diagram = {
                "present": True,
                "type": "bar chart" if detail > 0.14 else "table",
                "title": f"Mock {dominant} performance panel",
                "description": (
                    f"A labeled {dominant} panel with rectangular regions and readable values. "
                    "The mock provider treats high-detail interface frames as diagrams."
                ),
                "nodes": ["Revenue axis", "Monthly bars", "Summary card"],
                "edges": ["Monthly bars -> revenue trend", "Summary card -> total"],
                "labels": ["Month", "Revenue", "Target"],
                "values": [f"Frame {item['index']:04d}", "May: $96.4k"],
                "bbox": [0.04, 0.04, 0.92, 0.92],
                "confidence": round(rng.uniform(0.6, 0.85), 2),
            }

        return {
            "index": item["index"],
            "status": "ok",
            "scene": scene,
            "summary": f"Mock visual summary at {timestamp}.",
            "description": description,
            "setting": "Mock studio setting",
            "people": people,
            "objects": objects,
            "actions": actions,
            "screen_text": screen_text,
            "visual_notes": f"dominant={dominant}; detail={detail:.3f}; brightness={brightness:.2f}",
            "has_diagram": has_diagram,
            "diagram": diagram,
            "confidence": round(rng.uniform(0.55, 0.8), 2),
            "model": "mock-vision",
            "error": None,
        }

    def _mock_diagram(self, item: dict[str, Any], diagram: dict[str, Any]) -> dict[str, Any]:
        rng = random.Random(2000 + int(item.get("index", 0)))
        base = dict(diagram or {})
        base.update(
            {
                "present": True,
                "type": base.get("type") or "bar chart",
                "title": base.get("title") or f"Diagram at {fmt_ts(item.get('time', 0))}",
                "description": (
                    "The diagram combines a labeled vertical axis with monthly bars and a summary card. "
                    f"The mock image pass confirms the structure at {fmt_ts(item.get('time', 0))}."
                ),
                "nodes": ["Revenue axis", "May bar", "June bar", "Summary card"],
                "edges": ["May bar -> rising trend", "Summary card -> total revenue"],
                "labels": ["Month", "Revenue", "Target"],
                "values": [f"Frame {int(item.get('index', 0)):04d}", "May: $96.4k"],
                "analysis_model": "mock-vision",
                "confidence": round(rng.uniform(0.65, 0.9), 2),
            }
        )
        return base

    async def complete(self, system: str, user: str, max_tokens: int = 4096) -> str:
        if self.config.mock:
            raise VLMError("Mock mode does not run compilation requests", retryable=False)
        return await self._chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.2,
        )

    async def chunk_notes(self, digest: str) -> str:
        return await self.complete(CHUNK_SYSTEM, CHUNK_INSTRUCTIONS + digest, max_tokens=1800)

    async def synthesize(self, duration_label: str, interval: float, notes: str, max_tokens: int = 8192) -> str:
        return await self.complete(
            COMPILE_SYSTEM,
            compile_prompt(duration_label, interval, notes),
            max_tokens=max_tokens,
        )

    async def repair_json(self, raw: str) -> str:
        return await self.complete(REPAIR_SYSTEM, REPAIR_INSTRUCTIONS + raw[-14000:], max_tokens=8192)


async def check_health(config: VLMConfig | None = None, provider: str | None = None) -> dict[str, Any]:
    provider = (provider or settings.vlm_provider or "local").lower()
    if config is None:
        model = settings.openrouter_model if provider == "openrouter" else settings.vlm_model
        config = VLMConfig(model=model, provider=provider)
    cfg = config
    base = {
        "provider": cfg.provider,
        "configured": cfg.configured,
        "base_url": cfg.base_url,
        "model": cfg.model,
        "models": [],
        "latency_ms": 0,
    }
    if cfg.mock:
        return {**base, "ok": True, "mode": "mock", "models": ["mock-vision"]}
    if not cfg.configured:
        return {
            **base,
            "ok": False,
            "mode": cfg.mode,
            "error": "OpenRouter API key is not configured on the server.",
        }
    started = time.monotonic()
    try:
        async with VLMClient(cfg) as client:
            models = await client.list_models()
        return {
            **base,
            "ok": True,
            "mode": cfg.mode,
            "models": models,
            "latency_ms": int((time.monotonic() - started) * 1000),
        }
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        return {
            **base,
            "ok": False,
            "mode": cfg.mode,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "error": str(exc)[:500],
        }
