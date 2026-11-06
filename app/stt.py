"""OpenAI-compatible speech-to-text artifact support."""
from __future__ import annotations

import asyncio
import base64
import json
import subprocess
from pathlib import Path
from typing import Any

import httpx

from .config import settings
from . import db


async def extract_audio(source: Path, output: Path) -> None:
    """Extract the durable mono WAV evidence artifact used by the report."""
    output.parent.mkdir(parents=True, exist_ok=True)

    def run() -> None:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", str(output),
            ],
            check=True,
            capture_output=True,
        )

    await asyncio.to_thread(run)


async def _split_for_transcription(audio: Path, chunk_dir: Path, seconds: int) -> list[Path]:
    """Create compact, bounded multipart uploads from a durable WAV artifact."""
    chunk_dir.mkdir(parents=True, exist_ok=True)

    def run() -> None:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(audio), "-map", "0:a:0", "-ac", "1", "-ar", "16000",
                "-c:a", "libmp3lame", "-b:a", "32k", "-f", "segment", "-segment_time", str(seconds),
                "-reset_timestamps", "1", str(chunk_dir / "chunk-%04d.mp3"),
            ],
            check=True,
            capture_output=True,
        )

    await asyncio.to_thread(run)
    chunks = sorted(chunk_dir.glob("chunk-*.mp3"))
    if not chunks:
        raise RuntimeError("FFmpeg did not produce transcription chunks")
    # OpenRouter multipart uploads are capped at 25 MB. At 32 kbps, a 15-minute
    # chunk is roughly 3.6 MB; keep the guard for custom duration/bitrate changes.
    oversized = [path.name for path in chunks if path.stat().st_size > 24 * 1024 * 1024]
    if oversized:
        raise RuntimeError(f"Transcription chunk exceeds the multipart upload limit: {', '.join(oversized[:3])}")
    return chunks


async def _transcribe_file(audio: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Send one small audio file to an OpenAI-compatible transcription endpoint."""
    if config.get("stt_mode") == "mock" or settings.vlm_mode == "mock":
        return {"language": "en", "segments": [], "text": "", "model": config["stt_model"]}
    data = {"model": config["stt_model"], "response_format": "verbose_json"}
    language = config.get("stt_language", "auto")
    if language and language != "auto": data["language"] = language
    headers = {"Authorization": f"Bearer {config['stt_api_key']}"}
    async with httpx.AsyncClient(timeout=config["stt_timeout"], headers=headers) as client:
        endpoint = config["stt_base_url"].rstrip("/") + "/audio/transcriptions"
        if config.get("stt_provider") == "openrouter":
            # OpenRouter's native contract uses base64 JSON. Keeping chunks small
            # avoids request-size pressure while avoiding multipart proxy resets.
            data["input_audio"] = {
                "data": base64.b64encode(audio.read_bytes()).decode("ascii"),
                "format": audio.suffix.lstrip(".").lower() or "mp3",
            }
            response = await client.post(endpoint, json=data)
        else:
            with audio.open("rb") as handle:
                response = await client.post(endpoint, data=data, files={"file": (audio.name, handle, "audio/mpeg")})
    response.raise_for_status()
    payload = response.json()
    usage = payload.get("usage") or {}
    if config.get("job_id"):
        def optional_number(value: Any, converter: Any) -> Any:
            try:
                return converter(value) if value is not None else None
            except (TypeError, ValueError):
                return None
        db.add_model_cost(
            config["job_id"], "Transcription", config.get("stt_provider", "local"), config["stt_model"],
            optional_number(usage.get("input_tokens", usage.get("prompt_tokens")), int),
            optional_number(usage.get("output_tokens", usage.get("completion_tokens")), int),
            optional_number(usage.get("cost"), float),
            optional_number(usage.get("seconds"), float),
        )
    segments = []
    for index, segment in enumerate(payload.get("segments") or []):
        segments.append({"id": segment.get("id", index), "start": float(segment.get("start", 0)), "end": float(segment.get("end", segment.get("start", 0))), "text": str(segment.get("text", "")).strip(), "confidence": float(segment.get("avg_logprob", segment.get("confidence", 0.7)))})
    return {"language": payload.get("language") or language or "auto", "segments": segments, "text": payload.get("text", ""), "model": config["stt_model"]}


async def transcribe(audio: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Transcribe long recordings in timestamped, API-safe chunks.

    Long uncompressed WAV uploads can exceed OpenRouter's multipart limit or its
    upstream processing window. The original WAV remains the evidence artifact;
    MP3 chunks are only compact transport files and are retained for traceability.
    """
    if config.get("stt_mode") == "mock" or settings.vlm_mode == "mock":
        return await _transcribe_file(audio, config)

    chunk_seconds = int(config.get("stt_chunk_seconds") or 300)
    chunks = await _split_for_transcription(audio, audio.parent / "audio-chunks", chunk_seconds)
    all_segments: list[dict[str, Any]] = []
    text_parts: list[str] = []
    warnings: list[str] = []
    language = str(config.get("stt_language") or "auto")
    for chunk_index, chunk in enumerate(chunks):
        try:
            result = await _transcribe_file(chunk, config)
        except Exception as exc:  # Preserve partial work and surface the failed source window.
            warnings.append(f"{chunk.name}: {type(exc).__name__}: {exc}")
            continue
        language = str(result.get("language") or language)
        offset = chunk_index * chunk_seconds
        segments = result.get("segments") or []
        if segments:
            for segment_index, segment in enumerate(segments):
                text = str(segment.get("text") or "").strip()
                if text:
                    all_segments.append({
                        "id": len(all_segments),
                        "start": round(offset + float(segment.get("start") or 0), 3),
                        "end": round(offset + float(segment.get("end") or segment.get("start") or 0), 3),
                        "text": text,
                        "confidence": float(segment.get("confidence", 0.7)),
                        "source_chunk": chunk_index,
                    })
        elif str(result.get("text") or "").strip():
            text = str(result["text"]).strip()
            all_segments.append({"id": len(all_segments), "start": offset, "end": offset + chunk_seconds, "text": text, "confidence": 0.7, "source_chunk": chunk_index})
        if str(result.get("text") or "").strip():
            text_parts.append(str(result["text"]).strip())
    if not all_segments and warnings:
        raise RuntimeError("All transcription chunks failed: " + "; ".join(warnings[:3]))
    result = {"language": language, "segments": all_segments, "text": "\n".join(text_parts), "model": config["stt_model"], "chunks": len(chunks)}
    if warnings:
        result["warnings"] = warnings
    return result


def load_transcript(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"language": "auto", "segments": [], "text": ""}
