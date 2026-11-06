from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, db, pipeline, explainers
from .config import settings
from .utils import fmt_ts
from .vlm import VLMConfig, check_health
from .study import export_pdf

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpeg", ".mpg", ".ts"}
TERMINAL_STATUSES = {"done", "failed", "cancelled", "interrupted", "paused"}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    db.mark_interrupted()
    db.mark_explainers_interrupted()
    settings.ensure_dirs()
    for record in db.interrupted_explainers():
        job = db.get_job(record["job_id"])
        if job and explainers.available(job):
            db.update_explainer(job["id"], record["id"], status="queued", progress=0, message="Resuming after server restart", error=None)
            explainers.start(job, record)
    yield


app = FastAPI(
    title="vidFrame",
    description="Frame-by-frame visual analysis of video with a vision-language model.",
    version=__version__,
    lifespan=lifespan,
)


def _serialize(job: dict[str, Any] | None) -> dict[str, Any] | None:
    if not job:
        return None
    job = dict(job)
    job_dir = Path(job.get("dir") or "")
    job["note_available"] = bool(job_dir and (job_dir / "note.md").exists())
    job["note_json_available"] = bool(job_dir and (job_dir / "note.json").exists())
    job["live_notes_available"] = bool(job_dir and (job_dir / "live-notes.md").exists())
    job["observations_available"] = bool(job_dir and (job_dir / "observations.jsonl").exists())
    job["frames_available"] = bool(job_dir and (job_dir / "frames.json").exists())
    job["source_available"] = bool(job_dir and any(job_dir.glob("source.*")))
    job["transcript_available"] = bool(job_dir and (job_dir / "transcript.json").exists())
    job["audio_available"] = bool(job_dir and (job_dir / "audio.wav").exists())
    job["pdf_available"] = bool(job_dir and any((job_dir / "pdf" / name).exists() for name in ("video_notes.pdf", "study-notes.pdf")))
    job["active"] = pipeline.is_running(job["id"])
    job["display_title"] = job.get("note_title") or job.get("title") or job.get("filename")
    if isinstance(job.get("config"), dict):
        job["config"] = {key: value for key, value in job["config"].items() if key not in {"stt_api_key", "image_api_key"}}
    job.pop("dir", None)
    return job


def _require_job(job_id: str) -> dict[str, Any]:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _job_dir(job: dict[str, Any]) -> Path:
    return Path(job["dir"])


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "version": __version__, "mode": settings.vlm_mode}


@app.get("/api/config")
async def config() -> dict[str, Any]:
    data = settings.public()
    data.update({"version": __version__, "mock": settings.vlm_mode == "mock"})
    return data


@app.get("/api/vlm/models")
async def vlm_models(provider: Literal["local", "openrouter"] | None = None) -> dict[str, Any]:
    selected = provider or settings.vlm_provider or "local"
    model = settings.openrouter_model if selected == "openrouter" else settings.vlm_model
    health_info = await check_health(VLMConfig(model=model, provider=selected), provider=selected)
    return {
        **health_info,
        "configured_model": health_info.get("model") or model,
    }


class CompileOptions(BaseModel):
    provider: Literal["local", "openrouter"] | None = None
    compile_model: str | None = Field(default=None, max_length=200)
    request_timeout: float | None = Field(default=None, ge=10, le=900)
    stt_base_url: str | None = None
    stt_api_key: str | None = None
    stt_model: str | None = None
    stt_timeout: float | None = Field(default=None, ge=10, le=1800)
    stt_language: str | None = None
    stt_provider: str | None = None
    include_transcript_appendix: bool | None = None


class AnalyzeOptions(CompileOptions):
    model: str | None = Field(default=None, max_length=200)
    batch_size: int | None = Field(default=None, ge=1, le=8)


class ReprocessOptions(AnalyzeOptions):
    sample_fps: float | None = Field(default=None, ge=0.05, le=10)
    max_frames: int | None = Field(default=None, ge=10, le=5000)


@app.post("/api/jobs", status_code=201)
async def create_job(
    file: UploadFile = File(...),
    sample_fps: float | None = Form(default=None),
    max_frames: int | None = Form(default=None),
    batch_size: int | None = Form(default=None),
    provider: str | None = Form(default=None),
    model: str | None = Form(default=None),
    compile_model: str | None = Form(default=None),
    request_timeout: float | None = Form(default=None),
    stt_base_url: str | None = Form(default=None),
    stt_api_key: str | None = Form(default=None),
    stt_model: str | None = Form(default=None),
    stt_timeout: float | None = Form(default=None),
    stt_language: str | None = Form(default=None),
    stt_provider: str | None = Form(default=None),
    include_transcript_appendix: bool | None = Form(default=None),
) -> dict[str, Any]:
    filename = Path(file.filename or "video.mp4").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video type '{suffix or 'unknown'}'. Use one of: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    config = pipeline.normalize_config(
        {},
        {
            "sample_fps": sample_fps,
            "max_frames": max_frames,
            "batch_size": batch_size,
            "provider": provider,
            "model": model,
            "compile_model": compile_model,
            "request_timeout": request_timeout,
            "stt_base_url": stt_base_url,
            "stt_api_key": stt_api_key,
            "stt_model": stt_model,
            "stt_timeout": stt_timeout,
            "stt_language": stt_language,
            "stt_provider": stt_provider,
            "include_transcript_appendix": include_transcript_appendix,
            "original_filename": filename,
            "source_file": f"source{suffix}",
        },
    )
    job = db.create_job(filename, config, "")
    job_dir = settings.jobs_dir / job["id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    source_path = job_dir / f"source{suffix}"
    db.update_job(job["id"], dir=str(job_dir), config_json=json.dumps(config), model=config["model"])

    limit = settings.max_upload_mb * 1024 * 1024
    written = 0
    try:
        with source_path.open("wb") as target:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds the {settings.max_upload_mb} MB limit.",
                    )
                target.write(chunk)
        if written == 0:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        db.delete_job(job["id"])
        raise
    finally:
        await file.close()

    db.add_event(job["id"], "upload", f"Uploaded {filename} ({written / 1_048_576:.1f} MB)")
    pipeline.start_job(job["id"], "full")
    return _serialize(_require_job(job["id"]))  # type: ignore[return-value]


@app.get("/api/jobs")
async def list_jobs(limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
    return {"jobs": [_serialize(job) for job in db.list_jobs(limit)]}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    return _serialize(_require_job(job_id))  # type: ignore[return-value]


@app.get("/api/jobs/{job_id}/costs")
async def get_job_costs(job_id: str) -> dict[str, Any]:
    _require_job(job_id)
    return {"items": db.get_model_costs(job_id)}


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request, after: int = 0) -> StreamingResponse:
    _require_job(job_id)

    async def generate():
        last_id = max(0, after)
        while True:
            if await request.is_disconnected():
                break
            try:
                events = db.get_events(job_id, last_id)
                for event in events:
                    last_id = event["id"]
                    yield f"event: log\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                job = db.get_job(job_id)
                if job:
                    payload = _serialize(job)
                    yield f"event: status\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    if job["status"] in TERMINAL_STATUSES:
                        break
                else:
                    break
            except Exception as exc:  # noqa: BLE001 - keep the stream alive
                yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
                break
            await asyncio.sleep(0.8)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict[str, Any]:
    job = _require_job(job_id)
    if not pipeline.is_running(job_id):
        raise HTTPException(status_code=409, detail="Job is not running")
    pipeline.request_cancel(job_id)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/pause")
async def pause_job(job_id: str) -> dict[str, Any]:
    _require_job(job_id)
    if not pipeline.request_pause(job_id):
        raise HTTPException(status_code=409, detail="Job is not active")
    return {"ok": True}


@app.post("/api/jobs/{job_id}/resume")
async def resume_job(job_id: str) -> dict[str, Any]:
    job = _require_job(job_id)
    if job["status"] not in {"paused", "interrupted"}:
        raise HTTPException(status_code=409, detail="Only paused or interrupted jobs can be resumed")
    if not pipeline.resume_job(job_id):
        raise HTTPException(status_code=409, detail="Job could not be resumed")
    return {"ok": True}


@app.post("/api/jobs/{job_id}/recompile")
async def recompile_job(job_id: str, options: CompileOptions | None = None) -> dict[str, Any]:
    job = _require_job(job_id)
    if pipeline.is_running(job_id):
        raise HTTPException(status_code=409, detail="Job is already running")
    job_dir = _job_dir(job)
    if not (job_dir / "observations.jsonl").exists():
        raise HTTPException(status_code=409, detail="No stored observations to compile from")
    overrides = options.model_dump(exclude_none=True) if options else {}
    pipeline.start_job(job_id, "compile", overrides)
    return {"ok": True, "mode": "compile"}


@app.post("/api/jobs/{job_id}/reanalyze")
async def reanalyze_job(job_id: str, options: AnalyzeOptions | None = None) -> dict[str, Any]:
    job = _require_job(job_id)
    if pipeline.is_running(job_id):
        raise HTTPException(status_code=409, detail="Job is already running")
    job_dir = _job_dir(job)
    if not (job_dir / "frames.json").exists():
        raise HTTPException(status_code=409, detail="No extracted frames to reanalyze")
    overrides = options.model_dump(exclude_none=True) if options else {}
    pipeline.start_job(job_id, "analyze", overrides)
    return {"ok": True, "mode": "analyze"}


@app.post("/api/jobs/{job_id}/reprocess")
async def reprocess_job(job_id: str, options: ReprocessOptions | None = None) -> dict[str, Any]:
    job = _require_job(job_id)
    if pipeline.is_running(job_id):
        raise HTTPException(status_code=409, detail="Job is already running")
    job_dir = _job_dir(job)
    if not any(job_dir.glob("source.*")):
        raise HTTPException(status_code=409, detail="The original source video is no longer available")
    overrides = options.model_dump(exclude_none=True) if options else {}
    pipeline.start_job(job_id, "full", overrides)
    return {"ok": True, "mode": "full"}


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str) -> dict[str, Any]:
    job = _require_job(job_id)
    if pipeline.is_running(job_id):
        raise HTTPException(status_code=409, detail="Cancel the job before deleting it")
    if explainers.is_running(job_id):
        raise HTTPException(status_code=409, detail="Wait for the explainer render before deleting the lecture")
    shutil.rmtree(_job_dir(job), ignore_errors=True)
    db.delete_job(job_id)
    return {"ok": True}


class ExplainerRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=200)


def _explainer_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in row.items() if key != "output_dir"}
    if row["status"] == "done":
        base = f"/api/jobs/{row['job_id']}/explainers/{row['id']}"
        payload["playback_url"] = base + "/video"
        payload["download_url"] = base + "/download"
    return payload


@app.post("/api/jobs/{job_id}/explainers", status_code=202)
async def create_explainer(job_id: str, body: ExplainerRequest) -> dict[str, Any]:
    job = _require_job(job_id)
    if not explainers.available(job):
        raise HTTPException(status_code=409, detail="Completed study notes, transcript, and teacher speech are required")
    topic = body.topic.strip()
    if len(topic) < 3:
        raise HTTPException(status_code=422, detail="Enter a topic from this lecture")
    if not settings.openrouter_api_key:
        raise HTTPException(status_code=409, detail="Set VIDFRAME_OPENROUTER_API_KEY for cloned narration")
    if not explainers.manim_command():
        raise HTTPException(status_code=409, detail="Install Manim to render explainers")
    digest = hashlib.sha256((_job_dir(job) / "note.json").read_bytes()).hexdigest()
    record = db.create_explainer(job_id, topic, digest, explainers.TTS_MODEL, str(_job_dir(job) / "explainers"))
    explainers.start(job, record)
    return _explainer_payload(record)


@app.get("/api/jobs/{job_id}/explainers")
async def list_explainers(job_id: str) -> dict[str, Any]:
    job = _require_job(job_id)
    return {"available": explainers.available(job), "items": [_explainer_payload(x) for x in db.list_explainers(job_id)]}


@app.get("/api/jobs/{job_id}/explainers/{explainer_id}")
async def get_explainer(job_id: str, explainer_id: str) -> dict[str, Any]:
    _require_job(job_id)
    row = db.get_explainer(job_id, explainer_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Explainer not found")
    return _explainer_payload(row)


def _explainer_video(job_id: str, explainer_id: str, download: bool) -> FileResponse:
    _require_job(job_id)
    row = db.get_explainer(job_id, explainer_id)
    if row is None or row["status"] != "done":
        raise HTTPException(status_code=404, detail="Explainer video not ready")
    path = Path(row["output_dir"]) / "explainer.mp4"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Explainer video missing")
    return FileResponse(path, media_type="video/mp4", filename=f"explainer-{explainer_id}.mp4" if download else None)


@app.get("/api/jobs/{job_id}/explainers/{explainer_id}/video")
def play_explainer(job_id: str, explainer_id: str) -> FileResponse:
    return _explainer_video(job_id, explainer_id, False)


@app.get("/api/jobs/{job_id}/explainers/{explainer_id}/download")
def download_explainer(job_id: str, explainer_id: str) -> FileResponse:
    return _explainer_video(job_id, explainer_id, True)


def _frames_payload(job: dict[str, Any]) -> dict[str, Any]:
    path = _job_dir(job) / "frames.json"
    if not path.exists():
        return {"video": {}, "sample_fps": 0, "scene_count": 0, "frames": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _observations(job: dict[str, Any]) -> dict[int, dict]:
    path = _job_dir(job) / "observations.jsonl"
    observations: dict[int, dict] = {}
    if not path.exists():
        return observations
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            observations[int(item["index"])] = item
        except (ValueError, KeyError, TypeError):
            continue
    return observations


@app.get("/api/jobs/{job_id}/frames")
async def get_frames(
    job_id: str,
    scene: int | None = None,
    duplicates: bool = True,
    q: str | None = None,
    limit: int = Query(default=400, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    job = _require_job(job_id)
    payload = _frames_payload(job)
    observations = _observations(job)
    frames = payload.get("frames", [])
    if scene is not None:
        frames = [frame for frame in frames if frame.get("scene") == scene]
    if not duplicates:
        frames = [frame for frame in frames if not frame.get("duplicate")]
    if q:
        needle = q.lower()
        filtered = []
        for frame in frames:
            observation = observations.get(frame["index"], {})
            haystack = " ".join(
                str(value)
                for value in [
                    observation.get("scene"),
                    observation.get("summary"),
                    observation.get("description"),
                    observation.get("setting"),
                    observation.get("visual_notes"),
                    " ".join(observation.get("screen_text") or []),
                    " ".join(observation.get("objects") or []),
                    " ".join(observation.get("people") or []),
                ]
            ).lower()
            if needle in haystack:
                filtered.append(frame)
        frames = filtered

    total = len(frames)
    page = frames[offset : offset + limit]
    result = []
    for frame in page:
        item = dict(frame)
        observation = observations.get(frame["index"])
        if observation:
            item["observation"] = observation
        result.append(item)
    return {
        "video": payload.get("video", {}),
        "sample_fps": payload.get("sample_fps", 0),
        "scene_count": payload.get("scene_count", 0),
        "total": total,
        "offset": offset,
        "frames": result,
    }


@app.get("/api/jobs/{job_id}/frames/{index}/image")
async def frame_image(job_id: str, index: int) -> FileResponse:
    job = _require_job(job_id)
    payload = _frames_payload(job)
    frame = next((frame for frame in payload.get("frames", []) if frame["index"] == index), None)
    if not frame:
        raise HTTPException(status_code=404, detail="Frame not found")
    frames_dir = (_job_dir(job) / "frames").resolve()
    path = (frames_dir / frame["file"]).resolve()
    if frames_dir not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail="Frame image not found")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/jobs/{job_id}/diagrams/{name}")
async def diagram_image(job_id: str, name: str) -> FileResponse:
    job = _require_job(job_id)
    safe_name = Path(name).name
    if not safe_name or safe_name != name:
        raise HTTPException(status_code=404, detail="Diagram not found")
    diagrams_dir = (_job_dir(job) / "diagrams").resolve()
    path = (diagrams_dir / safe_name).resolve()
    if diagrams_dir not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail="Diagram not found")
    return FileResponse(path, media_type="image/png" if path.suffix.lower() == ".png" else "image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/jobs/{job_id}/generated-diagrams/{name}")
async def generated_diagram_image(job_id: str, name: str) -> FileResponse:
    job = _require_job(job_id)
    safe_name = Path(name).name
    directory = (_job_dir(job) / "generated-diagrams").resolve()
    path = (directory / safe_name).resolve()
    if safe_name != name or directory not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail="Generated diagram not found")
    return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/jobs/{job_id}/observations")
async def get_observations(job_id: str) -> dict[str, Any]:
    job = _require_job(job_id)
    payload = _frames_payload(job)
    observations = _observations(job)
    merged = []
    for frame in payload.get("frames", []):
        item = {"index": frame["index"], "time": frame["time"], "scene": frame.get("scene")}
        item.update(observations.get(frame["index"], {}))
        merged.append(item)
    return {"job_id": job_id, "observations": merged}


@app.get("/api/jobs/{job_id}/timeline")
async def get_timeline(job_id: str) -> dict[str, Any]:
    job = _require_job(job_id)
    payload = _frames_payload(job)
    observations = _observations(job)
    sample_fps = float(payload.get("sample_fps") or 0)
    interval = 1.0 / sample_fps if sample_fps > 0 else 1.0
    duration = float((payload.get("video") or {}).get("duration") or 0)
    scenes: dict[int, dict[str, Any]] = {}
    for frame in payload.get("frames", []):
        scene = frame.get("scene", 0)
        observation = observations.get(frame["index"], {})
        entry = scenes.setdefault(
            scene,
            {
                "scene": scene,
                "start": frame["time"],
                "end": frame["time"],
                "frames": [],
                "title": "",
                "screen_text": [],
            },
        )
        scene_end = frame["time"] + interval
        if duration:
            scene_end = min(duration, scene_end)
        entry["end"] = max(entry["end"], scene_end)
        entry["frames"].append(
            {
                "index": frame["index"],
                "time": frame["time"],
                "time_label": fmt_ts(frame["time"]),
                "file": frame.get("file"),
                "duplicate": frame.get("duplicate", False),
                "status": observation.get("status", "pending"),
                "description": observation.get("description") or observation.get("summary") or "",
                "screen_text": observation.get("screen_text") or [],
                "has_diagram": observation.get("has_diagram") or bool(observation.get("diagram")),
                "diagram": observation.get("diagram"),
            }
        )
        if not entry["title"]:
            entry["title"] = observation.get("scene") or observation.get("summary") or ""
        for text in observation.get("screen_text") or []:
            if text not in entry["screen_text"]:
                entry["screen_text"].append(text)

    groups = []
    for entry in scenes.values():
        entry["start_label"] = fmt_ts(entry["start"])
        entry["end_label"] = fmt_ts(entry["end"])
        groups.append(entry)
    return {
        "job_id": job_id,
        "sample_fps": payload.get("sample_fps", 0),
        "scenes": groups,
    }


@app.get("/api/jobs/{job_id}/live-notes")
async def get_live_notes(job_id: str) -> PlainTextResponse:
    job = _require_job(job_id)
    path = _job_dir(job) / "live-notes.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Live notes not ready")
    return PlainTextResponse(
        path.read_text(encoding="utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/jobs/{job_id}/note")
async def get_note(job_id: str, format: Literal["md", "json"] = "md") -> Any:
    job = _require_job(job_id)
    path = _job_dir(job) / ("note.md" if format == "md" else "note.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Note not ready")
    if format == "md":
        return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


_episode_tasks: dict[str, asyncio.Task] = {}
_episode_errors: dict[str, str] = {}


@app.post("/api/jobs/{job_id}/teaching-episodes", status_code=202)
async def generate_teaching_episodes(job_id: str) -> dict:
    from .persona_semantic import extract_semantic
    from .vlm import VLMClient

    job = _require_job(job_id)
    folder = _job_dir(job)
    if (folder / "persona" / "semantic-episodes.jsonl").is_file():
        return {"status": "ready"}
    if job_id in _episode_tasks:
        return {"status": "running"}
    if not all((folder / name).is_file() for name in ("transcript.json", "observations.jsonl")):
        raise HTTPException(status_code=409, detail="Saved transcript and frame observations are required.")
    config = job.get("config") or {}
    async def generate():
        try:
            async with VLMClient(VLMConfig(model=config.get("model") or settings.vlm_model,
                                           provider=config.get("provider") or settings.vlm_provider)) as client:
                await extract_semantic(folder, f"lecture-{job_id}", client)
        except Exception as exc:
            _episode_errors[job_id] = str(exc)
        finally:
            _episode_tasks.pop(job_id, None)

    _episode_errors.pop(job_id, None)
    _episode_tasks[job_id] = asyncio.create_task(generate())
    return {"status": "running"}


@app.get("/api/jobs/{job_id}/teaching-episodes")
def teaching_episodes(job_id: str) -> dict:
    folder = _job_dir(_require_job(job_id)) / "persona"
    for name in ("semantic-episodes.jsonl", "episodes.jsonl"):
        path = folder / name
        if path.is_file():
            with path.open(encoding="utf-8") as stream:
                return {"available": True, "episodes": [json.loads(line) for line in stream if line.strip()]}
    return {"available": False, "episodes": [], "running": job_id in _episode_tasks, "error": _episode_errors.get(job_id)}


@app.get("/api/jobs/{job_id}/source-video")
def source_video(job_id: str) -> FileResponse:
    directory = _job_dir(_require_job(job_id))
    for path in directory.glob("source.*"):
        if path.suffix.lower() in ALLOWED_EXTENSIONS and path.is_file():
            return FileResponse(path)
    raise HTTPException(status_code=404, detail="Source video unavailable")


@app.get("/api/jobs/{job_id}/transcript")
async def get_transcript(job_id: str) -> JSONResponse:
    job = _require_job(job_id)
    path = _job_dir(job) / "transcript.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Transcript not ready")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.post("/api/jobs/{job_id}/export/pdf")
async def render_pdf(job_id: str, include_transcript_appendix: bool = False) -> dict[str, Any]:
    """Render on demand so a transcript appendix is opt-in for this export only."""
    job = _require_job(job_id)
    note_path = _job_dir(job) / "note.json"
    if not note_path.exists():
        raise HTTPException(status_code=409, detail="Study notes are not ready")
    note = json.loads(note_path.read_text(encoding="utf-8"))
    transcript_path = _job_dir(job) / "transcript.json"
    if include_transcript_appendix and transcript_path.exists():
        note["_transcript_segments"] = json.loads(transcript_path.read_text(encoding="utf-8")).get("segments") or []
    try:
        await asyncio.to_thread(export_pdf, note, _job_dir(job), include_transcript_appendix)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF export failed: {exc}") from exc
    return {"ok": True, "download": f"/api/jobs/{job_id}/download?kind=study-pdf", "include_transcript_appendix": include_transcript_appendix}


@app.get("/api/jobs/{job_id}/download")
async def download(
    job_id: str,
    kind: Literal["note-md", "note-json", "observations", "frames", "diagrams", "study-pdf", "transcript", "generated-diagrams"] = "note-md",
) -> FileResponse:
    job = _require_job(job_id)
    stem = (job.get("note_title") or Path(job["filename"]).stem or "vidframe").replace("/", "-")
    if kind in {"diagrams", "generated-diagrams"}:
        diagrams_dir = _job_dir(job) / ("generated-diagrams" if kind == "generated-diagrams" else "diagrams")
        images = sorted(path for path in diagrams_dir.iterdir() if path.is_file()) if diagrams_dir.exists() else []
        if not images:
            raise HTTPException(status_code=404, detail="No diagrams were detected")
        archive = _job_dir(job) / "diagrams.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for image in images:
                bundle.write(image, arcname=f"{diagrams_dir.name}/{image.name}")
        return FileResponse(archive, media_type="application/zip", filename=f"{stem}-diagrams.zip")

    mapping = {
        "note-md": ("note.md", "text/markdown"),
        "note-json": ("note.json", "application/json"),
        "observations": ("observations.jsonl", "application/x-ndjson"),
        "frames": ("frames.json", "application/json"),
        "study-pdf": ("pdf/study-notes.pdf", "application/pdf"),
        "transcript": ("transcript.json", "application/json"),
    }
    filename, media_type = mapping[kind]
    path = _job_dir(job) / filename
    if kind == "study-pdf" and not path.exists():
        path = _job_dir(job) / "pdf" / "video_notes.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not ready")
    suffix = {"note-md": "md", "note-json": "json", "observations": "jsonl", "frames": "json", "study-pdf": "pdf", "transcript": "json"}[kind]
    download_name = f"{Path(job['filename']).stem}_notes.pdf" if kind == "study-pdf" else f"{stem}.{suffix}"
    return FileResponse(path, media_type=media_type, filename=download_name)


@app.get("/api/jobs/{job_id}/search")
async def search(job_id: str, q: str = Query(min_length=1, max_length=200)) -> dict[str, Any]:
    job = _require_job(job_id)
    payload = _frames_payload(job)
    frame_map = {frame["index"]: frame for frame in payload.get("frames", [])}
    observations = _observations(job)
    needle = q.lower()
    matches = []
    for index, observation in observations.items():
        haystack = " ".join(
            str(value)
            for value in [
                observation.get("scene"),
                observation.get("summary"),
                observation.get("description"),
                observation.get("setting"),
                observation.get("visual_notes"),
                " ".join(observation.get("screen_text") or []),
                " ".join(observation.get("objects") or []),
                " ".join(observation.get("people") or []),
            ]
        )
        if needle in haystack.lower():
            frame = frame_map.get(index, {})
            matches.append(
                {
                    "index": index,
                    "time": fmt_ts(observation.get("time") or frame.get("time")),
                    "time_seconds": observation.get("time") or frame.get("time"),
                    "scene": observation.get("scene"),
                    "description": observation.get("description"),
                    "screen_text": observation.get("screen_text") or [],
                }
            )
    matches.sort(key=lambda item: item["time_seconds"] or 0)
    return {"query": q, "count": len(matches), "matches": matches[:100]}


# Static frontend
app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(settings.static_dir / "index.html")


@app.get("/favicon.svg", include_in_schema=False)
async def favicon() -> FileResponse:
    return FileResponse(settings.static_dir / "favicon.svg", media_type="image/svg+xml")
