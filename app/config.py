from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env(path: Path | None = None) -> None:
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value is not None and value != "" else default


def _first_str(names: tuple[str, ...], default: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    return _str(name, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_str("VIDFRAME_DATA_DIR", str(ROOT / "data"))))
    vlm_base_url: str = field(default_factory=lambda: _str("VIDFRAME_VLM_BASE_URL", "http://localhost:8000/v1"))
    vlm_model: str = field(default_factory=lambda: _str("VIDFRAME_VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"))
    vlm_api_key: str = field(default_factory=lambda: _str("VIDFRAME_VLM_API_KEY", "EMPTY"))
    vlm_mode: str = field(default_factory=lambda: _str("VIDFRAME_VLM_MODE", "openai").lower())
    vlm_provider: str = field(default_factory=lambda: _str("VIDFRAME_VLM_PROVIDER", "local").lower())
    compile_model: str = field(default_factory=lambda: _str("VIDFRAME_COMPILE_MODEL", ""))
    stt_base_url: str = field(default_factory=lambda: _str("VIDFRAME_STT_BASE_URL", "http://localhost:8001/v1"))
    stt_api_key: str = field(default_factory=lambda: _str("VIDFRAME_STT_API_KEY", "EMPTY"))
    stt_model: str = field(default_factory=lambda: _str("VIDFRAME_STT_MODEL", "whisper-1"))
    stt_timeout: float = field(default_factory=lambda: _float("VIDFRAME_STT_TIMEOUT", 600.0))
    stt_provider: str = field(default_factory=lambda: _str("VIDFRAME_STT_PROVIDER", "local").lower())
    stt_language: str = field(default_factory=lambda: _str("VIDFRAME_STT_LANGUAGE", "auto"))
    stt_chunk_seconds: int = field(default_factory=lambda: max(30, _int("VIDFRAME_STT_CHUNK_SECONDS", 60)))
    openrouter_base_url: str = field(
        default_factory=lambda: _str("VIDFRAME_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    )
    openrouter_api_key: str = field(
        default_factory=lambda: _first_str(("VIDFRAME_OPENROUTER_API_KEY", "OPENROUTER_API_KEY"), "")
    )
    openrouter_model: str = field(
        default_factory=lambda: _str("VIDFRAME_OPENROUTER_MODEL", "qwen/qwen2.5-vl-72b-instruct")
    )
    openrouter_referer: str = field(
        default_factory=lambda: _str("VIDFRAME_OPENROUTER_REFERER", "http://127.0.0.1:8000")
    )
    request_timeout: float = field(default_factory=lambda: _float("VIDFRAME_REQUEST_TIMEOUT", 180.0))
    max_concurrency: int = field(default_factory=lambda: max(1, _int("VIDFRAME_MAX_CONCURRENCY", 4)))
    max_concurrent_jobs: int = field(default_factory=lambda: max(1, _int("VIDFRAME_MAX_CONCURRENT_JOBS", 3)))
    max_concurrent_requests: int = field(
        default_factory=lambda: max(1, _int("VIDFRAME_MAX_CONCURRENT_REQUESTS", 8))
    )
    default_fps: float = field(default_factory=lambda: _float("VIDFRAME_FPS", 1.0))
    default_max_frames: int = field(default_factory=lambda: _int("VIDFRAME_MAX_FRAMES", 900))
    default_batch_size: int = field(default_factory=lambda: _int("VIDFRAME_BATCH_SIZE", 4))
    max_image_side: int = field(default_factory=lambda: _int("VIDFRAME_MAX_IMAGE_SIDE", 768))
    jpeg_quality: int = field(default_factory=lambda: _int("VIDFRAME_JPEG_QUALITY", 82))
    max_upload_mb: int = field(default_factory=lambda: _int("VIDFRAME_MAX_UPLOAD_MB", 4096))
    keep_source: bool = field(default_factory=lambda: _bool("VIDFRAME_KEEP_SOURCE", True))
    compile_chunk_chars: int = field(default_factory=lambda: _int("VIDFRAME_COMPILE_CHUNK_CHARS", 12000))

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "vidframe.db"

    @property
    def static_dir(self) -> Path:
        return ROOT / "static"

    @property
    def compile_model_name(self) -> str:
        return self.compile_model or self.vlm_model

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def openrouter_configured(self) -> bool:
        return bool(self.openrouter_api_key)

    def public(self) -> dict:
        return {
            "vlm_base_url": self.vlm_base_url,
            "vlm_model": self.vlm_model,
            "vlm_mode": self.vlm_mode,
            "default_provider": self.vlm_provider,
            "openrouter_base_url": self.openrouter_base_url,
            "openrouter_model": self.openrouter_model,
            "openrouter_configured": self.openrouter_configured,
            "stt_base_url": self.stt_base_url,
            "stt_model": self.stt_model,
            "stt_timeout": self.stt_timeout,
            "stt_provider": self.stt_provider,
            "stt_language": self.stt_language,
            "stt_chunk_seconds": self.stt_chunk_seconds,
            "compile_model": self.compile_model_name,
            "default_fps": self.default_fps,
            "default_max_frames": self.default_max_frames,
            "default_batch_size": self.default_batch_size,
            "max_image_side": self.max_image_side,
            "max_upload_mb": self.max_upload_mb,
            "request_timeout": self.request_timeout,
            "max_concurrency": self.max_concurrency,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "max_concurrent_requests": self.max_concurrent_requests,
        }


_load_env()
settings = Settings()
settings.ensure_dirs()
