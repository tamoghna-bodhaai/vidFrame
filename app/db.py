from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import settings
from .utils import now_iso

JOB_FIELDS = {
    "filename",
    "title",
    "status",
    "stage",
    "progress",
    "message",
    "error",
    "duration",
    "width",
    "height",
    "source_fps",
    "sample_fps",
    "frame_count",
    "analyzed_count",
    "failed_count",
    "scene_count",
    "diagram_count",
    "model",
    "config_json",
    "dir",
    "note_title",
    "started_at",
    "finished_at",
    "elapsed",
    "resume_stage",
}

ACTIVE_STATUSES = ("queued", "running", "compiling")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    con = sqlite3.connect(settings.db_path, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    settings.ensure_dirs()
    with connect() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                title TEXT,
                status TEXT NOT NULL DEFAULT 'queued',
                stage TEXT NOT NULL DEFAULT 'queued',
                progress REAL NOT NULL DEFAULT 0,
                message TEXT NOT NULL DEFAULT '',
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                duration REAL,
                width INTEGER,
                height INTEGER,
                source_fps REAL,
                sample_fps REAL,
                frame_count INTEGER NOT NULL DEFAULT 0,
                analyzed_count INTEGER NOT NULL DEFAULT 0,
                failed_count INTEGER NOT NULL DEFAULT 0,
                scene_count INTEGER NOT NULL DEFAULT 0,
                diagram_count INTEGER NOT NULL DEFAULT 0,
                model TEXT,
                note_title TEXT,
                elapsed REAL,
                resume_stage TEXT,
                config_json TEXT NOT NULL DEFAULT '{}',
                dir TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                stage TEXT NOT NULL DEFAULT '',
                level TEXT NOT NULL DEFAULT 'info',
                message TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_events_job ON events(job_id, id);
            CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
            CREATE TABLE IF NOT EXISTS model_costs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                stage TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                cost_usd REAL,
                audio_seconds REAL
            );
            CREATE INDEX IF NOT EXISTS idx_model_costs_job ON model_costs(job_id, id);
            CREATE TABLE IF NOT EXISTS explainers (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                status TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0,
                message TEXT NOT NULL DEFAULT '',
                error TEXT,
                note_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                finished_at TEXT,
                duration REAL,
                model TEXT NOT NULL,
                output_dir TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_explainers_job ON explainers(job_id, created_at DESC);
            """
        )
        columns = {row["name"] for row in con.execute("PRAGMA table_info(jobs)").fetchall()}
        if "diagram_count" not in columns:
            con.execute("ALTER TABLE jobs ADD COLUMN diagram_count INTEGER NOT NULL DEFAULT 0")
        if "resume_stage" not in columns:
            con.execute("ALTER TABLE jobs ADD COLUMN resume_stage TEXT")
        cost_columns = {row["name"] for row in con.execute("PRAGMA table_info(model_costs)").fetchall()}
        if "audio_seconds" not in cost_columns:
            con.execute("ALTER TABLE model_costs ADD COLUMN audio_seconds REAL")


def _row_to_job(row: sqlite3.Row) -> dict[str, Any]:
    job = dict(row)
    try:
        job["config"] = json.loads(job.pop("config_json") or "{}")
    except json.JSONDecodeError:
        job["config"] = {}
    return job


def create_job(filename: str, config: dict[str, Any], job_dir: str) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:14]
    timestamp = now_iso()
    with connect() as con:
        con.execute(
            """
            INSERT INTO jobs (id, filename, title, status, stage, progress, message,
                              created_at, updated_at, model, config_json, dir)
            VALUES (?, ?, ?, 'queued', 'queued', 0, 'Waiting to start', ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                filename,
                filename.rsplit(".", 1)[0],
                timestamp,
                timestamp,
                config.get("model") or settings.vlm_model,
                json.dumps(config, ensure_ascii=False),
                job_dir,
            ),
        )
    return get_job(job_id)  # type: ignore[return-value]


def get_job(job_id: str) -> dict[str, Any] | None:
    with connect() as con:
        row = con.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_job(row) if row else None


def list_jobs(limit: int = 100) -> list[dict[str, Any]]:
    with connect() as con:
        rows = con.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 500)),)
        ).fetchall()
    return [_row_to_job(row) for row in rows]


def update_job(job_id: str, **fields: Any) -> None:
    clean = {k: v for k, v in fields.items() if k in JOB_FIELDS}
    if not clean:
        return
    clean["updated_at"] = now_iso()
    assignments = ", ".join(f"{key} = ?" for key in clean)
    values = list(clean.values()) + [job_id]
    with connect() as con:
        con.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", values)


def delete_job(job_id: str) -> bool:
    with connect() as con:
        con.execute("DELETE FROM explainers WHERE job_id = ?", (job_id,))
        con.execute("DELETE FROM events WHERE job_id = ?", (job_id,))
        con.execute("DELETE FROM model_costs WHERE job_id = ?", (job_id,))
        cursor = con.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    return cursor.rowcount > 0


def create_explainer(job_id: str, topic: str, note_sha256: str, model: str, output_parent: str) -> dict[str, Any]:
    explainer_id = uuid.uuid4().hex[:14]
    timestamp = now_iso()
    with connect() as con:
        con.execute(
            "INSERT INTO explainers (id, job_id, topic, status, message, note_sha256, created_at, updated_at, model, output_dir) "
            "VALUES (?, ?, ?, 'queued', 'Waiting to start', ?, ?, ?, ?, ?)",
            (explainer_id, job_id, topic, note_sha256, timestamp, timestamp, model, str(Path(output_parent) / explainer_id)),
        )
    return get_explainer(job_id, explainer_id)  # type: ignore[return-value]


def get_explainer(job_id: str, explainer_id: str) -> dict[str, Any] | None:
    with connect() as con:
        row = con.execute("SELECT * FROM explainers WHERE job_id = ? AND id = ?", (job_id, explainer_id)).fetchone()
    return dict(row) if row else None


def list_explainers(job_id: str) -> list[dict[str, Any]]:
    with connect() as con:
        rows = con.execute("SELECT * FROM explainers WHERE job_id = ? ORDER BY created_at DESC", (job_id,)).fetchall()
    return [dict(row) for row in rows]


def update_explainer(job_id: str, explainer_id: str, **fields: Any) -> None:
    allowed = {"status", "progress", "message", "error", "finished_at", "duration"}
    values = {key: value for key, value in fields.items() if key in allowed}
    if not values:
        return
    values["updated_at"] = now_iso()
    with connect() as con:
        con.execute(
            "UPDATE explainers SET " + ", ".join(f"{key} = ?" for key in values) + " WHERE job_id = ? AND id = ?",
            [*values.values(), job_id, explainer_id],
        )


def mark_explainers_interrupted() -> None:
    with connect() as con:
        con.execute(
            "UPDATE explainers SET status = 'interrupted', message = 'Server restarted during generation', updated_at = ? "
            "WHERE status IN ('queued', 'running')", (now_iso(),)
        )


def interrupted_explainers() -> list[dict[str, Any]]:
    with connect() as con:
        rows = con.execute("SELECT * FROM explainers WHERE status = 'interrupted' ORDER BY created_at").fetchall()
    return [dict(row) for row in rows]


def add_model_cost(job_id: str, stage: str, provider: str, model: str,
                   prompt_tokens: int | None, completion_tokens: int | None,
                   cost_usd: float | None, audio_seconds: float | None = None) -> None:
    with connect() as con:
        con.execute(
            "INSERT INTO model_costs (job_id, ts, stage, provider, model, prompt_tokens, completion_tokens, cost_usd, audio_seconds) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (job_id, now_iso(), stage, provider, model, prompt_tokens, completion_tokens, cost_usd, audio_seconds),
        )


def get_model_costs(job_id: str) -> list[dict[str, Any]]:
    with connect() as con:
        rows = con.execute("SELECT * FROM model_costs WHERE job_id = ? ORDER BY id", (job_id,)).fetchall()
    return [dict(row) for row in rows]


def add_event(job_id: str, stage: str, message: str, level: str = "info") -> None:
    with connect() as con:
        con.execute(
            "INSERT INTO events (job_id, ts, stage, level, message) VALUES (?, ?, ?, ?, ?)",
            (job_id, now_iso(), stage, level, message),
        )


def get_events(job_id: str, after_id: int = 0, limit: int = 400) -> list[dict[str, Any]]:
    with connect() as con:
        rows = con.execute(
            "SELECT id, ts, stage, level, message FROM events "
            "WHERE job_id = ? AND id > ? ORDER BY id LIMIT ?",
            (job_id, max(0, after_id), max(1, min(limit, 2000))),
        ).fetchall()
    return [dict(row) for row in rows]


def mark_interrupted() -> int:
    with connect() as con:
        placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
        cursor = con.execute(
            f"UPDATE jobs SET status = 'interrupted', resume_stage = stage, stage = 'interrupted', "
            f"message = 'Server restarted before the job finished', updated_at = ? "
            f"WHERE status IN ({placeholders})",
            (now_iso(), *ACTIVE_STATUSES),
        )
    return cursor.rowcount
