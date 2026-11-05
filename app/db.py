from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
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
            """
        )
        columns = {row["name"] for row in con.execute("PRAGMA table_info(jobs)").fetchall()}
        if "diagram_count" not in columns:
            con.execute("ALTER TABLE jobs ADD COLUMN diagram_count INTEGER NOT NULL DEFAULT 0")
        if "resume_stage" not in columns:
            con.execute("ALTER TABLE jobs ADD COLUMN resume_stage TEXT")


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
        con.execute("DELETE FROM events WHERE job_id = ?", (job_id,))
        cursor = con.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    return cursor.rowcount > 0


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
