#!/usr/bin/env python3
"""Run the current multimodal pipeline with its deterministic mock vision client."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(tempfile.mkdtemp(prefix="vidframe-smoke-"))
os.environ["VIDFRAME_DATA_DIR"] = str(DATA_DIR)
os.environ["VIDFRAME_VLM_MODE"] = "mock"
os.environ["VIDFRAME_VLM_PROVIDER"] = "local"
sys.path.insert(0, str(ROOT))


def main() -> None:
    from app import db, pipeline
    from app.prompts import frame_prompt

    assert '"raw": "visible equation exactly"' in frame_prompt([{"index": 0}])
    demo = DATA_DIR / "demo.mp4"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_demo_video.py"), "--out", str(demo)], check=True, stdout=subprocess.DEVNULL)
    db.init_db()
    config = pipeline.normalize_config({}, {"sample_fps": 1, "max_frames": 100, "batch_size": 4,
                                             "model": "mock-vision", "source_file": "source.mp4",
                                             "original_filename": "demo.mp4"})
    job = db.create_job("demo.mp4", config, "")
    job_dir = DATA_DIR / "jobs" / job["id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(demo, job_dir / "source.mp4")
    db.update_job(job["id"], dir=str(job_dir), config_json=json.dumps(config), model=config["model"])

    async def run() -> None:
        for mode in ("full", "compile", "analyze"):
            await pipeline.run_job(job["id"], mode)
            assert db.get_job(job["id"])["status"] == "done"

    asyncio.run(run())
    frames = json.loads((job_dir / "frames.json").read_text())["frames"]
    observations = pipeline._load_observations(job_dir / "observations.jsonl")
    assert frames and observations
    assert len(observations) <= len(frames)  # Duplicate frames reuse prior visual evidence.
    assert (job_dir / "transcript.json").is_file()
    assert (job_dir / "note.md").is_file()
    assert (job_dir / "note.json").is_file()
    assert len(list((job_dir / "versions").glob("*.json"))) >= 3
    for observation in observations.values():
        diagram = observation.get("diagram")
        if diagram:
            assert diagram.get("detail_analyzed")
            assert (job_dir / diagram["image"]).is_file()
    print(f"OK: {len(frames)} frames, {len(observations)} analyzed observations, "
          f"{sum(bool(o.get('diagram')) for o in observations.values())} detailed diagrams, "
          "full/recompile/reanalyze completed")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(DATA_DIR, ignore_errors=True)
