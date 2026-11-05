from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

from .config import settings


def _build_config(args: argparse.Namespace, source: Path) -> dict:
    from .pipeline import normalize_config

    return normalize_config(
        {},
        {
            "sample_fps": args.fps,
            "max_frames": args.max_frames,
            "batch_size": args.batch_size,
            "provider": args.provider,
            "model": args.model,
            "compile_model": args.compile_model,
            "request_timeout": args.timeout,
            "original_filename": source.name,
            "source_file": f"source{source.suffix.lower()}",
        },
    )


def cmd_analyze(args: argparse.Namespace) -> int:
    from . import db, pipeline

    if args.mock:
        settings.vlm_mode = "mock"
    source = Path(args.video).expanduser().resolve()
    if not source.exists() or not source.is_file():
        print(f"error: video not found: {source}", file=sys.stderr)
        return 2
    if source.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpeg", ".mpg", ".ts"}:
        print(f"error: unsupported video extension: {source.suffix}", file=sys.stderr)
        return 2

    db.init_db()
    config = _build_config(args, source)
    job = db.create_job(source.name, config, "")
    job_dir = settings.jobs_dir / job["id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, job_dir / f"source{source.suffix.lower()}")
    db.update_job(job["id"], dir=str(job_dir), config_json=json.dumps(config), model=config["model"])
    db.add_event(job["id"], "cli", f"Queued {source.name}")

    async def runner() -> None:
        task = asyncio.create_task(pipeline.run_job(job["id"], "full"))
        while not task.done():
            current = db.get_job(job["id"])
            if current:
                print(
                    f"\r[{current['stage']:<10}] {current['progress']:5.1f}%  {current['message'][:80]:<80}",
                    end="",
                    flush=True,
                )
            await asyncio.sleep(1.0)
        await task
        print()

    asyncio.run(runner())

    final = db.get_job(job["id"])
    if not final or final["status"] != "done":
        print(f"error: {final['error'] if final else 'job disappeared'}", file=sys.stderr)
        return 1

    note_path = job_dir / "note.md"
    if args.out:
        if args.out == "-":
            print(note_path.read_text(encoding="utf-8"))
        else:
            target = Path(args.out).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(note_path, target)
            print(f"note: {target}")
    if args.json:
        print(json.dumps(json.loads((job_dir / "note.json").read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
    if not args.out and not args.json:
        print(f"note: {note_path}")
    print(f"job:  {job['id']}  ({time.strftime('%H:%M:%S')})")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    script = settings.static_dir.parent / "scripts" / "make_demo_video.py"
    print(f"Demo video generator: {script}")
    print("Run: .venv/bin/python scripts/make_demo_video.py --out demo.mp4")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vidframe", description="Visual-only video analysis with a VLM")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Analyze a video and write a visual report")
    analyze.add_argument("video")
    analyze.add_argument("--fps", type=float, default=settings.default_fps, help="Frames per second to sample")
    analyze.add_argument("--max-frames", type=int, default=settings.default_max_frames)
    analyze.add_argument("--batch-size", type=int, default=settings.default_batch_size)
    analyze.add_argument("--provider", choices=["local", "openrouter"], default=settings.vlm_provider)
    analyze.add_argument("--model", default=None)
    analyze.add_argument("--compile-model", default=None)
    analyze.add_argument("--timeout", type=float, default=settings.request_timeout)
    analyze.add_argument("--out", help="Write markdown here ('-' for stdout)")
    analyze.add_argument("--json", action="store_true", help="Print the structured note as JSON")
    analyze.add_argument("--mock", action="store_true", help="Use the deterministic mock VLM")
    analyze.set_defaults(func=cmd_analyze)

    serve = sub.add_parser("serve", help="Run the web app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    demo = sub.add_parser("demo", help="Show how to generate a demo video")
    demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
