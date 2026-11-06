"""Build reviewable teaching episodes from an existing vidFrame job, offline."""
import argparse
import asyncio
import json
import math
import re
from pathlib import Path
from tempfile import NamedTemporaryFile


# Narrow, inspectable English cues. Keep the matched wording for auditing.
CUES = {
    "invite_prediction": r"\b(?:what (?:do you think|would happen|will happen)|can you predict)\b",
    "introduce_example": r"\b(?:for (?:example|instance)|let(?:'s| us) (?:take|try|consider) (?:a|an|the) example)\b",
    "rephrase": r"\b(?:in other words|let me (?:rephrase|put it another way))\b",
    "invite_recall": r"\b(?:do you remember|can you recall)\b",
    "offer_analogy": r"\b(?:think of (?:it|this) as|you can think of (?:it|this) as)\b",
    "flag_common_error": r"\b(?:a common (?:mistake|error)|you might be tempted to)\b",
}
PATTERNS = {k: re.compile(v, re.I) for k, v in CUES.items()}


def timestamp(value):
    t = float(value)
    if not math.isfinite(t) or t < 0:
        raise ValueError(f"Invalid source timestamp {value}")
    return t


def read_sources(job):
    transcript = json.loads((job / "transcript.json").read_text())
    segments = []
    for i, row in enumerate(transcript["segments"]):
        start, end = timestamp(row["start"]), timestamp(row["end"])
        if end < start:
            raise ValueError(f"Reversed transcript interval at row {i}")
        if row["text"].strip():
            segments.append(dict(id=row.get("id", i), start=start, end=end, text=row["text"]))
    if not segments:
        raise ValueError("No usable timestamped transcript segments")
    frames = []
    with (job / "observations.jsonl").open() as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                frames.append(dict(index=row["index"], time=timestamp(row["time"]),
                                   actions=row.get("actions", []), screen_text=row.get("screen_text", []),
                                   summary=row.get("summary", "")))
    return sorted(segments, key=lambda s: s["start"]), sorted(frames, key=lambda f: f["time"])


def episodes(segments, frames, lecture, teacher):
    for i, row in enumerate(segments):
        moves = []
        for name, pattern in PATTERNS.items():
            match = pattern.search(row["text"])
            if match:
                moves.append({"move": name, "cue": match.group()})
        if not moves:
            continue
        start, end = row["start"], row["end"]
        prior = [s for s in segments[max(0, i-4):i] if start-30 <= s["end"] <= start]
        context_start = prior[0]["start"] if prior else start - 10
        # Subsequent words may belong to a learner. Preserve them as follow-up,
        # never silently assign them to the teacher or leak them into the prompt.
        following = [s for s in segments[i+1:i+4] if end <= s["start"] <= end+15]
        yield {
            "schema_version": 1, "episode_id": f"{lecture}:segment:{i}",
            "lecture_id": lecture, "teacher_id": teacher, "review_status": "candidate",
            "speaker_verified": False,
            "context": {"speech": prior, "visuals": [f for f in frames if context_start <= f["time"] < start]},
            "target": {"speech": row, "moves": moves,
                       "visuals": [f for f in frames if start <= f["time"] < end]},
            "followup": {"speech": following},
        }


def extract(job, teacher):
    segments, frames = read_sources(job)
    folder = job / "persona"
    folder.mkdir(exist_ok=True)
    output = folder / "episodes.jsonl"
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=folder, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            count = 0
            for record in episodes(segments, frames, job.name, teacher):
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            stream.close()
            tmp.replace(output)
        finally:
            tmp.unlink(missing_ok=True)
    return count, output


async def run_semantic(job, teacher, model):
    from .config import settings
    from .persona_semantic import extract_semantic
    from .vlm import VLMClient, VLMConfig
    async with VLMClient(VLMConfig(model=model or settings.compile_model_name)) as client:
        return await extract_semantic(job, teacher, client)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path, help="Existing extraction job directory")
    parser.add_argument("--teacher-id", required=True, help="Stable teacher identifier supplied by the curator")
    parser.add_argument("--semantic", action="store_true", help="Analyze saved evidence with the configured model")
    parser.add_argument("--model", help="Model override for --semantic")
    args = parser.parse_args()
    try:
        job = args.job.resolve()
        if args.semantic:
            count, output = asyncio.run(run_semantic(job, args.teacher_id, args.model))
        else:
            count, output = extract(job, args.teacher_id)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Persona extraction failed: {exc}\n")
    print(f"{count} candidate episodes written to {output}")


if __name__ == "__main__":
    main()
