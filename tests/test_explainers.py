import asyncio
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx

from app import db, explainers
from app.config import settings
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app


class ExplainerTests(unittest.TestCase):
    def test_topic_match_and_unsupported(self):
        units = [{"id": "c0s0", "chapter": "Calculus", "title": "Chain rule", "text": "Differentiate a composition."}]
        client = SimpleNamespace(complete=AsyncMock(return_value='{"ids":["c0s0"]}'))
        selected = asyncio.run(explainers.match_topic("chain rule", units, client))
        self.assertEqual(selected, units)
        client.complete.return_value = '{"ids":[]}'
        with self.assertRaisesRegex(ValueError, "not covered"):
            asyncio.run(explainers.match_topic("unrelated", units, client))

    def test_reference_and_scene_validation(self):
        segments = [{"start": i * 3 + 2, "end": i * 3 + 4.8, "text": "Here is the next useful part of the explanation"} for i in range(8)]
        start, end = explainers.choose_reference(segments, 40)
        self.assertGreaterEqual(end - start, 8)
        self.assertLessEqual(end - start, 22)
        narration = " ".join(["This idea follows directly from the source notes"] * 11)
        scenes = [{"kind": "labels", "title": "Source idea", "narration": narration, "source_ids": ["c0s0"], "items": ["Start", "Result"]} for _ in range(8)]
        plan = explainers.validate_plan({"scenes": scenes}, {"c0s0"})
        self.assertEqual(len(plan), 8)
        scenes[0]["source_ids"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "unsupported source"):
            explainers.validate_plan({"scenes": scenes}, {"c0s0"})

    def test_cloning_capability_and_request_shape(self):
        seen = []
        def handler(request):
            seen.append(request)
            if request.method == "GET":
                return httpx.Response(200, json={"data": {"endpoints": [{"supports_voice_cloning": True, "status": 0}]}})
            return httpx.Response(200, content=b"a" * 2048, headers={"content-type": "audio/mpeg"})
        original_base = explainers.settings.openrouter_base_url
        original_key = explainers.settings.openrouter_api_key
        explainers.settings.openrouter_base_url = "https://openrouter.test/api/v1"
        explainers.settings.openrouter_api_key = "test-key"
        try:
            async def check():
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                    await explainers.ensure_cloning(client)
                    with tempfile.TemporaryDirectory() as temp:
                        path = Path(temp) / "speech.mp3"
                        await explainers.synthesize(client, "A short lesson.", b"reference", "The teacher's example.", path)
                        self.assertEqual(path.stat().st_size, 2048)
            asyncio.run(check())
            payload = json.loads(seen[-1].content)
            self.assertEqual(payload["model"], explainers.TTS_MODEL)
            self.assertEqual(payload["input_references"][0]["type"], "input_audio")
            self.assertEqual(payload["input_references"][1], {"type": "text", "text": "The teacher's example."})
            self.assertIn("Indian English", payload["instructions"])
            self.assertNotIn("voice", payload)
        finally:
            explainers.settings.openrouter_base_url = original_base
            explainers.settings.openrouter_api_key = original_key

    def test_cloning_unavailable(self):
        def handler(request):
            return httpx.Response(200, json={"data": {"endpoints": [{"supports_voice_cloning": False, "status": 0}]}})
        async def check():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                with self.assertRaisesRegex(RuntimeError, "no voice-cloning endpoint"):
                    await explainers.ensure_cloning(client)
        asyncio.run(check())

    def test_captions_are_short_and_follow_scene_timing(self):
        narration = "First, follow the current through the circuit. Then compare the two paths carefully."
        chunks = explainers.caption_chunks(narration)
        self.assertEqual(" ".join(chunks), narration)
        self.assertTrue(all(len(x) <= 42 and len(x.split()) <= 7 for x in chunks))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "captions.ass"
            explainers.write_ass_captions([{"narration": narration}, {"narration": "The paths match."}], [6.0, 2.0], path)
            ass = path.read_text()
            dialogues = [x for x in ass.splitlines() if x.startswith("Dialogue:")]
            self.assertEqual(len(dialogues), 2 * (len(chunks) + 1))
            self.assertIn("0:00:06.28", dialogues[-1])
            self.assertTrue(all("\\N" not in x for x in dialogues))
            self.assertTrue(any("\\p1" in x for x in dialogues))
            self.assertIn("Style: Captions,DejaVu Sans,30,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,0,0,2,90,90,28,1", ass)
            self.assertIn(r"\1c&H3A3125&\1a&H50&\bord0\shad0", ass)
            self.assertTrue(all(dialogues[i].startswith("Dialogue: 0,") and dialogues[i + 1].startswith("Dialogue: 1,")
                                for i in range(0, len(dialogues), 2)))

    def test_persistence_and_restart_recovery_state(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(settings, "data_dir", Path(temp)):
            db.init_db()
            job = db.create_job("lecture.mp4", {}, str(Path(temp) / "jobs" / "lecture"))
            record = db.create_explainer(job["id"], "chain rule", "abc123", explainers.TTS_MODEL, str(Path(job["dir"]) / "explainers"))
            self.assertEqual(len(db.list_explainers(job["id"])), 1)
            db.update_explainer(job["id"], record["id"], status="running", progress=42)
            db.mark_explainers_interrupted()
            self.assertEqual(db.get_explainer(job["id"], record["id"])["status"], "interrupted")
            db.delete_job(job["id"])
            self.assertEqual(db.list_explainers(job["id"]), [])

    def test_api_history_and_parent_deletion(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(settings, "data_dir", Path(temp)), \
                patch.object(settings, "openrouter_api_key", "test-key"), \
                patch.object(explainers, "manim_command", return_value="manim"), patch.object(explainers, "start"):
            folder = Path(temp) / "jobs" / "lecture"
            folder.mkdir(parents=True)
            (folder / "note.json").write_text(json.dumps({"chapters": [{"title": "Calculus", "subtopics": [
                {"title": "Chain rule", "explanation": ["Differentiate a composition."]}]}]}))
            (folder / "transcript.json").write_text(json.dumps({"segments": [{"start": 2, "end": 13, "text": "An explanation of the chain rule."}]}))
            (folder / "audio.wav").write_bytes(b"audio")
            with TestClient(app) as client:
                job = db.create_job("lecture.mp4", {}, str(folder))
                db.update_job(job["id"], status="done")
                response = client.post(f"/api/jobs/{job['id']}/explainers", json={"topic": "chain rule"})
                self.assertEqual(response.status_code, 202, response.text)
                eid = response.json()["id"]
                history = client.get(f"/api/jobs/{job['id']}/explainers").json()
                self.assertEqual([item["id"] for item in history["items"]], [eid])
                self.assertEqual(client.get(f"/api/jobs/{job['id']}/explainers/{eid}/video").status_code, 404)
                self.assertEqual(client.delete(f"/api/jobs/{job['id']}").status_code, 200)
                self.assertEqual(client.get(f"/api/jobs/{job['id']}/explainers").status_code, 404)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_manim_render_and_mux(self):
        if not explainers.manim_command():
            self.skipTest("Manim required")
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            audio = folder / "speech-00.mp3"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                            "-c:a", "libmp3lame", str(audio)], check=True)
            plan = folder / "plan.json"
            plan.write_text(json.dumps({"scenes": [{"kind": "diagram", "title": "A relationship",
                                                    "items": ["Input", "Rule", "Result"],
                                                    "narration": "Input and rule make a result.", "duration": 3}]}))
            duration = explainers.render(plan, folder, [3], [audio])
            self.assertAlmostEqual(duration, 3, delta=0.2)
            output = folder / "explainer.mp4"
            explainers.write_ass_captions(json.loads(plan.read_text())["scenes"], [3], folder / "captions.ass")
            explainers.mux_captioned(folder, output)
            self.assertAlmostEqual(explainers.audio_duration(output), 3, delta=0.3)
