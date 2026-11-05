import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from app import pipeline
from app.prompts import DIAGRAM_INSTRUCTIONS
from app.vlm import VLMClient, VLMConfig


class FakeVisionClient:
    calls = 0

    def __init__(self, config):
        self.config = config

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def analyze_diagram(self, *_args):
        type(self).calls += 1
        return {
            "present": True,
            "type": "schematic",
            "title": "Source circuit",
            "confidence": .95,
            "render_spec": {"confidence": .95, "elements": [
                {"shape": "battery", "x": .1, "y": .5, "x2": .3, "y2": .5},
                {"shape": "resistor", "x": .5, "y": .5, "x2": .8, "y2": .5},
            ]},
        }


class DiagramPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_diagram_pass_can_reject_false_positive(self):
        client = VLMClient(VLMConfig(model="vision", provider="local", timeout=10))
        async def reply(*_args, **_kwargs):
            return '{"present": false, "type": "other"}'
        with patch.object(client, "_chat", reply):
            result = await client.analyze_diagram({"time": 0}, "jpeg", {}, {"type": "schematic"})
        self.assertIsNone(result)
        self.assertIn('"render_spec": {', DIAGRAM_INSTRUCTIONS)

    async def test_crop_geometry_persists_and_recompile_renders_vector(self):
        FakeVisionClient.calls = 0
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp)
            (job_dir / "frames").mkdir()
            cv2.imwrite(str(job_dir / "frames" / "f00000.jpg"), np.full((120, 180, 3), 255, dtype=np.uint8))
            frames = [{"index": 0, "file": "f00000.jpg", "time": 3.0, "scene": 0}]
            observations = {0: {"index": 0, "status": "ok", "has_diagram": True, "diagram": {
                "present": True, "type": "schematic", "title": "Circuit", "confidence": .9,
                "bbox": [0, 0, 1, 1],
            }}}
            job = {"id": "test", "title": "Lecture", "filename": "lecture.mp4"}
            config = {"provider": "local", "model": "vision", "diagram_model": "vision", "request_timeout": 10}
            with patch.object(pipeline, "VLMClient", FakeVisionClient), \
                    patch.object(pipeline, "_write_live_notes"), \
                    patch.object(pipeline.db, "update_job"), \
                    patch.object(pipeline.db, "add_event"):
                count = await pipeline._analyze_diagrams(job, job_dir, frames, observations, config, 0)
                self.assertEqual(count, 1)
                self.assertEqual(FakeVisionClient.calls, 1)
                saved = pipeline._load_observations(job_dir / "observations.jsonl")
                self.assertTrue(saved[0]["diagram"]["detail_analyzed"])
                self.assertIn("render_spec", saved[0]["diagram"])
                await pipeline._analyze_diagrams(job, job_dir, frames, saved, config, 0, resume=True)
                self.assertEqual(FakeVisionClient.calls, 1)
                note = {"diagrams": [saved[0]["diagram"]]}
                await pipeline._generate_diagrams("test", job_dir, note, config)
                self.assertTrue((job_dir / note["diagrams"][0]["generated_image"]).is_file())


if __name__ == "__main__":
    unittest.main()
