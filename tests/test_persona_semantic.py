import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from app.persona_semantic import extract_semantic, records


class SemanticExtractionTests(unittest.TestCase):
    def test_joint_episode_and_cached_rerun(self):
        with TemporaryDirectory() as directory:
            job = Path(directory)
            speech = [{"id": i, "start": t, "end": t+1, "text": text}
                      for i, (t, text) in enumerate([(1, "Two paths."), (5, "Trace this branch."), (9, "They meet here.")])]
            (job / "transcript.json").write_text(json.dumps({"segments": speech}))
            frame = {"index": 8, "time": 6, "status": "ok", "actions": ["Points to the left branch."]}
            (job / "observations.jsonl").write_text(json.dumps(frame))
            item = {"move": "trace_path", "method": "Pairs the instruction with pointing.", "refs": ["f8", "s1"]}
            client = SimpleNamespace(config=SimpleNamespace(provider="local", base_url="local", model="test"),
                                     complete=AsyncMock(return_value=json.dumps({"episodes": [item]})))
            count, path = asyncio.run(extract_semantic(job, "teacher", client))
            self.assertEqual(count, 1)
            row = json.loads(path.read_text())
            self.assertEqual(row["context"]["speech"], [speech[0]])
            self.assertEqual(row["target"]["speech"], [speech[1]])
            self.assertEqual(row["target"]["visuals"][0]["actions"], frame["actions"])
            self.assertEqual(row["followup"]["speech"], [speech[2]])
            asyncio.run(extract_semantic(job, "teacher", client))
            client.complete.assert_awaited_once()

    def test_rejects_unknown_source(self):
        with self.assertRaises(ValueError):
            list(records({"episodes": [{"refs": ["f99"]}]}, {}, 0, "lecture", "teacher", "model"))
