import tempfile
import unittest
from pathlib import Path

from app.study import build_evidence, compile_study, evidence_chunks, export_pdf, study_markdown


class StudyNotesTests(unittest.TestCase):
    def test_chunks_preserve_speech_outside_frame_windows(self):
        transcript = {"segments": [{"id": i, "start": i * 10, "end": i * 10 + 2, "text": "fact " * 4} for i in range(8)]}
        chunks = evidence_chunks(transcript, [{"timestamp": 0}], max_chars=40)
        self.assertEqual([s["id"] for c in chunks for s in c["segments"]], list(range(8)))

    def test_assets_appear_under_matching_subtopic_and_equations_are_deduplicated(self):
        frames = [{"index": i, "time": i * 10} for i in range(3)]
        observations = {i: {"status": "ok", "equations": [{"raw": "E = mc2", "confidence": .5}]} for i in range(3)}
        sections = [{"title": "Physics", "start_seconds": 0, "subtopics": [
            {"title": "First concept", "start_seconds": 0, "explanation": ["First explanation."]},
            {"title": "Second concept", "start_seconds": 10, "explanation": ["Second explanation."]}]}]
        diagram = {"title": "Example diagram", "time_seconds": 15, "generated_image": "diagram.svg"}
        note = compile_study("Lecture", frames, observations, {"segments": []}, {"job_id": "abc"}, [diagram], sections)
        self.assertEqual(len(note["equations"]), 1)
        text = study_markdown(note)
        self.assertNotIn("E = mc2", text)  # Unverified OCR formulas remain source evidence.
        self.assertLess(text.index("Second concept"), text.index("Example diagram"))
        self.assertNotIn("Visual timeline", text)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "diagram.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"></svg>')
            try:
                export_pdf(note, Path(tmp), False)
            except RuntimeError as exc:
                if "pdflatex is not installed" not in str(exc): raise
            tex = (Path(tmp) / "pdf/study-notes.html").read_text()
            self.assertLess(tex.index("Second concept"), tex.index("Example diagram"))
            self.assertNotIn("Canonical diagrams", tex)

    def test_fallback_is_explicit_and_keeps_all_transcript(self):
        note = compile_study("Lecture", [], {}, {"segments": [{"text": "first"}, {"text": "second"}]}, {})
        text = study_markdown(note)
        self.assertIn("synthesis was unavailable", text)
        self.assertIn("first", text)
        self.assertIn("second", text)


if __name__ == "__main__":
    unittest.main()
