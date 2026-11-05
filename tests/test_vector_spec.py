import tempfile
import unittest
from pathlib import Path

from app.vector_spec import render_spec


class VectorSpecTests(unittest.TestCase):
    def test_circuit_elements_render_from_normalized_geometry(self):
        diagram = {"render_spec": {"confidence": .94, "elements": [
            {"shape": "battery", "x": .1, "y": .5, "x2": .3, "y2": .5, "text": "V"},
            {"shape": "line", "x": .3, "y": .5, "x2": .45, "y2": .5},
            {"shape": "resistor", "x": .45, "y": .5, "x2": .75, "y2": .5, "text": "R"},
        ]}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "circuit.svg"
            self.assertTrue(render_spec(diagram, path))
            svg = path.read_text()
            self.assertIn('<svg', svg)
            self.assertIn('>R</text>', svg)
            self.assertIn('<rect', svg)

    def test_rejects_untrusted_markup_and_out_of_bounds_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.svg"
            for element in [
                {"shape": "script", "x": .1, "y": .1},
                {"shape": "label", "x": 1.5, "y": .1, "text": "bad"},
            ]:
                diagram = {"render_spec": {"confidence": .95, "elements": [element, element]}}
                self.assertFalse(render_spec(diagram, path))
                self.assertFalse(path.exists())

    def test_rejects_nonfinite_confidence(self):
        diagram = {"render_spec": {"confidence": float("nan"), "elements": [
            {"shape": "line", "x": .1, "y": .1, "x2": .8, "y2": .8},
            {"shape": "label", "x": .5, "y": .5, "text": "x"},
        ]}}
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(render_spec(diagram, Path(tmp) / "bad.svg"))


if __name__ == "__main__":
    unittest.main()
