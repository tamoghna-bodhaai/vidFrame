import unittest
from app.content_units import units_from_plan, preserve_uncovered_source
from app.study import compile_study, study_markdown


class ContentUnitsTests(unittest.TestCase):
    def test_question_and_solution_share_unit_and_all_evidence_is_partitioned(self):
        segments = [{'id': i, 'start': i*10, 'text': text} for i, text in enumerate([
            'Given mass 2 kg, find the force.', 'Use acceleration 3 m/s2.', 'The answer is 6 N.',
            'Next, demonstrate how to align the lens.', 'Move the lens to the mark.'])]
        units = units_from_plan({'units': [{'start_segment': 0, 'kind': 'worked_question'},
                                         {'start_segment': 3, 'kind': 'procedure'}]}, segments,
                                [{'timestamp': 5}, {'timestamp': 32}, {'timestamp': 55}])
        self.assertEqual([s['id'] for s in units[0]['segments']], [0, 1, 2])
        self.assertEqual(sum(len(u['visuals']) for u in units), 3)
        self.assertEqual(units[1]['kind'], 'procedure')
        with self.assertRaises(ValueError):
            units_from_plan({'units': [{'start_segment': 99}]}, segments, [])

    def test_unreferenced_question_is_retained_instead_of_silently_lost(self):
        sections = [{'subtopics': [{'explanation': ['Use F=ma.'], 'source_segment_ids': [2]}]}]
        audit = preserve_uncovered_source(sections, [
            {'id': 1, 'start': 0, 'text': 'Find the force for a 2 kg mass.'},
            {'id': 2, 'start': 10, 'text': 'Use F=ma.'}, {'id': 3, 'start': 12, 'text': 'Right?'}])
        self.assertEqual([a['status'] for a in audit], ['retained_source', 'referenced', 'filler'])
        self.assertIn('Find the force', sections[0]['subtopics'][-1]['explanation'][-1])
        sections[0]['title'] = 'Question'
        rendered = study_markdown({'title': 'Notes', 'chapters': sections})
        self.assertIn('Use F=ma.', rendered)
        self.assertNotIn('Additional source details', rendered)
        self.assertNotIn('Find the force', rendered)

    def test_diagram_reference_overrides_capture_time_but_not_unit_boundaries(self):
        sections = [{'title': 'Problem', 'start_seconds': 0, 'subtopics': [
            {'title': 'Question', 'start_seconds': 0, 'diagram_ids': ['initial']},
            {'title': 'Solution', 'start_seconds': 10, 'diagram_ids': ['final']}]}]
        diagrams = [{'id': 'initial', 'time_seconds': 15, 'unit_start': 0, 'unit_end': 30},
                    {'id': 'final', 'time_seconds': 20, 'unit_start': 0, 'unit_end': 30}]
        note = compile_study('Notes', [], {}, {}, {}, diagrams, sections)
        self.assertEqual(note['chapters'][0]['subtopics'][0]['diagrams'][0]['id'], 'initial')
        self.assertEqual(note['chapters'][0]['subtopics'][1]['diagrams'][0]['id'], 'final')


class ContentJsonTests(unittest.TestCase):
    def test_tex_is_preserved_without_corrupting_newlines(self):
        from app.content_units import extract_content_json
        raw = r'{"text":"$\theta + \frac{1}{2} + \alpha$","prose":"one\ntwo","escaped":"$\\theta$"}'
        parsed = extract_content_json(raw)
        self.assertEqual(parsed['text'], r'$\theta + \frac{1}{2} + \alpha$')
        self.assertEqual(parsed['escaped'], r'$\theta$')
        self.assertEqual(parsed['prose'], 'one\ntwo')


class PlacementResolutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_valid_unassigned_figures_can_be_excluded(self):
        from app.content_units import resolve_unplaced_diagrams
        class Client:
            async def complete(self, *args, **kwargs):
                return '{"assignments":[{"id":"useful","target":"0:0"}],"excluded":[{"id":"incidental","reason":"Unrelated spring replay"},{"id":"useful","reason":"conflicting response"},{"id":"already","reason":"must not remove assigned"}]}'
        sections = [{'subtopics': [{'title': 'Question', 'start_seconds': 0, 'diagram_ids': ['already']}]}]
        diagrams = [{'id': identifier, 'unit_start': 0, 'unit_end': 10} for identifier in ('already', 'useful', 'incidental')]
        selected, audit = await resolve_unplaced_diagrams(Client(), diagrams, sections)
        self.assertEqual([d['id'] for d in selected], ['already', 'useful'])
        self.assertEqual(sections[0]['subtopics'][0]['diagram_ids'], ['already', 'useful'])
        self.assertEqual(audit[0]['id'], 'incidental')


class PdfMathBoundaryTests(unittest.TestCase):
    def test_unclosed_standalone_display_is_rendered(self):
        from app.study import _html_text
        result = _html_text(r"$$L=\frac{a+b}{2}.")
        self.assertIn('class="math display"', result)
        self.assertNotIn('$$', result)
        self.assertIn(r"\frac{a+b}{2}", result)

    def test_prose_and_existing_delimiters_are_preserved(self):
        from app.study import _html_text
        self.assertEqual(_html_text('Budget: $20'), 'Budget: $20')
        self.assertEqual(_html_text(r"$$x=\frac12$$").count('class="math display"'), 1)
        self.assertIn('class="math"', _html_text(r"Value $x=1$ here."))


class NarrationEditTests(unittest.IsolatedAsyncioTestCase):
    async def test_edits_only_flagged_prose_and_rejects_changed_math(self):
        from unittest.mock import AsyncMock
        from app.content_units import edit_recording_narration
        client = AsyncMock()
        client.complete.side_effect = [
            '{"edits":[{"id":0,"text":"Use $y$."}]}',
            '{"edits":[{"id":0,"text":"Use $x$."}]}',
        ]
        paragraph = "The speaker says to use $x$."
        sub = {"title": "Setup", "explanation": [paragraph, "Measure the loudspeaker impedance."]}
        evidence = {"kind": "source_evidence", "explanation": [paragraph]}
        audit = await edit_recording_narration(client, [{"subtopics": [sub, evidence]}])
        self.assertEqual(sub["explanation"], ["Use $x$.", "Measure the loudspeaker impedance."])
        self.assertEqual(evidence["explanation"], [paragraph])
        self.assertEqual(audit[0]["original"], paragraph)
