import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.diagrams import collect_diagrams, prepare_source_assets
from app.study import compile_study, study_markdown


class SourceAssetsTests(unittest.TestCase):
    def test_same_text_does_not_discard_changed_geometry_and_has_no_limit(self):
        frames = [{'index': i, 'time': i} for i in range(130)]
        observations = {i: {'has_diagram': True, 'diagram': {
            'type': 'other', 'title': 'Same title', 'labels': ['F'], 'confidence': .2,
        }} for i in range(130)}
        self.assertEqual(len(collect_diagrams(frames, observations)), 130)

    def test_exact_duplicates_changed_labels_and_missing_assets_are_accounted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'frames').mkdir()
            frames, diagrams = [], []
            for i in range(4):
                frame = {'index': i, 'time': i, 'file': f'{i}.png'}
                frames.append(frame)
                if i < 3:
                    pixels = np.full((80, 100, 3), 255, np.uint8)
                    if i == 2:
                        cv2.putText(pixels, 'F', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    cv2.imwrite(str(root / 'frames' / frame['file']), pixels)
                diagrams.append({'frame_index': i, 'time_seconds': i, 'scene_index': 0, 'title': 'Force'})
            assets, audit = prepare_source_assets(root, frames, diagrams)
            self.assertEqual(len(assets), 2)
            self.assertEqual([x['status'] for x in audit], ['included', 'duplicate', 'included', 'missing'])
            self.assertEqual(assets[0]['crop_scope'], 'full_frame')
            self.assertTrue(all((root / x['display_image']).is_file() for x in assets))
            sections = [{'title': 'Topic', 'start_seconds': 0, 'subtopics': [
                {'title': 'Before', 'start_seconds': 0}, {'title': 'After', 'start_seconds': 2}]}]
            note = compile_study('Lecture', [], {}, {}, {'job_id': 'test'}, assets, sections)
            self.assertEqual(note['chapters'][0]['subtopics'][1]['diagrams'][0]['frame_index'], 2)
            self.assertEqual(study_markdown(note).count('![Source diagram]'), 2)

    def test_original_video_resolution_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = cv2.VideoWriter(str(root / 'source.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 1, (640, 480))
            self.assertTrue(video.isOpened())
            video.write(np.full((480, 640, 3), 220, np.uint8))
            video.release()
            assets, _ = prepare_source_assets(root, [{'index': 0}], [
                {'frame_index': 0, 'bbox': [.25, .25, .5, .5], 'time_seconds': 0}])
            self.assertEqual(assets[0]['source_resolution'], 'original_video')
            pixels = cv2.imread(str(root / assets[0]['image']))
            self.assertGreater(pixels.shape[1], 320)
            self.assertLess(pixels.shape[1], 640)
