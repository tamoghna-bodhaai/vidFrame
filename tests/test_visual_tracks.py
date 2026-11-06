import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from app.visual_tracks import select_visual_states


class VisualTracksTests(unittest.TestCase):
    def test_camera_shift_and_title_changes_do_not_duplicate_drawing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = np.full((320, 480, 3), 245, np.uint8)
            for i in range(12):
                cv2.putText(base, f'R{i}={i+2}', (20+(i%3)*150, 40+(i//3)*70),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (10, 10, 10), 2)
            diagrams = []
            for i in range(10):
                image = cv2.warpAffine(base, np.float32([[1, 0, i], [0, 1, i//2]]), (480, 320), borderValue=(245,245,245))
                cv2.imwrite(str(root / f'{i}.png'), image)
                diagrams.append({'id': f'd{i}', 'image': f'{i}.png', 'time_seconds': i*5,
                                 'title': f'Different description {i}'})
            selected, audit, tracks = select_visual_states(root, diagrams, [{'start': 0, 'end': 100}])
            self.assertEqual(len(tracks), 1)
            self.assertEqual(len(selected), 1)
            self.assertEqual(len(audit), 10)
            selected2, _, tracks2 = select_visual_states(root, diagrams, [{'start': 0, 'end': 25}, {'start': 25, 'end': 100}])
            self.assertEqual(len(tracks2), 2)  # Same visual in a new question is not deleted.
            self.assertEqual(len(selected2), 2)

    def test_new_annotations_keep_initial_and_completed_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = np.full((340, 600, 3), 245, np.uint8)
            for i in range(12):
                cv2.putText(base, f'P{i}={i+2}', (20+(i%2)*130, 40+(i//2)*45),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (10,10,10), 2)
            diagrams = []
            for i in range(8):
                image = base.copy()
                if i >= 4:
                    for j in range(12):
                        cv2.putText(image, f'Q{j}={j+9}', (310+(j%2)*130, 40+(j//2)*45),
                                    cv2.FONT_HERSHEY_SIMPLEX, .6, (10,10,10), 2)
                cv2.imwrite(str(root / f'{i}.png'), image)
                diagrams.append({'id': f'd{i}', 'image': f'{i}.png', 'time_seconds': i*10})
            selected, _, tracks = select_visual_states(root, diagrams, [{'start': 0, 'end': 100}])
            self.assertEqual(len(tracks), 1)
            self.assertEqual({d['visual_role'] for d in selected}, {'initial', 'completed'})
            self.assertLess(selected[0]['time_seconds'], 40)
            self.assertGreaterEqual(selected[-1]['time_seconds'], 40)
