"""Track source drawings through camera motion and incremental annotation.

Features and geometric agreement establish identity, not titles or exact pixels.
All omitted observations remain in the audit and source files.
"""
from pathlib import Path
import re

import cv2
import numpy as np


def _features(path):
    image = cv2.imread(str(path))
    if image is None:
        return None
    scale = min(1., 800 / max(image.shape[:2]))
    gray = cv2.cvtColor(cv2.resize(image, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
    points, descriptors = cv2.SIFT_create(nfeatures=1400, contrastThreshold=.025).detectAndCompute(gray, None)
    return {'points': np.float32([p.pt for p in points]), 'descriptors': descriptors,
            'shape': gray.shape, 'area': image.shape[0]*image.shape[1],
            'sharpness': float(cv2.Laplacian(gray, cv2.CV_32F).var())}


def _agreement(a, b):
    """Return geometric inliers and descriptor coverage; reject incidental matches."""
    if a is None or b is None or a['descriptors'] is None or b['descriptors'] is None:
        return 0, 0.
    matches = cv2.BFMatcher().knnMatch(a['descriptors'], b['descriptors'], k=2)
    good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < .72*pair[1].distance]
    if len(good) < 10:
        return 0, 0.
    x = np.float32([a['points'][m.queryIdx] for m in good])
    y = np.float32([b['points'][m.trainIdx] for m in good])
    matrix, mask = cv2.estimateAffinePartial2D(x, y, method=cv2.RANSAC, ransacReprojThreshold=3)
    if matrix is None or mask is None:
        return 0, 0.
    valid = mask.ravel().astype(bool)
    count = int(valid.sum())
    if count < 10 or count/len(good) < .55:
        return 0, 0.
    # Matches confined to a tiny logo or repeated word do not establish a drawing.
    spans = [np.ptp(points[valid], axis=0) / np.array(feature['shape'][::-1])
             for points, feature in ((x, a), (y, b))]
    if max(float(span.prod()) for span in spans) < .12:
        return 0, 0.
    return count, count / max(1, min(len(a['points']), len(b['points'])))


def _state_similarity(a, b):
    count, _ = _agreement(a, b)
    if a is None or b is None:
        return 0.
    # Symmetric coverage distinguishes new annotations from a contained crop.
    return count / max(1, len(a['points']), len(b['points']))


def select_visual_states(job_dir: Path, diagrams: list[dict], units: list[dict]):
    """Select legible opening/completed views per drawing within a content unit."""
    features = {d['id']: _features(job_dir / d['image']) for d in diagrams}
    tracks = []
    ordered = sorted(diagrams, key=lambda d: d.get("time_seconds", 0))
    for diagram in ordered:
        time = float(diagram.get('time_seconds') or 0)
        unit_index = max((i for i, u in enumerate(units) if u['start'] <= time), default=0)
        feature = features[diagram['id']]
        # A new slide often appears just before its spoken question. Require visual
        # agreement with the next unit before moving that pre-roll across the boundary.
        if unit_index + 1 < len(units):
            boundary = units[unit_index + 1]['start']
            if 0 < boundary - time <= 12:
                upcoming = [d for d in ordered if boundary <= d['time_seconds'] <= boundary + 20]
                terms = lambda text: {word.rstrip('s') for word in re.findall(r"[a-z]{4,}", text.lower())
                                      if word not in {'diagram', 'physics', 'system', 'model', 'question', 'problem', 'light', 'vertical',
                                                      'horizontal', 'center', 'point', 'value', 'function', 'force'}}
                source_terms = terms(str(diagram.get('title', '')) + ' ' + str(diagram.get('description', '')))
                next_terms = terms(units[unit_index + 1].get('title', ''))
                current_terms = terms(units[unit_index].get('title', ''))
                if ((len(source_terms & next_terms) >= 1 and
                     len(source_terms & next_terms) > len(source_terms & current_terms)) or
                        any(count >= 30 and coverage >= .35
                            for count, coverage in (_agreement(feature, features[d['id']]) for d in upcoming))):
                    unit_index += 1
        best, score = None, 0
        for track in tracks:
            if track['unit_index'] != unit_index:
                continue
            members = track['members']
            for other in (members[0], members[-1], max(members, key=lambda d: (features[d['id']] or {}).get('area', 0))):
                count, coverage = _agreement(features[other['id']], feature)
                if count >= 12 and coverage >= .08 and count > score:
                    best, score = track, count
        if best is None:
            best = {'id': f'visual-track-{len(tracks)+1:03d}', 'unit_index': unit_index, 'members': []}
            tracks.append(best)
        best['members'].append(diagram)

    selected, audit, manifests = [], [], []
    for track in tracks:
        members = track['members']
        start, end = (float(members[i]['time_seconds']) for i in (0, -1))
        def quality(d):
            f = features[d['id']]
            if not f:
                return 0
            # Prefer a complete region over a tiny, sharp subset of the same board.
            return np.log1p(f['area']) + .25*np.log1p(f['sharpness'])
        opening = [d for d in members if d['time_seconds'] <= start + min(25, (end-start)*.25)]
        closing = [d for d in members if d['time_seconds'] >= end - min(25, (end-start)*.25)]
        first, last = max(opening, key=quality), max(closing, key=quality)
        similarity = _state_similarity(features[first['id']], features[last['id']])
        def facts(d):
            return {re.sub(r"\W+", "", str(value).lower()) for key in ('labels', 'values', 'edges')
                    for value in d.get(key, []) if str(value).strip()}
        facts_changed = facts(first) != facts(last)
        if first['id'] == last['id'] or (similarity >= .65 and not facts_changed):
            choices = [(max(members, key=quality), 'representative')]
        else:
            choices = [(first, 'initial'), (last, 'completed')]
        # Retain a transient intermediate state only when it persists in a second
        # observation and is poorly represented by both chosen endpoints.
        for position in range(1, len(members)-1):
            candidate, following = members[position:position+2]
            if candidate['id'] in {d['id'] for d, _ in choices}:
                continue
            coverage = max(_state_similarity(features[candidate['id']], features[d['id']])
                           for d, _ in choices)
            count, persistence = _agreement(features[candidate['id']], features[following['id']])
            if coverage < .3 and count >= 20 and persistence >= .65:
                choices.append((candidate, 'intermediate'))
        choices.sort(key=lambda pair: pair[0]['time_seconds'])
        selected_ids = {d['id'] for d, _ in choices}
        for diagram, role in choices:
            result = dict(diagram, track_id=track['id'], visual_role=role,
                          unit_index=track['unit_index'], unit_start=units[track['unit_index']]['start'],
                          track_start=start, track_end=end)
            if np.isfinite(units[track['unit_index']]['end']):
                result['unit_end'] = units[track['unit_index']]['end']
            selected.append(result)
        for diagram in members:
            audit.append({'id': diagram['id'], 'track_id': track['id'],
                          'status': 'selected' if diagram['id'] in selected_ids else 'represented',
                          'represented_by': sorted(selected_ids),
                          'reason': 'Temporal drawing track; opening/completed or best unchanged view'})
        manifests.append({**{k: v for k, v in track.items() if k != 'members'},
                          'start': start, 'end': end, 'source_ids': [d['id'] for d in members],
                          'selected_ids': sorted(selected_ids)})
    return sorted(selected, key=lambda d: d['time_seconds']), audit, manifests
