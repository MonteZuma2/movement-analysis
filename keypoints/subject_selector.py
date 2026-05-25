from __future__ import annotations

import numpy as np


def select_primary_subject(candidates: list[dict], frame_shape) -> dict:
    """Select the most likely patient/subject.

    Current heuristic:
    - prefer larger person box
    - prefer central person box
    - prefer higher detection confidence

    Later improvement:
    - add temporal continuity with IoU against previous selected bbox
    - optionally lock a track ID for the entire session
    """
    h, w = frame_shape[:2]
    cx_frame, cy_frame = w / 2.0, h / 2.0
    diag = max((w ** 2 + h ** 2) ** 0.5, 1.0)

    best = None
    best_score = -1e9

    for c in candidates:
        box = np.asarray(c["bbox"], dtype=float)
        x1, y1, x2, y2 = box
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_norm = area / max(w * h, 1)
        box_cx, box_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        center_dist = ((box_cx - cx_frame) ** 2 + (box_cy - cy_frame) ** 2) ** 0.5 / diag
        det_score = float(c.get("bbox_score", 0.5))

        score = 2.0 * area_norm + 1.0 * det_score - 0.75 * center_dist
        if score > best_score:
            best_score = score
            best = c

    return best
