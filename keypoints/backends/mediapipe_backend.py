from __future__ import annotations

import numpy as np

from keypoints.detect import PoseDetection
from keypoints.subject_selector import select_primary_subject


class MediaPipeBackend:
    """Legacy MediaPipe Pose fallback backend.

    Returns 33 MediaPipe landmarks mapped to COCO-17.
    Does not provide per-keypoint confidence scores — scores are set to 1.0
    for visible landmarks and 0.0 for NaN/unavailable.
    """

    def __init__(self, model_tier: str = "balanced", device: str = "cuda:0"):
        import mediapipe as mp
        import os

        self._device = device
        model_path = os.path.join(os.path.dirname(__file__), "..", "pose_landmarker_full.task")
        if not os.path.exists(model_path):
            model_path = ""

        self._mp = mp
        self._pose = mp.solutions.pose.Pose(
            model_asset_path=model_path if os.path.exists(model_path) else "",
            static_image_mode=True,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # MediaPipe → COCO-17 mapping
        self._mp33_to_coco = [
            0,      # nose
            None,   # left_eye
            None,   # right_eye
            None,   # left_ear
            None,   # right_ear
            None,   # (5-10 unused)
            5,      # left_shoulder
            6,      # right_shoulder
            7,      # left_elbow
            8,      # right_elbow
            9,      # left_wrist
            10,     # right_wrist
            None,   # (17-22 unused)
            11,     # left_hip
            12,     # right_hip
            13,     # left_knee
            14,     # right_knee
            15,     # left_ankle
            16,     # right_ankle
            None,   # foot_tip
            None,   # foot_tip
            None,   # heel
            None,   # heel
        ]
        self._coco_from_mp = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.3):
        import mediapipe as mp

        rgb = np.clip(np.flip(frame, axis=2), 0, 255).astype(np.uint8)
        results = self._pose.process(rgb)

        if not results.pose_landmarks:
            return None

        raw_kps = np.asarray(results.pose_landmarks, dtype=np.float64)
        h, w = frame.shape[:2]

        # Extract COCO-17 keypoints
        keypoints = np.full((17, 2), np.nan, dtype=np.float64)
        scores = np.zeros(17, dtype=np.float64)

        for mp_idx, coco_idx in enumerate(self._mp33_to_coco):
            if coco_idx is None:
                continue
            lm = raw_kps[mp_idx]
            cx, cy = float(lm.x), float(lm.y)
            vis = float(lm.visibility) if hasattr(lm, "visibility") else 1.0
            if vis < conf_threshold:
                continue
            keypoints[coco_idx] = [cx * w, cy * h]
            scores[coco_idx] = vis

        # BBox: computed from visible keypoints
        valid = ~np.isnan(keypoints[:, 0])
        if not valid.any():
            return None

        x_min = np.nanmin(keypoints[valid, 0])
        x_max = np.nanmax(keypoints[valid, 0])
        y_min = np.nanmin(keypoints[valid, 1])
        y_max = np.nanmax(keypoints[valid, 1])
        bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float64)

        return PoseDetection(
            keypoints=keypoints,
            scores=scores,
            bbox=bbox,
            bbox_score=1.0,
            backend="mediapipe",
            meta={"num_people": 1},
        )
