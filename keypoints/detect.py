from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("movement_analysis.keypoints")

JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

LOWER_LIMB = {
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


@dataclass
class PoseDetection:
    keypoints: np.ndarray
    scores: np.ndarray
    bbox: Optional[np.ndarray]
    bbox_score: Optional[float]
    backend: str
    track_id: Optional[int] = None
    meta: dict = field(default_factory=dict)


_BACKEND = None


def _load_backend():
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    backend_name = os.getenv("POSE_BACKEND", "rtmpose").lower().strip()
    model_tier = os.getenv("POSE_MODEL_TIER", "balanced").lower().strip()
    device = os.getenv("POSE_DEVICE", "cuda:0")

    errors = []

    if backend_name in ("rtmpose", "auto"):
        try:
            from keypoints.backends.rtmpose_backend import RTMPoseBackend
            _BACKEND = RTMPoseBackend(model_tier=model_tier, device=device)
            logger.info("Pose backend initialized: RTMPose tier=%s device=%s", model_tier, device)
            return _BACKEND
        except Exception as exc:
            errors.append(f"RTMPose unavailable: {exc}")
            logger.warning("RTMPose unavailable: %s", exc)
            if backend_name == "rtmpose":
                raise

    if backend_name in ("yolo", "ultralytics", "auto"):
        try:
            from keypoints.backends.yolo_pose_backend import YoloPoseBackend
            _BACKEND = YoloPoseBackend(model_tier=model_tier, device=device)
            logger.info("Pose backend initialized: YOLO tier=%s device=%s", model_tier, device)
            return _BACKEND
        except Exception as exc:
            errors.append(f"YOLO unavailable: {exc}")
            logger.warning("YOLO unavailable: %s", exc)
            if backend_name in ("yolo", "ultralytics"):
                raise

    try:
        from keypoints.backends.mediapipe_backend import MediaPipeBackend
        _BACKEND = MediaPipeBackend(model_tier=model_tier, device=device)
        logger.info("Pose backend initialized: MediaPipe fallback")
        return _BACKEND
    except Exception as exc:
        errors.append(f"MediaPipe unavailable: {exc}")
        raise RuntimeError("No pose backend could be initialized: " + " | ".join(errors))


def detect_pose(frame: np.ndarray, conf_threshold: float = 0.3) -> Optional[PoseDetection]:
    backend = _load_backend()
    result = backend.detect(frame, conf_threshold=conf_threshold)
    if result is None:
        return None

    kps = np.asarray(result.keypoints, dtype=np.float64)
    scores = np.asarray(result.scores, dtype=np.float64)

    if kps.shape != (17, 2):
        raise ValueError(f"Expected keypoints shape (17, 2), got {kps.shape}")
    if scores.shape != (17,):
        raise ValueError(f"Expected scores shape (17,), got {scores.shape}")

    kps[scores < conf_threshold] = np.nan
    scores[scores < conf_threshold] = 0.0

    result.keypoints = kps
    result.scores = scores
    return result


def detect_keypoints(frame: np.ndarray, conf_threshold: float = 0.3) -> Optional[list[np.ndarray]]:
    result = detect_pose(frame, conf_threshold=conf_threshold)
    if result is None:
        return None
    return [result.keypoints]
