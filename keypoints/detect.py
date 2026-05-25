"""2D keypoint detection via MediaPipe Pose (Tasks API).

Falls back to returning None if MediaPipe is not installed.
MediaPipe Pose provides 33 3D landmarks (COCO topology) at ~50-100 fps on CPU.

Installation:
    pip install mediapipe
    # Downloads pose_landmarker_full.task automatically on first use,
    # OR manually: wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task

Return shape matches MMPose convention used by run_session.py:
    detect_keypoints(frame, conf_threshold) -> list of (17, 2) arrays in pixel coords
"""

from __future__ import annotations

import os
import logging
from typing import Optional

import numpy as np
import cv2

logger = logging.getLogger("movement_analysis.keypoints")


# ---------------------------------------------------------------------------
# Model download / asset resolution
# ---------------------------------------------------------------------------

_MODEL_PATH = "pose_landmarker_full.task"
_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
)


def _ensure_model():
    """Download the pose landmarker model if not present."""
    if not os.path.exists(_MODEL_PATH):
        logger.info(f"Downloading MediaPipe Pose model → {_MODEL_PATH} ...")
        try:
            import urllib.request
            urllib.request.urlretrieve(_URL, _MODEL_PATH)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download MediaPipe model from {_URL}\n"
                "You can also manually download it and place it in the movement_analysis/ directory."
            ) from e
        size_mb = os.path.getsize(_MODEL_PATH) / 1_048_576
        logger.info(f"  Downloaded ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Suppress MediaPipe / TensorFlow INFO logs
# ---------------------------------------------------------------------------

try:
    os.environ.setdefault("GLOG_minloglevel", "2")
    import absl.logging

    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass


# ---------------------------------------------------------------------------
# COCO-17 subset mapping
# ---------------------------------------------------------------------------
# Standard COCO 17-keypoint ordering (used by angles_2d.py):
#   0  = nose
#   1  = left_eye
#   2  = right_eye
#   3  = left_ear
#   4  = right_ear
#   5  = left_shoulder
#   6  = right_shoulder
#   7  = left_elbow
#   8  = right_elbow
#   9  = left_wrist
#   10 = right_wrist
#   11 = left_hip
#   12 = right_hip
#   13 = left_knee
#   14 = right_knee
#   15 = left_ankle
#   16 = right_ankle
#
# MediaPipe 33-landmark indices for these joints:
#   0  = nose,        11 = left_shoulder,  12 = right_shoulder,
#   13 = left_elbow,  14 = right_elbow,   15 = left_wrist,  16 = right_wrist,
#   23 = left_hip,    24 = right_hip,     25 = left_knee,   26 = right_knee,
#   27 = left_ankle,  28 = right_ankle
#
# The 12 missing MP landmarks (eyes, ears — indices 1-4, 17-22, 29-32)
# are set to NaN in the output. This matches standard COCO format.

COCO17_FROM_MP33 = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Mapping: for each of the 33 MediaPipe landmark positions, which COCO-17 index
# does it correspond to? None = not in COCO-17 (eyes, ears, etc.).
#
# Standard COCO 17-keypoint order:
#   0=nose, 1=LEye, 2=REye, 3=LEar, 4=REar,
#   5=LSh, 6=RSh, 7=LEl, 8=REl, 9=LWr, 10=RWr,
#   11=LHip, 12=RHip, 13=LKnee, 14=RKnee, 15=LAnk, 16=RAnk
#
# MediaPipe 33-landmark indices that map to COCO:
#   MP[0]  = nose                   → COCO[0]
#   MP[11] = left_shoulder          → COCO[5]
#   MP[12] = right_shoulder         → COCO[6]
#   MP[13] = left_elbow             → COCO[7]
#   MP[14] = right_elbow            → COCO[8]
#   MP[15] = left_wrist             → COCO[9]
#   MP[16] = right_wrist            → COCO[10]
#   MP[23] = left_hip               → COCO[11]
#   MP[24] = right_hip              → COCO[12]
#   MP[25] = left_knee              → COCO[13]
#   MP[26] = right_knee             → COCO[14]
#   MP[27] = left_ankle             → COCO[15]
#   MP[28] = right_ankle            → COCO[16]
#   All other MP indices (1-4, 17-22, 29-32) → None (not in COCO-17)

MP33_TO_COCO17 = [
    0,                                                              #  0: nose → COCO[0]
    None, None, None, None, None, None, None, None, None, None,      #  1-10: not in COCO
    5, 6, 7, 8, 9, 10,                                              # 11-16 → COCO[5-10]
    None, None, None, None, None, None,                              # 17-22: not in COCO
    11, 12, 13, 14, 15, 16, None, None, None, None,                 # 23-28 → COCO[11-16], MP29-32 → NaN
]
logger.info(f"[MODLOAD] MP33_TO_COCO17 defined with {len(MP33_TO_COCO17)} elements")

# ---------------------------------------------------------------------------
# Global model singleton
# ---------------------------------------------------------------------------

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        _ensure_model()
        from mediapipe.tasks.python.vision import pose_landmarker
        from mediapipe.tasks.python.vision.core import image as mp_image
        from mediapipe.tasks.python.core import base_options

        options = pose_landmarker.PoseLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=pose_landmarker._RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _detector = pose_landmarker.PoseLandmarker.create_from_options(options)
    return _detector


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


def detect_keypoints(
    frame: np.ndarray,
    conf_threshold: float = 0.3,
) -> Optional[list[np.ndarray]]:
    """Run MediaPipe Pose on a single frame.

    Args:
        frame: BGR uint8 image (HxWx3) — must be 3 channels, uint8
        conf_threshold: minimum keypoint confidence (0-1); keypoints below
            this are returned as NaN

    Returns:
        None if no person detected,
        otherwise list of length 1: [ (17, 2) numpy array in pixel coords ]
        Matches MMPose convention used by run_session.py:
            [person][joints][xy]
    """
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        return None

    try:
        detector = _get_detector()
    except Exception as e:
        logger.warning(f"Failed to load MediaPipe detector: {e}")
        return None

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    from mediapipe.tasks.python.vision.core import image as mp_image
    mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_img)

    if not result.pose_landmarks:
        return None

    # result.pose_landmarks[0] is a list of 33 NormalizedLandmark objects
    all_lms = result.pose_landmarks[0]   # list[NormalizedLandmark], len=33

    # We always produce a (17, 2) array in standard COCO order.
    # Joints not in MediaPipe's 33-landmark set (eyes, ears, etc.) → NaN.
    kp = np.full((17, 2), np.nan, dtype=np.float64)

    for coco_i, mp_i in enumerate(MP33_TO_COCO17):
        if coco_i >= 17:
            continue  # skip indices beyond COCO-17 capacity
        if mp_i is None:
            continue  # no MP landmark for this COCO joint → NaN
        lm = all_lms[mp_i]
        if lm.visibility >= conf_threshold:
            kp[coco_i, 0] = lm.x * w
            kp[coco_i, 1] = lm.y * h

    # Return in MMPose format: list of (17, 2) per person — single person, list of 1
    return [kp]