from __future__ import annotations

import os
import sys
import importlib.util
import numpy as np

from keypoints.detect import PoseDetection


def _load_module(name: str, path: str):
    """Load a single module file by absolute path into sys.modules."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _venv_site_packages() -> str:
    """Return the venv site-packages path, walking up from this file.

    Path depth: movement_analysis/keypoints/backends/mediapipe_backend.py
    → up 3 levels → movement_analysis/ → .venv/lib/python3.14/site-packages
    """
    up = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(up, ".venv", "lib", "python3.14", "site-packages")


def _mediapipe_vision_path() -> str:
    return os.path.join(_venv_site_packages(), "mediapipe", "tasks", "python", "vision")


def _mediapipe_python_path() -> str:
    return os.path.join(_venv_site_packages(), "mediapipe", "tasks", "python")


class MediaPipeBackend:
    """Legacy MediaPipe Pose fallback backend using the Tasks Vision Python API.

    Returns 33 MediaPipe landmarks mapped to COCO-17.
    Uses IMAGE running mode (synchronous per-frame detection).
    """

    def __init__(self, model_tier: str = "balanced", device: str = "cuda:0"):
        vision_path = _mediapipe_vision_path()
        python_path = _mediapipe_python_path()

        # Load pose_landmarker (contains PoseLandmarker, PoseLandmarkerOptions, _RunningMode)
        pl = _load_module(
            "pose_landmarker",
            os.path.join(vision_path, "pose_landmarker.py")
        )

        # Load vision/core/image
        img_mod = _load_module(
            "mediapipe.tasks.python.vision.core.image",
            os.path.join(vision_path, "core", "image.py")
        )
        base_mod = _load_module(
            "mediapipe.tasks.python.core.base_options",
            os.path.join(python_path, "core", "base_options.py")
        )

        self._device = device
        self._pose_landmarker = pl
        self._PoseLandmarker = pl.PoseLandmarker
        self._PoseLandmarkerOptions = pl.PoseLandmarkerOptions
        self._RunningMode = pl._RunningMode
        self._Image = img_mod.Image
        self._ImageFormat = img_mod.ImageFormat
        self._BaseOptions = base_mod.BaseOptions

        # Model tier → asset filename
        model_map = {
            "fast":     "pose_landmarker_lite.task",
            "balanced": "pose_landmarker.task",
            "accurate": "pose_landmarker_full.task",
        }
        model_name = model_map.get(model_tier, "pose_landmarker.task")
        model_path = os.path.join(os.path.dirname(__file__), "..", model_name)

        if not os.path.exists(model_path):
            model_url = (
                f"https://storage.googleapis.com/mediapipe-assets/{model_name}"
            )
            import urllib.request
            print(f"  Downloading MediaPipe Pose model: {model_name} ...")
            urllib.request.urlretrieve(model_url, model_path)
            print(f"  Downloaded → {model_path}")

        options = self._PoseLandmarkerOptions(
            base_options=self._BaseOptions(model_asset_path=model_path),
            running_mode=self._RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._pose = self._PoseLandmarker.create_from_options(options)

        # MediaPipe 33-landmark → COCO-17 mapping
        # For each COCO index, specify which MP landmark maps to it.
        # COCO 0-16 = nose, l/r eye, l/r ear, l/r shoulder,
        #             l/r elbow, l/r wrist, l/r hip, l/r knee, l/r ankle
        self._coco_from_mp = [
            0,      # 0  nose → MP 0
            1,      # 1  left_eye → MP 1
            2,      # 2  right_eye → MP 2
            3,      # 3  left_ear → MP 3
            4,      # 4  right_ear → MP 4
            5,      # 5  left_shoulder → MP 5
            6,      # 6  right_shoulder → MP 6
            7,      # 7  left_elbow → MP 7
            8,      # 8  right_elbow → MP 8
            9,      # 9  left_wrist → MP 9
            10,     # 10 right_wrist → MP 10
            11,     # 11 left_hip → MP 11
            12,     # 12 right_hip → MP 12
            13,     # 13 left_knee → MP 13
            14,     # 14 right_knee → MP 14
            15,     # 15 left_ankle → MP 15
            16,     # 16 right_ankle → MP 16
        ]

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.3):
        h, w = frame.shape[:2]

        # BGR → RGB for MediaPipe
        rgb = np.flip(frame, axis=2)
        mp_image = self._Image(
            image_format=self._ImageFormat.SRGB,
            data=rgb
        )

        result = self._pose.detect(mp_image)

        if not result.pose_landmarks:
            return None

        raw_kps = result.pose_landmarks[0]

        keypoints = np.full((17, 2), np.nan, dtype=np.float64)
        scores = np.zeros(17, dtype=np.float64)

        for coco_idx, mp_idx in enumerate(self._coco_from_mp):
            lm = raw_kps[mp_idx]
            # presence < 0 means absent
            conf = float(getattr(lm, 'presence', 1.0))
            if conf < 0 or conf < conf_threshold:
                continue
            keypoints[coco_idx] = [lm.x * w, lm.y * h]
            scores[coco_idx] = max(conf, 1e-6)

        valid = ~np.isnan(keypoints[:, 0])
        if not valid.any():
            return None

        x_min, x_max = np.nanmin(keypoints[valid, 0]), np.nanmax(keypoints[valid, 0])
        y_min, y_max = np.nanmin(keypoints[valid, 1]), np.nanmax(keypoints[valid, 1])
        bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float64)

        return PoseDetection(
            keypoints=keypoints,
            scores=scores,
            bbox=bbox,
            bbox_score=1.0,
            backend="mediapipe",
            meta={"num_people": 1},
        )
