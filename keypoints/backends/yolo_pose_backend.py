from __future__ import annotations

import numpy as np

from keypoints.detect import PoseDetection


class YoloPoseBackend:
    """Ultralytics YOLO-Pose backend — COCO-17 keypoints directly.

    YOLO-Pose outputs 17 COCO-format keypoints natively (no mapping needed).
    Confidence is per-keypoint, set via conf threshold.
    """

    def __init__(self, model_tier: str = "balanced", device: str = "cpu"):
        from ultralytics import YOLO

        # yolov8n = nano (fastest), yolov8s = small, yolov8m = medium, yolov8x = xlarge
        model_map = {
            "fast":     "yolov8n-pose.pt",
            "balanced": "yolov8s-pose.pt",
            "accurate": "yolov8m-pose.pt",
        }
        self._model_name = model_map.get(model_tier, "yolov8s-pose.pt")
        self._device = device
        self._model = YOLO(self._model_name)

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.3):
        """Run pose detection on a single frame.

        Returns PoseDetection with (17, 2) keypoints and (17,) confidence scores,
        or None if no person detected above threshold.
        """
        results = self._model.predict(
            frame, conf=conf_threshold, verbose=False, device=self._device
        )
        if not results or results[0].keypoints is None:
            return None

        r = results[0]
        xy = r.keypoints.xy     # (N, 17, 2)
        conf = r.keypoints.conf  # (N, 17)

        if xy.shape[0] == 0:
            return None

        # Take the first (most confident) person
        kps = xy[0].cpu().numpy()       # (17, 2) — pixel coords
        scores = conf[0].cpu().numpy()   # (17,)

        h, w = frame.shape[:2]
        keypoints = kps.astype(np.float64)
        valid = scores >= 0.0

        if not valid.any():
            return None

        x_min, x_max = keypoints[valid, 0].min(), keypoints[valid, 0].max()
        y_min, y_max = keypoints[valid, 1].min(), keypoints[valid, 1].max()
        bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float64)

        return PoseDetection(
            keypoints=keypoints,
            scores=scores.astype(np.float64),
            bbox=bbox,
            bbox_score=float(scores.max()),
            backend="YOLO-Pose",
        )
