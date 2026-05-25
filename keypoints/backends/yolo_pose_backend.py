from __future__ import annotations

import numpy as np

from keypoints.detect import PoseDetection


class YoloPoseBackend:
    """Ultralytics YOLO-Pose backend (practical fallback).

    Returns 17 COCO keypoints directly with per-keypoint confidence.
    """

    def __init__(self, model_tier: str = "balanced", device: str = "cuda:0"):
        from ultralytics import YOLO

        model_map = {
            "fast": "yolo11s-pose.pt",
            "balanced": "yolo11m-pose.pt",
            "accurate": "yolo11x-pose.pt",
        }
        self._model_name = model_map.get(model_tier, "yolo11m-pose.pt")
        self._device = device
        self._model = YOLO(self._model_name)

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.3):
        results = self._model.predict(frame, conf=0.25, verbose=False, device=self._device)
        if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
            return None

        r = results[0]
        boxes = r.boxes.xyxy.detach().cpu().numpy()
        box_scores = r.boxes.conf.detach().cpu().numpy()
        kps_xy = r.keypoints.xy.detach().cpu().numpy()
        kps_conf = r.keypoints.conf.detach().cpu().numpy()

        if len(boxes) == 0:
            return None

        # Select largest high-confidence person
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        weights = areas * np.maximum(box_scores, 0.01)
        idx = int(np.argmax(weights))

        return PoseDetection(
            keypoints=kps_xy[idx].astype(np.float64),
            scores=kps_conf[idx].astype(np.float64),
            bbox=boxes[idx].astype(np.float64),
            bbox_score=float(box_scores[idx]),
            backend="yolo11-pose",
            meta={"num_people": int(len(boxes)), "model": self._model_name},
        )
