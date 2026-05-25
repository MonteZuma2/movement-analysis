from __future__ import annotations

import os
import numpy as np

from keypoints.detect import PoseDetection
from keypoints.subject_selector import select_primary_subject


class RTMPoseBackend:
    """RTMPose + RTMDet backend — preferred clinical backend.

    Uses a two-stage detector-crop-pose pipeline for maximum robustness.
    Requires OpenMMLab: mmdet, mmpose.
    """

    def __init__(self, model_tier: str = "balanced", device: str = "cuda:0"):
        from mmdet.apis import init_detector
        from mmpose.apis import init_model as init_pose

        self._tier = model_tier
        self._device = device

        cfg = self._resolve_config(model_tier)
        self._detector = init_detector(
            cfg["det_config"], cfg["det_checkpoint"], device=device
        )
        self._pose_model = init_pose(
            cfg["pose_config"], cfg["pose_checkpoint"], device=device
        )

    def _resolve_config(self, model_tier: str) -> dict:
        presets = {
            "fast": {
                "pose_config": (
                    "configs/body_2d_keypoint/rtmpose/coco/"
                    "rtmpose-s_8xb256-420e_coco-256x192.py"
                ),
                "pose_checkpoint": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/"
                                   "rtmpose/coco/rtmpose-s_simcc-coco-popart_256x192-b59268d1_20230109.pth",
                "det_config": "configs/rtmdet/rtmdet_m_640-8xb32_coco-person.py",
                "det_checkpoint": "https://download.openmmlab.com/mmdetection/v2.0/rtmdet/"
                                  "rtmdet_m_640-8xb32_coco-person/rtmdet_m_640-8xb32_coco-person_20220731_155618-b5fbdbae.pth",
            },
            "balanced": {
                "pose_config": (
                    "configs/body_2d_keypoint/rtmpose/coco/"
                    "rtmpose-l_8xb256-420e_coco-256x192.py"
                ),
                "pose_checkpoint": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/"
                                   "rtmpose/coco/rtmpose-l_simcc-coco-popart_256x192-b59268d1_20230109.pth",
                "det_config": "configs/rtmdet/rtmdet_m_640-8xb32_coco-person.py",
                "det_checkpoint": "https://download.openmmlab.com/mmdetection/v2.0/rtmdet/"
                                  "rtmdet_m_640-8xb32_coco-person/rtmdet_m_640-8xb32_coco-person_20220731_155618-b5fbdbae.pth",
            },
            "accurate": {
                "pose_config": (
                    "configs/body_2d_keypoint/rtmpose/coco/"
                    "rtmpose-x_8xb256-700e_coco-384x288.py"
                ),
                "pose_checkpoint": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/"
                                   "rtmpose/coco/rtmpose-x_simcc-coco-popart_384x288-b59268d1_20230109.pth",
                "det_config": "configs/rtmdet/rtmdet_m_640-8xb32_coco-person.py",
                "det_checkpoint": "https://download.openmmlab.com/mmdetection/v2.0/rtmdet/"
                                  "rtmdet_m_640-8xb32_coco-person/rtmdet_m_640-8xb32_coco-person_20220731_155618-b5fbdbae.pth",
            },
        }
        return presets.get(model_tier, presets["balanced"])

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.3):
        from mmdet.apis import inference_detector
        from mmpose.apis import inference_topdown

        det_result = inference_detector(self._detector, frame)
        pred = det_result.pred_instances

        bboxes = pred.bboxes.detach().cpu().numpy()
        scores = pred.scores.detach().cpu().numpy()
        labels = pred.labels.detach().cpu().numpy()

        person_bboxes = []
        for bbox, score, label in zip(bboxes, scores, labels):
            if int(label) == 0 and float(score) >= 0.30:
                person_bboxes.append({
                    "bbox": bbox.astype(np.float64),
                    "bbox_score": float(score),
                })

        if not person_bboxes:
            return None

        selected = select_primary_subject(person_bboxes, frame_shape=frame.shape)
        pose_results = inference_topdown(self._pose_model, frame, [selected])

        if not pose_results:
            return None

        inst = pose_results[0].pred_instances
        keypoints = np.asarray(inst.keypoints[0], dtype=np.float64)
        keypoint_scores = np.asarray(inst.keypoint_scores[0], dtype=np.float64)

        return PoseDetection(
            keypoints=keypoints,
            scores=keypoint_scores,
            bbox=np.asarray(selected["bbox"], dtype=np.float64),
            bbox_score=float(selected.get("bbox_score", 0.0)),
            backend="rtmpose",
            meta={"num_people": len(person_bboxes)},
        )
