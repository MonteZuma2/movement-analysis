"""Skeleton visualizer — draw COCO-17 pose skeleton over video frames.

Usage:
    from utils.visualize import visualize_keypoints
    visualize_keypoints(frames, kps_filled, output_path, fps=30)

Inputs:
  frames:        list of HxWx3 BGR uint8 numpy arrays (cv2.VideoCapture output)
  kps_filled:    list of (17, 2) float arrays — output from _fill_keypoint_gaps()
  output_path:    path to write MP4 video (e.g. "output/skeleton.mp4")
  fps:           frames per second for output video
"""

from __future__ import annotations

import cv2
import numpy as np


# -------------------------------------------------------------------------------
# COCO-17 skeleton connectivity + colors
# -------------------------------------------------------------------------------
# Each entry: (joint_a, joint_b, color_bgr)
# Colors are tuned for dark-on-light/Lung video (light grey/black skeleton)
_SKELETON = [
    # ── Head (COCO 0-4) ───────────────────────────────────────────────────────
    (0,  1,  (220, 220, 220)),   # nose → left eye
    (0,  2,  (220, 220, 220)),   # nose → right eye
    (1,  3,  (200, 200, 200)),   # left eye → left ear
    (2,  4,  (200, 200, 200)),   # right eye → right ear

    # ── Trunk / torso triangle (COCO 5, 6, 11, 12) ───────────────────────────
    (5,  6,  (160, 160, 160)),   # left shoulder ↔ right shoulder
    (5,  11, (140, 140, 140)),   # left shoulder → left hip
    (6,  12, (140, 140, 140)),   # right shoulder → right hip
    (11, 12, (140, 140, 140)),   # left hip ↔ right hip (pelvis)

    # ── Left arm chain: shoulder → elbow → wrist (COCO 5 → 7 → 9) ───────────
    (5,  7,  (60, 220, 180)),    # left shoulder → left elbow
    (7,  9,  (60, 220, 180)),    # left elbow → left wrist
    # ── Right arm chain: shoulder → elbow → wrist (COCO 6 → 8 → 10) ──────────
    (6,  8,  (100, 220, 180)),   # right shoulder → right elbow
    (8,  10, (100, 220, 180)),   # right elbow → right wrist

    # ── Left leg chain: hip → knee → ankle (COCO 11 → 13 → 15) ────────────────
    (11, 13, (50, 200, 255)),    # left hip → left knee
    (13, 15, (50, 200, 255)),    # left knee → left ankle
    # ── Right leg chain: hip → knee → ankle (COCO 12 → 14 → 16) ──────────────
    (12, 14, (80, 180, 255)),    # right hip → right knee
    (14, 16, (80, 180, 255)),    # right knee → right ankle
]

_JOINT_COLORS = {
    0:  (255, 255, 255),   # nose — white
    1:  (200, 200, 200),   # left_eye
    2:  (200, 200, 200),   # right_eye
    3:  (190, 190, 190),   # left_ear
    4:  (190, 190, 190),   # right_ear
    5:  (200, 200, 200),   # left_shoulder
    6:  (200, 200, 200),   # right_shoulder
    7:  (60, 220, 180),    # left_elbow
    8:  (100, 220, 180),   # right_elbow
    9:  (60, 180, 180),    # left_wrist
    10: (100, 180, 180),   # right_wrist
    11: (50, 200, 255),    # left_hip
    12: (80, 180, 255),    # right_hip
    13: (50, 200, 255),    # left_knee
    14: (80, 180, 255),    # right_knee
    15: (50, 200, 255),    # left_ankle
    16: (80, 180, 255),    # right_ankle
}


def draw_skeleton(frame: np.ndarray, kp: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    """Draw pose skeleton over a single BGR frame.

    Args:
        frame:       HxWx3 BGR uint8 image
        kp:          (17, 2) float array — pixel coords; NaN for missing joints
        alpha:       blend factor (1.0 = fully opaque skeleton)

    Returns:
        BGR uint8 image with skeleton overlaid
    """
    # Reduce opacity if the person is faint in frame (light gray on light bg)
    # Draw skeleton lines
    out = frame.copy()
    for ja, jb, color in _SKELETON:
        pa = kp[ja]
        pb = kp[jb]
        if np.isnan(pa).any() or np.isnan(pb).any():
            continue
        pt_a = (int(round(pa[0])), int(round(pa[1])))
        pt_b = (int(round(pb[0])), int(round(pb[1])))
        # Body/leg bones: thicker; head/ear: thinner
        thickness = 3 if jb in {5, 6, 11, 12} else 2
        cv2.line(out, pt_a, pt_b, color, thickness=thickness, lineType=cv2.LINE_AA)

    # Draw joint circles (sized by importance: 6 for main body, 3 for head/extremities)
    for j in range(17):
        p = kp[j]
        if np.isnan(p).any():
            continue
        cx, cy = int(round(p[0])), int(round(p[1]))
        color = _JOINT_COLORS.get(j, (200, 200, 200))
        # Hip/knee/ankle and shoulder/elbow/wrist: larger dots; head/ear: smaller
        radius = 5 if j in {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16} else 3
        cv2.circle(out, (cx, cy), radius=radius, color=color, thickness=-1, lineType=cv2.LINE_AA)

    return out


def visualize_keypoints(
    frames: list[np.ndarray],
    kps_filled: list[np.ndarray],
    output_path: str,
    fps: float = 30.0,
) -> str:
    """Render pose skeleton overlay on frames and write mp4 video.

    Args:
        frames:      list of HxWx3 BGR uint8 frames from cv2.VideoCapture.read()
        kps_filled:  list of (17, 2) float arrays (same length as frames)
        output_path: path to output MP4 file
        fps:         frames per second for output video

    Returns:
        Absolute path to the written MP4 file.
    """
    if not frames:
        raise ValueError("No frames provided to visualize_keypoints")

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter could not open: {output_path}")

    for frame, kp in zip(frames, kps_filled):
        annotated = draw_skeleton(frame, kp)
        writer.write(annotated)

    writer.release()
    return output_path
