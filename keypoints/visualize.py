from __future__ import annotations

import cv2
import numpy as np
from typing import Optional

# Skeleton: (joint_a, joint_b, color_bgr)
_SKELETON = [
    # Head
    (0, 1, (220, 220, 220)),
    (0, 2, (220, 220, 220)),
    (1, 3, (220, 220, 220)),
    (2, 4, (220, 220, 220)),
    # Trunk
    (5, 6, (150, 150, 150)),
    (5, 11, (150, 150, 150)),
    (6, 12, (150, 150, 150)),
    (11, 12, (150, 150, 150)),
    # Left arm
    (5, 7, (60, 220, 180)),
    (7, 9, (60, 220, 180)),
    # Right arm
    (6, 8, (100, 220, 180)),
    (8, 10, (100, 220, 180)),
    # Left leg
    (11, 13, (50, 200, 255)),
    (13, 15, (50, 200, 255)),
    # Right leg
    (12, 14, (80, 180, 255)),
    (14, 16, (80, 180, 255)),
]

_JOINT_CONFIDENCE_COLORS = {
    "green": (0, 255, 0),    # >= 0.5
    "yellow": (0, 255, 255), # 0.25–0.5
    "red": (0, 0, 255),      # < 0.25 or NaN
}

_QUALITY_COLORS = {
    "excellent": (0, 255, 0),
    "good": (100, 255, 100),
    "caution": (0, 255, 255),
    "poor": (0, 0, 255),
}


def _confidence_color(score: float) -> tuple:
    if score >= 0.5:
        return _JOINT_CONFIDENCE_COLORS["green"]
    elif score >= 0.25:
        return _JOINT_CONFIDENCE_COLORS["yellow"]
    else:
        return _JOINT_CONFIDENCE_COLORS["red"]


def draw_debug_overlay(
    frame: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    bbox: Optional[np.ndarray],
    backend: str,
    frame_number: int,
    lower_limb_quality: float,
    quality_grade: str,
    show_warning: bool = False,
) -> np.ndarray:
    """Draw pose skeleton with confidence coloring and debug info overlay.

    Parameters
    ----------
    frame : ndarray
        BGR source frame.
    keypoints : ndarray, shape (17, 2)
        Pixel coordinates.
    scores : ndarray, shape (17,)
        Per-joint confidence 0..1.
    bbox : ndarray or None
        xyxy bounding box.
    backend : str
        Backend name string for label.
    frame_number : int
        Frame index for label.
    lower_limb_quality : float
        Current frame lower-limb completeness ratio.
    quality_grade : str
        Session quality grade for banner.
    show_warning : bool
        If True, draw a red warning banner.

    Returns
    -------
    ndarray
        BGR frame with overlay drawn.
    """
    out = frame.copy()
    h, w = frame.shape[:2]

    # Bbox
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (200, 200, 200), 1)

    # Skeleton lines
    for a, b, color in _SKELETON:
        if not (np.isnan(keypoints[a, 0]) or np.isnan(keypoints[b, 0])):
            pt_a = (int(keypoints[a, 0]), int(keypoints[a, 1]))
            pt_b = (int(keypoints[b, 0]), int(keypoints[b, 1]))
            if 0 <= pt_a[0] < w and 0 <= pt_a[1] < h and 0 <= pt_b[0] < w and 0 <= pt_b[1] < h:
                cv2.line(out, pt_a, pt_b, color, 2)

    # Joint circles
    for i in range(17):
        x, y = keypoints[i]
        if np.isnan(x) or np.isnan(y):
            continue
        ix, iy = int(x), int(y)
        if not (0 <= ix < w and 0 <= iy < h):
            continue
        color = _confidence_color(float(scores[i]))
        radius = 3 if i < 5 else 5
        cv2.circle(out, (ix, iy), radius, color, -1)

    # Top-left info text
    info_lines = [
        f"Frame: {frame_number}",
        f"Backend: {backend}",
        f"LL Quality: {lower_limb_quality:.2f}",
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(
            out, line, (8, 20 + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    # Bottom-right quality badge
    grade_color = _QUALITY_COLORS.get(quality_grade, (255, 255, 255))
    label = f"Quality: {quality_grade}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    bg_x1, bg_y1 = w - tw - 16, h - th - 16
    bg_x2, bg_y2 = w, h
    cv2.rectangle(out, (bg_x1, bg_y1), (bg_x2, bg_y2), (40, 40, 40), -1)
    cv2.putText(
        out, label, (bg_x1 + 4, bg_y2 - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, grade_color, 2,
    )

    # Warning banner
    if show_warning:
        banner_h = 30
        cv2.rectangle(out, (0, 0), (w, banner_h), (0, 0, 200), -1)
        cv2.putText(
            out, "POOR TRACKING — VERIFY OUTPUT MANUALLY",
            (w // 2 - 240, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

    return out


def write_debug_video(
    frames: list[np.ndarray],
    pose_detections: list[tuple[np.ndarray, np.ndarray, object, str]],
    output_path: str,
    fps: float = 30.0,
    frame_qualities: list = None,
    quality_grade: str = "good",
) -> None:
    """Write a debug overlay MP4 from a list of frames and pose results.

    Parameters
    ----------
    frames : list of BGR frames
    pose_detections : list of (keypoints, scores, bbox, backend) tuples
    output_path : str
    fps : float
    frame_qualities : list of FrameQuality, optional
    quality_grade : str
        Overall session quality grade.
    """
    if not frames:
        return

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for i, (frame, PD) in enumerate(zip(frames, pose_detections)):
        keypoints, scores, bbox, backend = PD

        ll_quality = 0.0
        show_warning = False
        if frame_qualities is not None and i < len(frame_qualities):
            fq = frame_qualities[i]
            ll_quality = (fq.left_ll_ratio + fq.right_ll_ratio) / 2.0
            show_warning = fq.low_ll_confidence and quality_grade in ("caution", "poor")

        annotated = draw_debug_overlay(
            frame, keypoints, scores, bbox, backend,
            frame_number=i,
            lower_limb_quality=ll_quality,
            quality_grade=quality_grade,
            show_warning=show_warning,
        )
        writer.write(annotated)

    writer.release()
