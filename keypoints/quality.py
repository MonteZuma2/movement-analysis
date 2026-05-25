from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


CRITICAL_JOINTS = {
    "trunk": [5, 6, 11, 12],
    "left_lower_limb": [11, 13, 15],
    "right_lower_limb": [12, 14, 16],
    "all_lower_limb": [11, 12, 13, 14, 15, 16],
}

QUALITY_THRESHOLDS = {
    "excellent": 0.90,
    "good": 0.80,
    "caution": 0.65,
    "poor": 0.50,
}


@dataclass
class FrameQuality:
    visible_ratio: float
    trunk_ratio: float
    left_ll_ratio: float
    right_ll_ratio: float
    mean_ll_confidence: float
    bbox_area_pct: float
    feet_outside_frame: bool
    low_ll_confidence: bool
    joint_scores: np.ndarray  # shape (17,)


@dataclass
class SessionQuality:
    total_frames: int
    usable_frame_ratio: float
    left_ll_usable_ratio: float
    right_ll_usable_ratio: float
    longest_missing_streak: dict
    interpolation_count: dict
    quality_grade: str
    clinical_warnings: list[str]
    frame_details: list[dict] = field(default_factory=list)


def assess_frame(keypoints: np.ndarray, scores: np.ndarray,
                frame_shape) -> FrameQuality:
    """Compute per-frame tracking quality metrics.

    Parameters
    ----------
    keypoints : ndarray, shape (17, 2)
        Pixel coordinates, NaN for unavailable joints.
    scores : ndarray, shape (17,)
        Per-joint confidence in [0, 1].
    frame_shape : tuple
        (height, width, ...) of the source frame.

    Returns
    -------
    FrameQuality
    """
    h, w = frame_shape[:2]
    visible = ~np.isnan(keypoints[:, 0])
    total = 17

    # Ratios for joint groups
    def group_ratio(indices):
        g_visible = visible[indices]
        return g_visible.sum() / len(indices) if len(indices) else 0.0

    trunk_ratio = group_ratio(CRITICAL_JOINTS["trunk"])
    left_ll_ratio = group_ratio(CRITICAL_JOINTS["left_lower_limb"])
    right_ll_ratio = group_ratio(CRITICAL_JOINTS["right_lower_limb"])

    # Mean confidence for lower-limb joints
    ll_idx = CRITICAL_JOINTS["all_lower_limb"]
    mean_ll_confidence = float(np.mean(scores[ll_idx])) if len(ll_idx) else 0.0

    # BBox area as % of frame
    valid_keypoints = keypoints[visible]
    if valid_keypoints.shape[0] > 0:
        x_min, x_max = valid_keypoints[:, 0].min(), valid_keypoints[:, 0].max()
        y_min, y_max = valid_keypoints[:, 1].min(), valid_keypoints[:, 1].max()
        bbox_area_pct = (x_max - x_min) * (y_max - y_min) / max(w * h, 1) * 100
    else:
        bbox_area_pct = 0.0

    # Feet outside frame check (ankle joints near edges)
    feet_outside_frame = False
    if total > 0:
        ankle_indices = [15, 16]
        for ai in ankle_indices:
            if visible[ai]:
                ax, ay = keypoints[ai]
                if ax < 5 or ax > w - 5 or ay < 5 or ay > h - 5:
                    feet_outside_frame = True

    low_ll_confidence = mean_ll_confidence < 0.5

    return FrameQuality(
        visible_ratio=visible.sum() / total,
        trunk_ratio=trunk_ratio,
        left_ll_ratio=left_ll_ratio,
        right_ll_ratio=right_ll_ratio,
        mean_ll_confidence=mean_ll_confidence,
        bbox_area_pct=bbox_area_pct,
        feet_outside_frame=feet_outside_frame,
        low_ll_confidence=low_ll_confidence,
        joint_scores=np.asarray(scores, dtype=np.float64),
    )


def assess_session(frame_qualities: list[FrameQuality],
                   interpolation_counts: dict,
                   total_frames: int) -> SessionQuality:
    """Compute per-session quality summary.

    Parameters
    ----------
    frame_qualities : list of FrameQuality, one per frame
    interpolation_counts : dict[int, int]
        Mapping from joint index to number of interpolated frames.
    total_frames : int
        Total frames processed.

    Returns
    -------
    SessionQuality
    """
    n = len(frame_qualities)
    if n == 0:
        return SessionQuality(
            total_frames=total_frames,
            usable_frame_ratio=0.0,
            left_ll_usable_ratio=0.0,
            right_ll_usable_ratio=0.0,
            longest_missing_streak={},
            interpolation_count=interpolation_counts,
            quality_grade="poor",
            clinical_warnings=["No valid frames detected."],
        )

    # Per-frame usable (>= 4/6 lower-limb joints visible)
    def ll_usable_ratio(side: str) -> float:
        ratio_attr = f"{side}_ll_ratio"
        usable = sum(getattr(fq, ratio_attr) >= 0.667 for fq in frame_qualities)
        return usable / n if n else 0.0

    usable_frame_ratio = sum(
        fq.visible_ratio >= 0.70 for fq in frame_qualities
    ) / n

    left_ll_usable = ll_usable_ratio("left")
    right_ll_usable = ll_usable_ratio("right")

    # Longest missing streak per joint
    longest_streak = {}
    for joint_idx in range(17):
        streak = max_run([1 if fq.joint_scores[joint_idx] > 0 else 0
                          for fq in frame_qualities])
        longest_streak[joint_idx] = streak

    # Quality grade
    mean_ll = np.mean([fq.mean_ll_confidence for fq in frame_qualities])
    critical_usable_pct = (usable_frame_ratio + left_ll_usable + right_ll_usable) / 3.0

    if critical_usable_pct >= QUALITY_THRESHOLDS["excellent"]:
        grade = "excellent"
    elif critical_usable_pct >= QUALITY_THRESHOLDS["good"]:
        grade = "good"
    elif critical_usable_pct >= QUALITY_THRESHOLDS["caution"]:
        grade = "caution"
    else:
        grade = "poor"

    # Clinical warnings
    warnings = []
    if grade in ("caution", "poor"):
        warnings.append(
            f"Tracking quality is '{grade}'. Review outputs before clinical use."
        )
    if left_ll_usable < 0.80:
        warnings.append("Left lower-limb tracking below 80% usable frames.")
    if right_ll_usable < 0.80:
        warnings.append("Right lower-limb tracking below 80% usable frames.")
    if any(streak > 10 for streak in longest_streak.values()):
        warnings.append("One or more joints have gaps > 10 frames — check interpolation.")
    if interpolation_counts:
        total_interp = sum(interpolation_counts.values())
        if total_interp > n * 0.20:
            warnings.append(
                f"More than 20% of frames were interpolated ({total_interp}/{n})."
            )

    return SessionQuality(
        total_frames=total_frames,
        usable_frame_ratio=round(usable_frame_ratio, 4),
        left_ll_usable_ratio=round(left_ll_usable, 4),
        right_ll_usable_ratio=round(right_ll_usable, 4),
        longest_missing_streak=longest_streak,
        interpolation_count=interpolation_counts,
        quality_grade=grade,
        clinical_warnings=warnings,
    )


def max_run(booleans: list[int]) -> int:
    """Return the longest consecutive True run in a boolean list."""
    max_run = cur_run = 0
    for v in booleans:
        cur_run = cur_run + 1 if v else 0
        if cur_run > max_run:
            max_run = cur_run
    return max_run


def assessment_to_dict(assessment: SessionQuality) -> dict:
    """Serialize SessionQuality to a JSON-serializable dict."""
    d = {
        "total_frames": assessment.total_frames,
        "usable_frame_ratio": assessment.usable_frame_ratio,
        "left_ll_usable_ratio": assessment.left_ll_usable_ratio,
        "right_ll_usable_ratio": assessment.right_ll_usable_ratio,
        "longest_missing_streak": assessment.longest_missing_streak,
        "interpolation_count": assessment.interpolation_count,
        "quality_grade": assessment.quality_grade,
        "clinical_warnings": assessment.clinical_warnings,
    }
    return d
