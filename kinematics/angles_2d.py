"""2D sagittal-plane angle computation from single-camera keypoints.

For use when only one camera is available — angles are computed directly
from the 2D image coordinates treating the image plane as the sagittal plane.

This is clinically valid for flexion/extension angles when the camera
is positioned perpendicular to the walking direction (true lateral view).

Outputs are suitable for screening and gross assessment only — not for
clinical publication or research-grade gait analysis.
"""

from __future__ import annotations

import numpy as np


# -----------------------------------------------------------------------------
# COCO joint indices (same as keypoints/detect.py)
# -----------------------------------------------------------------------------
J = {
    "nose": 0,
    "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}


def _safe_vec2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = b - a
    if np.isnan(a).any() or np.isnan(b).any():
        return np.full_like(out, np.nan)
    return out


def _angle2(ba: np.ndarray, bc: np.ndarray) -> float:
    if np.any(np.isnan(ba)) or np.any(np.isnan(bc)):
        return np.nan
    na = np.linalg.norm(ba)
    nc = np.linalg.norm(bc)
    if na < 1e-9 or nc < 1e-9:
        return np.nan
    cos = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def hip_angle_2d(
    keypoints_seq: list[np.ndarray],   # list of (17, 2) — one per frame
    side: str = "left",
) -> np.ndarray:
    """Hip flexion/extension angle from 2D keypoints.

    Camera must be in the sagittal plane (side view).
    Uses image x (horizontal) and y (vertical) as the sagittal plane.
    """
    hip_idx  = J[f"{side}_hip"]
    knee_idx = J[f"{side}_knee"]
    l_sh_idx = J["left_shoulder"]
    r_sh_idx = J["right_shoulder"]

    n = len(keypoints_seq)
    angles = np.full(n, np.nan)

    for f, kp in enumerate(keypoints_seq):
        hip  = kp[hip_idx]
        knee = kp[knee_idx]
        l_sh = kp[l_sh_idx]
        r_sh = kp[r_sh_idx]

        thigh = _safe_vec2(knee, hip)
        trunk = _safe_vec2(hip, (l_sh + r_sh) / 2)
        angles[f] = _angle2(thigh, trunk)

    return angles


def knee_angle_2d(
    keypoints_seq: list[np.ndarray],
    side: str = "left",
) -> np.ndarray:
    """Knee flexion/extension from 2D keypoints."""
    hip_idx   = J[f"{side}_hip"]
    knee_idx  = J[f"{side}_knee"]
    ankle_idx = J[f"{side}_ankle"]

    n = len(keypoints_seq)
    angles = np.full(n, np.nan)

    for f, kp in enumerate(keypoints_seq):
        hip   = kp[hip_idx]
        knee  = kp[knee_idx]
        ankle = kp[ankle_idx]

        thigh = _safe_vec2(hip, knee)
        shank = _safe_vec2(knee, ankle)
        angles[f] = _angle2(thigh, shank)

    return angles


def ankle_angle_2d(
    keypoints_seq: list[np.ndarray],
    side: str = "left",
) -> np.ndarray:
    """Ankle dorsi/plantar flexion from 2D keypoints.

    Uses shank vector vs. vertical (gravity) as a proxy when no toe landmark
    is available (YOLO-Pose COCO-17 has no toe/heel indices beyond ankle).
    """
    knee_idx  = J[f"{side}_knee"]
    ankle_idx = J[f"{side}_ankle"]

    n = len(keypoints_seq)
    angles = np.full(n, np.nan)

    for f, kp in enumerate(keypoints_seq):
        knee  = kp[knee_idx]
        ankle = kp[ankle_idx]

        shank = _safe_vec2(knee, ankle)
        if shank is None:
            continue

        # Vertical proxy: angle between shank and downward gravity vector (0, 1)
        # This gives a simplified dorsi/plantar-flexion proxy without needing toe
        shank_len = np.linalg.norm(shank)
        if shank_len < 1e-6:
            continue
        shank_unit = shank / shank_len
        # Vertical down: (0, 1)
        vertical = np.array([0.0, 1.0])
        cos_a = np.clip(np.dot(shank_unit, vertical), -1.0, 1.0)
        angles[f] = np.arccos(cos_a) * 180.0 / np.pi

    return angles


def estimate_gait_speed_2d(
    keypoints_seq: list[np.ndarray],
    pixel_to_m: float = None,
    focal_length_px: float = None,
    subject_distance_m: float = None,
    side: str = "left",
) -> tuple[float, np.ndarray]:
    """Estimate walking speed from 2D ankle displacement.

    Requires either:
      - pixel_to_m: known pixels-per-metre at the subject's distance
      - OR focal_length_px + subject_distance_m: use pinhole model

    pixel_to_m = focal_px / subject_distance_m

    Returns:
        (speed_m_s, ankle_x_displacement_per_frame)
    """
    ankle_idx = J[f"{side}_ankle"]

    ankle_x = np.array([
        kp[ankle_idx, 0] if not np.isnan(kp[ankle_idx, 0]) else np.nan
        for kp in keypoints_seq
    ])

    if pixel_to_m is None and focal_length_px is not None and subject_distance_m is not None:
        pixel_to_m = focal_length_px / subject_distance_m

    if pixel_to_m is None:
        return np.nan, ankle_x

    ankle_x_m = ankle_x / pixel_to_m
    displacement_m = np.abs(np.diff(ankle_x_m))

    # Approximate speed as mean forward velocity between steps
    valid = displacement_m[~np.isnan(displacement_m)]
    if len(valid) == 0:
        return np.nan, ankle_x_m

    # Use median to be robust against step detection noise
    mean_velocity = np.nanmedian(valid)
    return float(mean_velocity), ankle_x_m


def compute_all_angles_2d(
    keypoints_seq: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute all sagittal-plane angles from 2D keypoint sequences."""
    return {
        "left_hip_angle":  hip_angle_2d(keypoints_seq, "left"),
        "right_hip_angle": hip_angle_2d(keypoints_seq, "right"),
        "left_knee_angle":  knee_angle_2d(keypoints_seq, "left"),
        "right_knee_angle": knee_angle_2d(keypoints_seq, "right"),
        "left_ankle_angle":  ankle_angle_2d(keypoints_seq, "left"),
        "right_ankle_angle": ankle_angle_2d(keypoints_seq, "right"),
    }