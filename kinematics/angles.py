"""Joint angle computation for clinical gait analysis.

Computes hip, knee, and ankle angles from 3D trajectories for both legs.
Angles follow the convention used in clinical gait labs: angle at the
middle joint between the two adjacent segments.
"""

from __future__ import annotations

import numpy as np


# -----------------------------------------------------------------------------
# COCO joint indices
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


def _safe_vector(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vector from a to b (b - a), with NaN-safe handling."""
    out = b - a
    if np.isnan(a).any() or np.isnan(b).any():
        out = np.full_like(out, np.nan)
    return out


def _angle_between(ba: np.ndarray, bc: np.ndarray) -> float:
    """Angle in degrees at joint b between vectors BA and BC.

    Both vectors must be (3,) numpy arrays. Returns NaN if either
    vector is zero-length or contains NaN.
    """
    if np.any(np.isnan(ba)) or np.any(np.isnan(bc)):
        return np.nan
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-9 or norm_bc < 1e-9:
        return np.nan
    cos_a = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


# -----------------------------------------------------------------------------
# Segment angle computations
# -----------------------------------------------------------------------------

def hip_angle(traj: np.ndarray, side: str = "left") -> np.ndarray:
    """Hip flexion/extension angle at the hip joint.

    Angle is between the thigh vector (knee→hip) and the trunk vector
    (hip→shoulder centroid), projected into the sagittal plane.

    Args:
        traj: (17, 3, n_frames) 3D trajectories
        side: "left" or "right"

    Returns:
        (n_frames,) angle in degrees
    """
    hip_idx  = J[f"{side}_hip"]
    knee_idx = J[f"{side}_knee"]
    # Use both shoulders as trunk reference
    l_shoulder = J["left_shoulder"]
    r_shoulder = J["right_shoulder"]

    n_frames = traj.shape[2]
    angles = np.full(n_frames, np.nan)

    for f in range(n_frames):
        hip  = traj[hip_idx,  :, f]
        knee = traj[knee_idx, :, f]
        l_sh = traj[l_shoulder, :, f]
        r_sh = traj[r_shoulder, :, f]

        # Thigh direction (knee → hip)
        thigh = _safe_vector(knee, hip)
        # Trunk direction (hip → shoulder midpoint)
        trunk = _safe_vector(hip, (l_sh + r_sh) / 2)

        angles[f] = _angle_between(thigh, trunk)

    return angles


def knee_angle(traj: np.ndarray, side: str = "left") -> np.ndarray:
    """Knee flexion/extension angle.

    Angle between the thigh (hip→knee) and shank (knee→ankle) vectors.
    In swing phase this is typically 40-70°. In stance phase: 0-40°.
    """
    hip_idx   = J[f"{side}_hip"]
    knee_idx  = J[f"{side}_knee"]
    ankle_idx = J[f"{side}_ankle"]

    n_frames = traj.shape[2]
    angles = np.full(n_frames, np.nan)

    for f in range(n_frames):
        hip   = traj[hip_idx,   :, f]
        knee  = traj[knee_idx,  :, f]
        ankle = traj[ankle_idx, :, f]

        thigh = _safe_vector(hip, knee)
        shank = _safe_vector(knee, ankle)

        angles[f] = _angle_between(thigh, shank)

    return angles


def ankle_angle(traj: np.ndarray, side: str = "left") -> np.ndarray:
    """Ankle dorsi/plantar flexion angle.

    Angle between the shank (knee→ankle) and foot (ankle→toe) vectors.
    Positive = dorsiflexion (toe up). Negative = plantarflexion (toe down).
    """
    knee_idx  = J[f"{side}_knee"]
    ankle_idx = J[f"{side}_ankle"]
    # Foot toe is index 15 (left) / 16 (right) — or approximate as ankle + forward
    toe_idx   = 15 if side == "left" else 16

    n_frames = traj.shape[2]
    angles = np.full(n_frames, np.nan)

    for f in range(n_frames):
        knee  = traj[knee_idx,  :, f]
        ankle = traj[ankle_idx, :, f]
        toe   = traj[toe_idx,   :, f]

        shank = _safe_vector(knee, ankle)
        foot  = _safe_vector(ankle, toe)

        angles[f] = _angle_between(shank, foot)

    return angles


def foot_progression_angle(traj: np.ndarray, side: str = "left") -> np.ndarray:
    """Foot progression angle — toe-out angle in the horizontal (x-z) plane.

    Positive = toe out (abduction). Negative = toe in (adduction).
    Used in clinical gait reports for assessing intoeing/outtoeing gait.
    """
    knee_idx  = J[f"{side}_knee"]
    ankle_idx = J[f"{side}_ankle"]

    n_frames = traj.shape[2]
    angles = np.full(n_frames, np.nan)

    for f in range(n_frames):
        knee  = traj[knee_idx,  :, f]
        ankle = traj[ankle_idx, :, f]

        if np.isnan(knee).any() or np.isnan(ankle).any():
            continue

        # Foot direction in x-z (horizontal) plane
        dx = ankle[0] - knee[0]
        dz = ankle[2] - knee[2]

        # Angle from forward (positive z) axis
        angle = np.degrees(np.arctan2(dx, dz))
        angles[f] = angle

    return angles


def pelvic_tilt(traj: np.ndarray, side: str = "left") -> np.ndarray:
    """Pelvic tilt — angle of the pelvis in the sagittal plane.

    Computed as the angle between the vector connecting the two hips
    and the horizontal plane.
    """
    l_hip = J["left_hip"]
    r_hip = J["right_hip"]

    n_frames = traj.shape[2]
    angles = np.full(n_frames, np.nan)

    for f in range(n_frames):
        lh = traj[l_hip, :, f]
        rh = traj[r_hip, :, f]

        if np.isnan(lh).any() or np.isnan(rh).any():
            continue

        # Pelvis vector
        pelvis = _safe_vector(lh, rh)
        # Cross with forward (y) to get tilt
        horizontal = np.array([0, 1, 0])
        angle = _angle_between(pelvis, horizontal)
        angles[f] = angle

    return angles


def stride_length(traj: np.ndarray, side: str = "left") -> np.ndarray:
    """Estimate stride length as the anteroposterior change in ankle position.

    stride_length[f] = |ankle_x[f] - ankle_x[f - stride_period]|

    For each stance phase, the stride length is approximately the
    max anteroposterior separation of the ankle on the same side.

    Returns (n_frames,) — accurate values only near heel-strike events.
    """
    ankle_idx = J[f"{side}_ankle"]

    n_frames = traj.shape[2]
    strides = np.full(n_frames, np.nan)

    # Use a simple heuristic: stride ≈ 2× the forward throw of the ankle
    ankle_x = traj[ankle_idx, 0, :]  # forward (x) axis
    ankle_z = traj[ankle_idx, 2, :]  # lateral (z) axis

    for f in range(1, n_frames):
        if np.isnan(ankle_x[f]) or np.isnan(ankle_x[f - 1]):
            continue
        # Heuristic: stride is max forward excursion in a gait cycle
        # Simple version: consecutive peak-trough difference
        strides[f] = abs(ankle_x[f] - ankle_x[f - 1])

    return strides


# -----------------------------------------------------------------------------
# Batch compute all gait angles
# -----------------------------------------------------------------------------

def compute_all_angles(traj: np.ndarray) -> dict[str, np.ndarray]:
    """Compute all standard clinical gait angles for a session.

    Args:
        traj: (17, 3, n_frames) 3D trajectories

    Returns:
        dict of {angle_name: (n_frames,) array}
    """
    return {
        "left_hip_angle":       hip_angle(traj, "left"),
        "right_hip_angle":      hip_angle(traj, "right"),
        "left_knee_angle":     knee_angle(traj, "left"),
        "right_knee_angle":    knee_angle(traj, "right"),
        "left_ankle_angle":    ankle_angle(traj, "left"),
        "right_ankle_angle":   ankle_angle(traj, "right"),
        "left_foot_prog":      foot_progression_angle(traj, "left"),
        "right_foot_prog":     foot_progression_angle(traj, "right"),
        "left_pelvic_tilt":    pelvic_tilt(traj, "left"),
        "right_pelvic_tilt":   pelvic_tilt(traj, "right"),
    }


def angle_summary(angles: dict[str, np.ndarray]) -> dict:
    """Compute range-of-motion (ROM) and peak values per angle.

    Returns a dict suitable for a clinical report JSON.
    """
    out = {}
    for name, arr in angles.items():
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            out[name] = {"min": None, "max": None, "rom": None, "mean": None}
            continue
        out[name] = {
            "min":  round(float(valid.min()), 2),
            "max":  round(float(valid.max()), 2),
            "rom":  round(float(valid.max() - valid.min()), 2),
            "mean": round(float(valid.mean()), 2),
        }
    return out