"""C3D export and clinical report generation.

C3D (Coordinate 3) is the standard biomechanics format supported by
Visual3D, Vicon Nexus, Plug-in-Gait, and most clinical gait labs.

This module writes:
  - session.c3d  — full 3D trajectories (markers × frames × axes)
  - angles.csv   — joint angle time series per frame
  - report.json  — ROM summary, cadence, gait speed
"""

from __future__ import annotations

import json
import csv
from pathlib import Path

import numpy as np


# -----------------------------------------------------------------------------
# COCO joint name list (matching keypoints/detect.py)
# -----------------------------------------------------------------------------
JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


# -----------------------------------------------------------------------------
# C3D export via ezc3d
# -----------------------------------------------------------------------------

def export_c3d(
    trajectories: np.ndarray,   # (17, 3, n_frames)
    fps: float,
    output_path: str,
    labels: list[str] = None,
) -> str:
    """Write trajectories to a C3D file using ezc3d.

    Args:
        trajectories: (17, 3, n_frames) 3D positions in metres
        fps: frames per second
        output_path: path to write .c3d file
        labels: optional list of 17 joint names (uses JOINT_NAMES if None)

    Returns:
        Absolute path to the written C3D file.
    """
    try:
        import ezc3d
    except ImportError:
        raise ImportError("ezc3d not installed — run: pip install ezc3d")

    if labels is None:
        labels = JOINT_NAMES

    # ezc3d expects (n_markers, 3, n_frames)  or (3, n_markers, n_frames)?
    # Actually ezc3d c3d['data']['points'] = (4, n_markers, n_frames)
    # where 4 = [x, y, z, residual]
    n_markers = trajectories.shape[0]
    n_frames = trajectories.shape[2]

    # Build the ezc3d structure
    points = np.zeros((4, n_markers, n_frames), dtype=np.float32)
    points[:3, :, :] = trajectories.transpose(1, 0, 2)  # → (3, 17, n_frames)
    points[3, :, :] = 0.0  # residual placeholder

    c3d = ezc3d.c3d()
    c3d["data"] = {"points": points, "labels": labels}
    c3d["header"] = {
        "parameters": {
            "POINT": {
                "RATE": {"values": [fps]},
                "LABELS": {"values": [labels]},
            }
        }
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    c3d.write(output_path)
    return str(Path(output_path).resolve())


# -----------------------------------------------------------------------------
# CSV angle export
# -----------------------------------------------------------------------------

def export_angles_csv(
    angles: dict[str, np.ndarray],   # {name: (n_frames,)}
    output_path: str,
) -> str:
    """Write joint angle time series to a CSV file.

    Args:
        angles: dict of {angle_name: (n_frames,)} arrays
        output_path: target .csv file path

    Returns:
        Absolute path to the written file.
    """
    n_frames = next(iter(angles.values())).shape[0]

    rows = [{"frame": f, **{name: angles[name][f] for name in angles}} for f in range(n_frames)]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", *angles.keys()])
        writer.writeheader()
        writer.writerows(rows)

    return str(Path(output_path).resolve())


# -----------------------------------------------------------------------------
# Trajectory NPZ export (intermediate format)
# -----------------------------------------------------------------------------

def export_trajectories_npz(
    trajectories: np.ndarray,
    output_path: str,
) -> str:
    """Save raw 3D trajectories as a compressed NPZ (used for debugging/reprocessing)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, trajectories=trajectories)
    return str(Path(output_path).resolve())


# -----------------------------------------------------------------------------
# Clinical summary report
# -----------------------------------------------------------------------------

def compute_gait_summary(
    trajectories: np.ndarray,
    angles: dict[str, np.ndarray],
    fps: float,
) -> dict:
    """Compute clinically relevant summary metrics from trajectories + angles.

    Args:
        trajectories: (17, 3, n_frames)
        fps: sampling frequency

    Returns:
        dict with clinically meaningful gait metrics
    """
    n_frames = trajectories.shape[2]
    duration_s = n_frames / fps

    # --- Ankle trajectory for step detection ---
    l_ankle_x = trajectories[15, 0, :]   # left ankle
    r_ankle_x = trajectories[16, 0, :]   # right ankle

    # Simple heuristic: zero-crossings of forward velocity = heel strikes
    l_vel = np.diff(l_ankle_x)
    r_vel = np.diff(r_ankle_x)

    # Count steps (zero crossings of velocity sign change)
    l_steps = np.sum(np.diff(np.sign(l_vel)) != 0)
    r_steps = np.sum(np.diff(np.sign(r_vel)) != 0)
    total_steps = max(l_steps, r_steps, 1)

    # Gait speed: forward progression of the pelvis centre
    l_hip_x = trajectories[11, 0, :]
    r_hip_x = trajectories[12, 0, :]
    pelvis_forward = (l_hip_x + r_hip_x) / 2
    total_forward_distance = pelvis_forward[-1] - pelvis_forward[0]

    gait_speed_mps = total_forward_distance / duration_s if duration_s > 0 else 0

    # Cadence: steps per minute
    cadence = (total_steps / duration_s) * 60 if duration_s > 0 else 0

    # Stride length: total forward distance / number of full gait cycles
    n_cycles = total_steps / 2  # 2 steps per gait cycle
    stride_length = (total_forward_distance / n_cycles) if n_cycles > 0 else 0

    # Foot progression angles — use mean of absolute peak values
    def peak_mean(arr):
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return None
        return round(float(np.mean(np.abs(valid))), 2)

    # Range of motion for each angle
    def rom(arr):
        valid = arr[~np.isnan(arr)]
        if len(valid) < 2:
            return None
        return round(float(valid.max() - valid.min()), 2)

    return {
        "session": {
            "fps": fps,
            "duration_s": round(duration_s, 2),
            "n_frames": n_frames,
        },
        "gait_metrics": {
            "speed_m_s":     round(gait_speed_mps, 3),
            "cadence_steps_min": round(cadence, 1),
            "stride_length_m":  round(stride_length, 3),
            "n_cycles":         round(n_cycles, 1),
        },
        "left_leg": {
            "hip_rom_deg":              rom(angles.get("left_hip_angle", np.array([]))),
            "knee_rom_deg":             rom(angles.get("left_knee_angle", np.array([]))),
            "ankle_rom_deg":            rom(angles.get("left_ankle_angle", np.array([]))),
            "foot_progression_mean_deg": peak_mean(angles.get("left_foot_prog", np.array([]))),
        },
        "right_leg": {
            "hip_rom_deg":              rom(angles.get("right_hip_angle", np.array([]))),
            "knee_rom_deg":             rom(angles.get("right_knee_angle", np.array([]))),
            "ankle_rom_deg":            rom(angles.get("right_ankle_angle", np.array([]))),
            "foot_progression_mean_deg": peak_mean(angles.get("right_foot_prog", np.array([]))),
        },
    }


def export_report(
    trajectories: np.ndarray,
    angles: dict[str, np.ndarray],
    fps: float,
    output_path: str,
) -> str:
    """Write the full clinical report JSON."""
    summary = compute_gait_summary(trajectories, angles, fps)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    return str(Path(output_path).resolve())