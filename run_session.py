"""Movement Analysis pipeline — dual/single-camera markerless movement analysis.

Modes:
  dual   — Two sync'd cameras → pose 2D → DLT triangulation → 3D angles + C3D
  single — One camera (sagittal view) → pose 2D → 2D sagittal angles → CSV/JSON only

Supports: gait, squats, step-downs, jump-downs, lunges, sit-to-stand, balance tasks,
and any other clinical functional movements.

A session consists of:
  1. Load calibration (projection matrices P_l, P_r for dual; intrinsics for single)
  2. Extract frames from video(s)
  3. Run pose detection on each frame (or frame pair)
  4. (dual only) Triangulate to 3D trajectories
  5. Smooth/interpolate
  6. Compute joint angles
  7. Export C3D + CSV + report JSON (dual) or CSV + report JSON (single)
  8. Export tracking_quality.json and optional debug overlay video
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2

from keypoints.detect import detect_keypoints, detect_pose, JOINT_NAMES, PoseDetection
from geometry.triangulate import (
    load_calibration,
    triangulate_frame,
    smooth_trajectories,
    fill_gaps,
)
from kinematics.angles import compute_all_angles as compute_angles_3d, angle_summary
from kinematics.angles_2d import compute_all_angles_2d, estimate_gait_speed_2d
from kinematics.export import (
    export_c3d,
    export_angles_csv,
    export_trajectories_npz,
    export_report,
)
from keypoints.quality import (
    assess_frame,
    assess_session,
    assessment_to_dict,
    FrameQuality,
    CRITICAL_JOINTS,
)
from keypoints.visualize import write_debug_video

logger = logging.getLogger("movement_analysis")


# =============================================================================
# Frame extraction — shared
# =============================================================================

def extract_frames_single(video_path: str, target_fps: float = 120) -> tuple[list, list, float]:
    """Extract frames from a single video at ~target_fps.

    Returns:
        frames: list of (frame_idx, image)
        frame_indices: list of original frame indices
        actual_fps: detected video FPS
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(round(fps / target_fps)))

    frames = []
    extracted = []

    for fi in range(0, n_total, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append((fi, frame))
        extracted.append(fi)

    cap.release()
    return frames, extracted, fps


def extract_frames_dual(
    video_l: str, video_r: str, target_fps: float = 120
) -> tuple[list, list, list, float]:
    """Extract synchronised frames from two videos.

    Returns:
        frame_indices, frames_l, frames_r, actual_fps
    """
    cap_l = cv2.VideoCapture(video_l)
    cap_r = cv2.VideoCapture(video_r)

    fps_l = cap_l.get(cv2.CAP_PROP_FPS)
    fps_r = cap_r.get(cv2.CAP_PROP_FPS)
    fps = min(fps_l, fps_r)

    n_frames = int(min(
        cap_l.get(cv2.CAP_PROP_FRAME_COUNT),
        cap_r.get(cv2.CAP_PROP_FRAME_COUNT),
    ))
    stride = max(1, int(round(fps / target_fps)))

    frames_l = []
    frames_r = []
    extracted = []

    for fi in range(0, n_frames, stride):
        cap_l.set(cv2.CAP_PROP_POS_FRAMES, fi)
        cap_r.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok_l, frame_l = cap_l.read()
        ok_r, frame_r = cap_r.read()
        if not (ok_l and ok_r):
            continue
        frames_l.append((fi, frame_l))
        frames_r.append((fi, frame_r))
        extracted.append(fi)

    cap_l.release()
    cap_r.release()
    return extracted, frames_l, frames_r, fps


# =============================================================================
# Keypoint detection — shared (preserves API + enriches with PoseDetection)
# =============================================================================

def detect_frames_single(
    frames: list,
    conf_threshold: float = 0.3,
) -> tuple[list[np.ndarray], list[PoseDetection]]:
    """Run pose detection on single-camera frames.

    Returns:
        keypoint_arrays: list of (17, 2) keypoint arrays (legacy format)
        pose_detections: list of PoseDetection objects with per-frame metadata
    """
    keypoint_arrays = []
    pose_detections = []
    total = len(frames)

    for i, (fi, frame) in enumerate(frames):
        if i % 10 == 0:
            logger.info(f"  keypoints: {i}/{total}")
        result = detect_pose(frame, conf_threshold)
        if result is None:
            keypoint_arrays.append(np.full((17, 2), np.nan))
            pose_detections.append(None)
        else:
            keypoint_arrays.append(result.keypoints)
            pose_detections.append(result)

    return keypoint_arrays, pose_detections


def detect_frames_dual(
    frames_l, frames_r,
    conf_threshold: float = 0.3,
) -> tuple[list[np.ndarray], list[np.ndarray], list[PoseDetection], list[PoseDetection]]:
    """Run pose detection on dual-camera frame pairs.

    Returns:
        keypoint_arrays_l, keypoint_arrays_r: lists of (17, 2) arrays
        pose_detections_l, pose_detections_r: PoseDetection objects
    """
    kps_l, kps_r = [], []
    pd_l, pd_r = [], []
    total = len(frames_l)

    for i, ((fi, frame_l), (fi_r, frame_r)) in enumerate(zip(frames_l, frames_r)):
        if i % 10 == 0:
            logger.info(f"  keypoints: {i}/{total}")

        res_l = detect_pose(frame_l, conf_threshold)
        res_r = detect_pose(frame_r, conf_threshold)

        if res_l is None:
            kps_l.append(np.full((17, 2), np.nan))
            pd_l.append(None)
        else:
            kps_l.append(res_l.keypoints)
            pd_l.append(res_l)

        if res_r is None:
            kps_r.append(np.full((17, 2), np.nan))
            pd_r.append(None)
        else:
            kps_r.append(res_r.keypoints)
            pd_r.append(res_r)

    return kps_l, kps_r, pd_l, pd_r


# =============================================================================
# 3D triangulation helpers (dual mode only)
# =============================================================================

def triangulate_all(
    kps_l, kps_r,
    P_l, P_r,
    use_ransac: bool = False,
    ransac_thresh: float = 5.0,
):
    """Convert 2D keypoint lists → 3D trajectories (17, 3, n_frames)."""
    n_frames = len(kps_l)
    trajectories = np.full((17, 3, n_frames), np.nan, dtype=np.float64)

    for i in range(n_frames):
        kl = kps_l[i]
        kr = kps_r[i]
        if use_ransac:
            from geometry.triangulate import triangulate_ransac
            _, pt3d = triangulate_ransac(kl, kr, P_l, P_r, inlier_thresh=ransac_thresh)
        else:
            pt3d = triangulate_frame(kl, kr, P_l, P_r)
        trajectories[:, :, i] = pt3d

    return trajectories


# =============================================================================
# 2D smoothing (single mode)
# =============================================================================

def _fill_keypoint_gaps(
    kps: list,
    max_gap: int = 5,
) -> list:
    """Linearly interpolate short gaps in a 2D keypoint sequence."""
    n_frames = len(kps)
    n_joints = kps[0].shape[0]

    out = [[None, None] for _ in range(n_joints)]

    for j in range(n_joints):
        for ax in range(2):
            series = np.array(
                [kps[f][j, ax] if not np.isnan(kps[f][j, ax]) else np.nan for f in range(n_frames)],
                dtype=float
            )
            mask = ~np.isnan(series)
            x = np.arange(n_frames)
            valid_x = x[mask]
            valid_y = series[mask]

            if len(valid_x) < 2:
                out[j][ax] = series
                continue

            interp = series.copy()
            all_nan = np.isnan(series)
            if all_nan.all():
                out[j][ax] = series
                continue

            interp = np.interp(x, valid_x, valid_y)
            out[j][ax] = interp

    result = np.zeros((n_frames, n_joints, 2), dtype=float)
    for j in range(n_joints):
        for ax in range(2):
            result[:, j, ax] = out[j][ax]

    return [result[f] for f in range(n_frames)]


def _smooth_keypoints_2d(
    kps: list,
    window: int = 7,
    fs: float = 120.0,
) -> list:
    """Savitzky-Golay smooth 2D keypoint sequence over time."""
    from scipy.signal import savgol_filter

    n_frames = len(kps)
    n_joints = kps[0].shape[0]

    stacked = np.stack(kps, axis=0)
    smoothed = stacked.copy()
    pad = window // 2

    for j in range(n_joints):
        for ax in range(2):
            series = stacked[:, j, ax].copy()
            mask = ~np.isnan(series)
            x = np.arange(n_frames)
            valid_x = x[mask]
            valid_y = series[mask]

            if len(valid_x) < 4:
                continue

            interp = np.interp(x, valid_x, valid_y)
            padded = np.concatenate([[interp[0]] * pad, interp, [interp[-1]] * pad])

            try:
                filtered = savgol_filter(padded, window, min(2, window - 1))
                smoothed[:, j, ax] = filtered[pad:-pad]
            except Exception:
                smoothed[:, j, ax] = interp

    return [smoothed[f] for f in range(n_frames)]


# =============================================================================
# Quality tracking helpers
# =============================================================================

def _compute_quality_frame(
    keypoints: np.ndarray,
    scores: np.ndarray,
    frame_shape,
) -> dict:
    """Produce a JSON-serializable frame quality dict for tracking_quality.json."""
    from keypoints.quality import assess_frame as _assess

    fq = _assess(keypoints, scores, frame_shape)
    return {
        "visible_ratio": round(fq.visible_ratio, 4),
        "trunk_ratio": round(fq.trunk_ratio, 4),
        "left_ll_ratio": round(fq.left_ll_ratio, 4),
        "right_ll_ratio": round(fq.right_ll_ratio, 4),
        "mean_ll_confidence": round(fq.mean_ll_confidence, 4),
        "bbox_area_pct": round(fq.bbox_area_pct, 2),
        "feet_outside_frame": fq.feet_outside_frame,
        "low_ll_confidence": fq.low_ll_confidence,
    }


def _quality_report_path(output_dir: Path) -> Path:
    return output_dir / "tracking_quality.json"


def _run_quality_assessment(
    pose_detections: list[PoseDetection | None],
    raw_frames: list[np.ndarray],
    output_dir: Path,
    debug_overlay: bool = False,
    fps: float = 30.0,
) -> dict:
    """Run quality assessment and write tracking_quality.json.

    Also writes debug_pose_overlay.mp4 if debug_overlay=True.
    """
    frame_qualities = []
    interpolation_counts = {i: 0 for i in range(17)}

    for i, pd in enumerate(pose_detections):
        if pd is None or pd.keypoints is None:
            # All-NaN frame
            shape = raw_frames[i].shape if i < len(raw_frames) else (720, 1280, 3)
            fake_kps = np.full((17, 2), np.nan)
            fake_scores = np.zeros(17)
            fq = assess_frame(fake_kps, fake_scores, shape)
        else:
            fq = assess_frame(pd.keypoints, pd.scores, pd.keypoints.shape)
            # Count interpolated joints (score == 0 means NaN'd due to confidence)
            for j in range(17):
                if pd.scores[j] == 0 and not np.isnan(pd.keypoints[j, 0]):
                    interpolation_counts[j] += 1

        frame_qualities.append(fq)

    # Longest missing streak per joint
    total_frames = len(pose_detections)
    assessment = assess_session(frame_qualities, interpolation_counts, total_frames)
    assessment_dict = assessment_to_dict(assessment)

    # Per-frame details (JSON-serializable)
    assessment_dict["frame_details"] = [
        {
            "frame": i,
            "visible_ratio": round(fq.visible_ratio, 4),
            "trunk_ratio": round(fq.trunk_ratio, 4),
            "left_ll_ratio": round(fq.left_ll_ratio, 4),
            "right_ll_ratio": round(fq.right_ll_ratio, 4),
            "mean_ll_confidence": round(fq.mean_ll_confidence, 4),
            "bbox_area_pct": round(fq.bbox_area_pct, 2),
            "feet_outside_frame": fq.feet_outside_frame,
            "low_ll_confidence": fq.low_ll_confidence,
        }
        for i, fq in enumerate(frame_qualities)
    ]

    # Write JSON
    report_path = _quality_report_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(assessment_dict, f, indent=2)
    logger.info(f"  Tracking quality: {assessment.quality_grade} — {report_path}")

    # Debug overlay video
    if debug_overlay:
        debug_path = str(output_dir / "debug_pose_overlay.mp4")
        logger.info(f"  Writing debug overlay: {debug_path}")

        # Build pose_detection tuples for write_debug_video
        pd_tuples = []
        for pd in pose_detections:
            if pd is None:
                fake = PoseDetection(
                    keypoints=np.full((17, 2), np.nan),
                    scores=np.zeros(17),
                    bbox=None,
                    bbox_score=None,
                    backend="none",
                )
                pd_tuples.append((fake.keypoints, fake.scores, fake.bbox, fake.backend))
            else:
                pd_tuples.append((pd.keypoints, pd.scores, pd.bbox, pd.backend))

        write_debug_video(
            frames=raw_frames,
            pose_detections=pd_tuples,
            output_path=debug_path,
            fps=fps,
            frame_qualities=frame_qualities,
            quality_grade=assessment.quality_grade,
        )
        logger.info(f"  Debug overlay written: {debug_path}")

    return assessment_dict


# =============================================================================
# Single-camera session
# =============================================================================

def run_session_single(
    session_dir: Path,
    calibration_path: str,
    video: str,
    fps: float = None,
    conf_threshold: float = 0.3,
    smooth_window: int = 7,
    output_dir: Path = None,
    debug_overlay: bool = False,
) -> dict:
    """Run single-camera sagittal-plane movement analysis.

    No 3D reconstruction. Outputs CSV + report JSON + tracking_quality.json
    + optional skeleton overlay + optional debug overlay video.
    """
    session_id = session_dir.name
    start = datetime.now()
    logger.info(f"=== Movement Analysis session (single): {session_id} ===")

    # ── 1. Load calibration (intrinsics only) ─────────────────────────────────
    logger.info("Loading calibration (intrinsics)...")
    try:
        P_l, P_r, mtx_l, mtx_r, Q = load_calibration(calibration_path)
    except Exception:
        logger.warning("No calibration file — proceeding without camera parameters")
        mtx_l = None

    # ── 2. Extract frames ────────────────────────────────────────────────────
    logger.info("Extracting frames...")
    frames, frame_indices, actual_fps = extract_frames_single(video, target_fps=120)
    n_frames = len(frames)
    if n_frames == 0:
        raise RuntimeError("No frames extracted — check video file")
    logger.info(f"  {n_frames} frames at {actual_fps:.1f} fps")

    # ── 3. 2D keypoint detection ─────────────────────────────────────────────
    logger.info("Running pose detection...")
    kps, pose_detections = detect_frames_single(frames, conf_threshold)
    logger.info(f"  Detection complete: {n_frames} frames")

    raw_frames = [f for _, f in frames]

    # ── 3b. Tracking quality assessment ─────────────────────────────────────
    logger.info("Assessing tracking quality...")
    output_dir = output_dir or session_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    quality_dict = _run_quality_assessment(
        pose_detections, raw_frames, output_dir,
        debug_overlay=debug_overlay, fps=actual_fps,
    )
    quality_grade = quality_dict.get("quality_grade", "unknown")

    # ── 4. Interpolate gaps + smooth ────────────────────────────────────────
    kps_filled = _fill_keypoint_gaps(kps, max_gap=5)
    kps_smooth = _smooth_keypoints_2d(kps_filled, window=smooth_window, fs=actual_fps)
    logger.info("  Smoothing complete")

    # ── 4b. Render skeleton video ─────────────────────────────────────────────
    logger.info("  Rendering skeleton video...")
    from utils.visualize import visualize_keypoints
    skeleton_path = str(output_dir / "skeleton.mp4")
    visualize_keypoints(raw_frames, kps_smooth, skeleton_path, fps=actual_fps)
    logger.info(f"  Skeleton video: {skeleton_path}")

    # ── 5. Compute 2D sagittal angles ────────────────────────────────────────
    logger.info("Computing sagittal-plane joint angles...")
    angles = compute_all_angles_2d(kps_smooth)

    if mtx_l is not None:
        speed, _ = estimate_gait_speed_2d(
            kps_smooth,
            focal_length_px=mtx_l[0, 0],
            subject_distance_m=3.0,
        )
    else:
        speed = None

    angle_sum = angle_summary(angles)
    logger.info("  Angles computed: hip/knee/ankle (sagittal)")

    # ── 6. Export outputs ────────────────────────────────────────────────────
    logger.info("Exporting outputs...")
    csv_path = export_angles_csv(angles, str(output_dir / "angles.csv"))
    report_path = _export_report_single(
        angles, speed, actual_fps, n_frames, quality_grade, str(output_dir / "report.json")
    )

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"=== Session complete in {elapsed:.1f}s ===")
    logger.info(f"  CSV: {csv_path}")
    logger.info(f"  JSON: {report_path}")
    logger.info(f"  Quality grade: {quality_grade}")

    return {
        "status": "success",
        "mode": "single",
        "session_id": session_id,
        "duration_s": round(elapsed, 1),
        "n_frames": n_frames,
        "fps": round(actual_fps, 1),
        "outputs": {
            "csv": csv_path,
            "report_json": report_path,
            "skeleton_mp4": skeleton_path,
            "tracking_quality_json": str(_quality_report_path(output_dir)),
        },
        "gait_metrics": angle_sum,
        "tracking_quality_grade": quality_grade,
        "clinical_warnings": quality_dict.get("clinical_warnings", []),
    }


def _export_report_single(
    angles: dict,
    speed_m_s: float | None,
    fps: float,
    n_frames: int,
    quality_grade: str,
    output_path: str,
) -> str:
    """Export single-camera movement analysis report JSON."""
    duration_s = n_frames / fps

    def count_steps(arr):
        vel = np.diff(arr)
        signs = np.sign(vel)
        changes = np.diff(signs)
        return int(np.sum(changes != 0))

    n_left_steps = 0
    n_right_steps = 0
    if "left_ankle_angle" in angles:
        n_left_steps = count_steps(angles["left_ankle_angle"])
    if "right_ankle_angle" in angles:
        n_right_steps = count_steps(angles["right_ankle_angle"])
    total_steps = max(n_left_steps, n_right_steps, 1)

    def rom(arr):
        valid = arr[~np.isnan(arr)]
        if len(valid) < 2:
            return None
        return round(float(valid.max() - valid.min()), 2)

    report = {
        "mode": "single-camera (2D sagittal, screening only)",
        "session": {
            "fps": fps,
            "duration_s": round(duration_s, 2),
            "n_frames": n_frames,
        },
        "tracking_quality_grade": quality_grade,
        "note": "No 3D reconstruction — use dual-camera mode for clinical-grade metrics",
        "gait_metrics": {
            "speed_m_s": speed_m_s,
            "cadence_steps_min": round((total_steps / duration_s) * 60, 1) if duration_s > 0 else None,
            "n_steps": int(total_steps),
        },
        "left_leg": {
            "hip_rom_deg": rom(angles.get("left_hip_angle", np.array([]))),
            "knee_rom_deg": rom(angles.get("left_knee_angle", np.array([]))),
            "ankle_rom_deg": rom(angles.get("left_ankle_angle", np.array([]))),
        },
        "right_leg": {
            "hip_rom_deg": rom(angles.get("right_hip_angle", np.array([]))),
            "knee_rom_deg": rom(angles.get("right_knee_angle", np.array([]))),
            "ankle_rom_deg": rom(angles.get("right_ankle_angle", np.array([]))),
        },
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    return str(Path(output_path).resolve())


# =============================================================================
# Dual-camera session
# =============================================================================

def run_session_dual(
    session_dir: Path,
    calibration_path: str,
    video_l: str,
    video_r: str,
    fps: float = None,
    conf_threshold: float = 0.3,
    smooth_window: int = 7,
    output_dir: Path = None,
    debug_overlay: bool = False,
) -> dict:
    """Run full dual-camera 3D movement analysis."""
    session_id = session_dir.name
    start = datetime.now()
    logger.info(f"=== Movement Analysis session (dual): {session_id} ===")

    # ── 1. Load calibration ──────────────────────────────────────────────────
    logger.info("Loading calibration...")
    P_l, P_r, mtx_l, mtx_r, Q = load_calibration(calibration_path)
    logger.info(f"  Left camera: focal {mtx_l[0,0]:.0f}, principal {mtx_l[0,2]:.1f},{mtx_l[1,2]:.1f}")
    logger.info(f"  Right camera: focal {mtx_r[0,0]:.0f}")
    logger.info(f"  Baseline: |T| = {np.linalg.norm(P_r[:,3]):.3f} m")

    # ── 2. Extract frames ────────────────────────────────────────────────────
    logger.info("Extracting frames...")
    frame_indices, frames_l, frames_r, actual_fps = extract_frames_dual(video_l, video_r, target_fps=120)
    n_frames = len(frame_indices)
    if n_frames == 0:
        raise RuntimeError("No frames extracted — check video files")
    logger.info(f"  {n_frames} frames at {actual_fps:.1f} fps")

    # ── 3. 2D keypoint detection ─────────────────────────────────────────────
    logger.info("Running pose detection...")
    kps_l, kps_r, pd_l, pd_r = detect_frames_dual(frames_l, frames_r, conf_threshold)
    logger.info(f"  Detection complete: {n_frames} frame pairs")

    raw_frames_l = [f for _, f in frames_l]
    raw_frames_r = [f for _, f in frames_r]

    # ── 3b. Tracking quality assessment ─────────────────────────────────────
    # Use left camera frames as reference for quality
    logger.info("Assessing tracking quality...")
    if output_dir is None:
        output_dir = session_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    quality_dict = _run_quality_assessment(
        pd_l, raw_frames_l, output_dir,
        debug_overlay=debug_overlay, fps=actual_fps,
    )
    quality_grade = quality_dict.get("quality_grade", "unknown")

    # ── 4. Triangulation to 3D ──────────────────────────────────────────────
    logger.info("Triangulating 3D trajectories...")
    trajectories_raw = triangulate_all(kps_l, kps_r, P_l, P_r)
    logger.info(f"  Raw trajectories: {n_frames} frames")

    # ── 5. Fill gaps + smooth ────────────────────────────────────────────────
    logger.info("Interpolating gaps and smoothing...")
    trajectories_filled = fill_gaps(trajectories_raw, max_gap=5)
    trajectories_smooth = smooth_trajectories(trajectories_filled, window=smooth_window, fs=actual_fps)
    logger.info("  Smoothing complete")

    # ── 6. Compute joint angles ──────────────────────────────────────────────
    logger.info("Computing joint angles...")
    angles = compute_angles_3d(trajectories_smooth)
    angle_sum = angle_summary(angles)
    logger.info("  Angles computed: hip/knee/ankle/foot-progression")

    # ── 7. Export outputs ────────────────────────────────────────────────────
    logger.info("Exporting outputs...")
    c3d_path = export_c3d(trajectories_smooth, actual_fps, str(output_dir / "session.c3d"))
    npz_path = export_trajectories_npz(trajectories_smooth, str(output_dir / "trajectories_3d.npz"))
    csv_path = export_angles_csv(angles, str(output_dir / "angles.csv"))
    json_path = export_report(trajectories_smooth, angles, actual_fps, str(output_dir / "report.json"))

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"=== Session complete in {elapsed:.1f}s ===")
    logger.info(f"  C3D: {c3d_path}")
    logger.info(f"  CSV: {csv_path}")
    logger.info(f"  JSON: {json_path}")
    logger.info(f"  Quality grade: {quality_grade}")

    return {
        "status": "success",
        "mode": "dual",
        "session_id": session_id,
        "duration_s": round(elapsed, 1),
        "n_frames": n_frames,
        "fps": round(actual_fps, 1),
        "outputs": {
            "c3d": c3d_path,
            "npz": npz_path,
            "csv": csv_path,
            "report_json": json_path,
            "tracking_quality_json": str(_quality_report_path(output_dir)),
        },
        "gait_metrics": angle_sum,
        "tracking_quality_grade": quality_grade,
        "clinical_warnings": quality_dict.get("clinical_warnings", []),
    }


# =============================================================================
# Unified run_session dispatcher
# =============================================================================

def run_session(
    session_dir: Path,
    calibration_path: str,
    video_l: str = None,
    video_r: str = None,
    mode: str = "dual",
    fps: float = None,
    conf_threshold: float = 0.3,
    smooth_window: int = 7,
    output_dir: Path = None,
    debug_overlay: bool = False,
) -> dict:
    """Unified entry point — dispatches to single or dual based on mode."""
    if mode == "single":
        if video_l is None:
            raise ValueError("Single-camera mode requires --video (--left)")
        return run_session_single(
            session_dir=session_dir,
            calibration_path=calibration_path,
            video=video_l,
            fps=fps,
            conf_threshold=conf_threshold,
            smooth_window=smooth_window,
            output_dir=output_dir,
            debug_overlay=debug_overlay,
        )
    elif mode == "dual":
        if video_l is None or video_r is None:
            raise ValueError("Dual-camera mode requires --left and --right videos")
        return run_session_dual(
            session_dir=session_dir,
            calibration_path=calibration_path,
            video_l=video_l,
            video_r=video_r,
            fps=fps,
            conf_threshold=conf_threshold,
            smooth_window=smooth_window,
            output_dir=output_dir,
            debug_overlay=debug_overlay,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'single' or 'dual'.")


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Movement Analysis — markerless dual/single-camera "
                    "biomechanics pipeline"
    )
    p.add_argument("--session", required=True, help="Session directory")
    p.add_argument("--calibration", required=True, help="Calibration YAML file")
    p.add_argument(
        "--mode", default="dual", choices=["single", "dual"],
        help="'single' = one camera (sagittal 2D); 'dual' = two cameras (3D)"
    )
    p.add_argument("--video", default=None,
                   help="[single mode] Path to video file")
    p.add_argument("--left", dest="video_l", default=None,
                   help="[dual mode] Left camera video")
    p.add_argument("--right", dest="video_r", default=None,
                   help="[dual mode] Right camera video")
    p.add_argument("--fps", type=float, default=None, help="Override FPS")
    p.add_argument("--conf-threshold", type=float, default=0.3,
                   help="Min keypoint confidence 0-1")
    p.add_argument("--smooth-window", type=int, default=7,
                   help="SG smooth window (odd integer)")
    p.add_argument("--output", default=None, help="Output directory override")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING"])

    # ── Pose backend args ──────────────────────────────────────────────────────
    p.add_argument(
        "--pose-backend", default=os.getenv("POSE_BACKEND", "auto"),
        choices=["rtmpose", "yolo", "mediapipe", "auto"],
        help="Pose detection backend (default: auto — RTMPose → YOLO → MediaPipe)"
    )
    p.add_argument(
        "--pose-model-tier", default=os.getenv("POSE_MODEL_TIER", "balanced"),
        choices=["fast", "balanced", "accurate"],
        help="Model accuracy tier (default: balanced)"
    )
    p.add_argument(
        "--pose-device", default=os.getenv("POSE_DEVICE", "cuda:0"),
        help="Device for pose backend (e.g. cuda:0 or cpu)"
    )
    p.add_argument(
        "--debug-overlay", action="store_true",
        help="Generate debug_pose_overlay.mp4 with confidence coloring"
    )
    p.add_argument(
        "--quality-report", action="store_true", default=True,
        help="Produce tracking_quality.json (default: True)"
    )
    return p


# =============================================================================
# main
# =============================================================================

def main():
    args = build_parser().parse_args()

    # Propagate pose backend args → environment variables for _load_backend()
    if args.pose_backend and args.pose_backend != "auto":
        os.environ["POSE_BACKEND"] = args.pose_backend
    if args.pose_model_tier:
        os.environ["POSE_MODEL_TIER"] = args.pose_model_tier
    if args.pose_device:
        os.environ["POSE_DEVICE"] = args.pose_device

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    session_dir = Path(args.session).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output) if args.output else None

    video_l = args.video or args.video_l
    video_r = args.video_r

    if args.mode == "single" and video_l is None:
        raise ValueError("Single mode requires a video file (--video or --left)")

    result = run_session(
        session_dir=session_dir,
        calibration_path=args.calibration,
        video_l=video_l,
        video_r=video_r,
        mode=args.mode,
        fps=args.fps,
        conf_threshold=args.conf_threshold,
        smooth_window=args.smooth_window,
        output_dir=output_dir,
        debug_overlay=args.debug_overlay,
    )

    print("\n=== Session Report ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
