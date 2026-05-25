"""Stereo triangulation — convert dual-camera 2D keypoints → 3D trajectories.

Uses Direct Linear Transform (DLT) with RANSAC outlier rejection for robustness
against mismatched keypoints across frames.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import null_space


# -----------------------------------------------------------------------------
# Projection matrices from calibration
# -----------------------------------------------------------------------------

def make_projection_matrix(intrinsic: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 3×4 projection matrix P = K [R|t].

    Args:
        intrinsic: (3, 3) camera matrix K
        R: (3, 3) rotation from world to camera
        t: (3,) translation vector
    Returns:
        (3, 4) projection matrix
    """
    return intrinsic @ np.hstack([R, t.reshape(3, 1)])


def make_reference_projection(intrinsic: np.ndarray) -> np.ndarray:
    """Projection matrix for left camera at origin (identity R, zero t)."""
    return make_projection_matrix(intrinsic, np.eye(3), np.zeros(3))


# -----------------------------------------------------------------------------
# Direct Linear Transform triangulation
# -----------------------------------------------------------------------------

def triangulate_dlt(
    kp_l: np.ndarray,
    kp_r: np.ndarray,
    P_l: np.ndarray,
    P_r: np.ndarray,
) -> np.ndarray:
    """Triangulate a single 3D point from two 2D observations via DLT.

    Args:
        kp_l: (2,) 2D point in left image (x, y)
        kp_r: (2,) 2D point in right image (x, y)
        P_l: (3, 4) projection matrix for left camera
        P_r: (3, 4) projection matrix for right camera

    Returns:
        (3,) 3D point in world coordinates
    """
    x_l, y_l = kp_l
    x_r, y_r = kp_r

    A = np.array([
        x_l * P_l[2, :] - P_l[0, :],
        y_l * P_l[2, :] - P_l[1, :],
        x_r * P_r[2, :] - P_r[0, :],
        y_r * P_r[2, :] - P_r[1, :],
    ])

    # Solve Ax = 0 via SVD — last singular vector is the solution
    _, _, v = np.linalg.svd(A)
    X = v[-1, :]
    return X[:3] / X[3]


def triangulate_frame(
    keypoints_l: np.ndarray,   # (17, 2)
    keypoints_r: np.ndarray,   # (17, 2)
    P_l: np.ndarray,
    P_r: np.ndarray,
) -> np.ndarray:
    """Triangulate all 17 COCO keypoints from one stereo frame.

    Args:
        keypoints_l: (17, 2) 2D keypoints from left camera
        keypoints_r: (17, 2) 2D keypoints from right camera
        P_l, P_r: (3, 4) projection matrices

    Returns:
        (17, 3) 3D keypoint positions in world coordinates
        Points where either observation is NaN are returned as NaN.
    """
    n = keypoints_l.shape[0]
    out = np.full((n, 3), np.nan, dtype=np.float64)

    for i in range(n):
        kl = keypoints_l[i]
        kr = keypoints_r[i]
        if np.isnan(kl[0]) or np.isnan(kr[0]):
            continue
        try:
            out[i] = triangulate_dlt(kl, kr, P_l, P_r)
        except np.linalg.LinAlgError:
            continue

    return out


# -----------------------------------------------------------------------------
# RANSAC outlier rejection for robust triangulation
# -----------------------------------------------------------------------------

def triangulate_ransac(
    keypoints_l: np.ndarray,
    keypoints_r: np.ndarray,
    P_l: np.ndarray,
    P_r: np.ndarray,
    n_iter: int = 100,
    inlier_thresh: float = 5.0,   # pixels re-projection error threshold
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate with RANSAC outlier rejection.

    Returns:
        (inlier_mask, triangulated_points) — mask is (17,) bool.
        Points flagged as outliers are returned as NaN in the output.
    """
    n = keypoints_l.shape[0]
    best_inliers = np.zeros(n, dtype=bool)
    best_mask = np.full(n, np.nan)

    for _ in range(n_iter):
        # Random sample of 5 correspondences to compute hypothesis
        valid = np.where(
            ~np.isnan(keypoints_l[:, 0]) & ~np.isnan(keypoints_r[:, 0])
        )[0]
        if len(valid) < 5:
            break

        choose = np.random.default_rng().choice(valid, size=min(5, len(valid)), replace=False)

        # Triangulate all (no masking yet for hypothesis)
        pts3d = triangulate_frame(keypoints_l, keypoints_r, P_l, P_r)

        # Check inliers by re-projecting and checking error
        inliers = np.zeros(n, dtype=bool)
        for idx in range(n):
            if np.isnan(pts3d[idx]).any():
                continue

            # Project to both images
            Xh = np.append(pts3d[idx], 1)
            xl = P_l @ Xh; xl = xl[:2] / xl[2]
            xr = P_r @ Xh; xr = xr[:2] / xr[2]

            err_l = np.linalg.norm(keypoints_l[idx] - xl)
            err_r = np.linalg.norm(keypoints_r[idx] - xr)
            if max(err_l, err_r) < inlier_thresh:
                inliers[idx] = True

        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_mask = inliers.copy()

    # Final triangulation using only inliers
    pts3d = triangulate_frame(keypoints_l, keypoints_r, P_l, P_r)
    pts3d[~best_mask] = np.nan
    return best_mask, pts3d


# -----------------------------------------------------------------------------
# Temporal filtering — Savitzky-Golay smoothing + outlier interpolation
# -----------------------------------------------------------------------------

def smooth_trajectories(
    trajectories: np.ndarray,   # (17, 3, n_frames)
    window: int = 7,
    poly_order: int = 2,
    fs: float = 120.0,
) -> np.ndarray:
    """Savitzky-Golay filter over time for each keypoint joint.

    Args:
        trajectories: (17, 3, n_frames) 3D positions
        window: SG filter window length (must be odd)
        poly_order: polynomial order (default 2)
        fs: sampling frequency for 0-padding edges

    Returns:
        smoothed (17, 3, n_frames)
    """
    from scipy.signal import savgol_filter

    out = trajectories.copy()
    n_frames = trajectories.shape[2]

    for j in range(17):
        for c in range(3):
            series = trajectories[j, c]
            mask = ~np.isnan(series)

            if mask.sum() < poly_order + 1:
                continue

            # Interpolate gaps before filtering
            x = np.arange(n_frames)
            valid = series[mask]
            valid_x = x[mask]
            if len(valid_x) < 2:
                continue

            interp = np.interp(x, valid_x, valid)
            # Pad edges to avoid border artifacts
            pad = window // 2
            padded = np.concatenate([[interp[0]] * pad, interp, [interp[-1]] * pad])

            try:
                filtered = savgol_filter(padded, window, poly_order)
                out[j, c] = filtered[pad:-pad]
            except Exception:
                out[j, c] = interp

    return out


def fill_gaps(
    trajectories: np.ndarray,  # (17, 3, n_frames)
    max_gap: int = 5,
) -> np.ndarray:
    """Linearly interpolate short gaps in trajectories."""
    out = trajectories.copy()
    n_frames = trajectories.shape[2]

    for j in range(17):
        for c in range(3):
            series = trajectories[j, c]
            mask = np.isnan(series)
            if not mask.any():
                continue

            x = np.arange(n_frames)
            valid = ~mask

            # Fill short gaps
            runs = np.diff(np.concatenate([[False], valid, [False]]).astype(int))
            starts = np.where(runs == 1)[0]
            ends = np.where(runs == -1)[0]

            for s, e in zip(starts, ends):
                if e - s <= max_gap + 1:
                    out[j, c, s:e] = np.interp(
                        x[s:e],
                        [s - 1, e],
                        [series[s - 1], series[e] if e < n_frames else series[s - 1]]
                    )

    return out


# -----------------------------------------------------------------------------
# Load calibration and build projection matrices
# -----------------------------------------------------------------------------

def load_calibration(cal_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a calibration YAML and return projection matrices + intrinsics.

    Returns:
        (P_l, P_r, mtx_l, mtx_r, Q) — projection matrices and intrinsics for downstream use.
    """
    import yaml

    with open(cal_path) as f:
        cal = yaml.safe_load(f)

    mtx_l = np.array(cal["intrinsics_left"]["mtx"])
    mtx_r = np.array(cal["intrinsics_right"]["mtx"])
    dist_l = np.array(cal["intrinsics_left"]["dist"])
    dist_r = np.array(cal["intrinsics_right"]["dist"])
    R = np.array(cal["extrinsics"]["R"])
    T = np.array(cal["extrinsics"]["T"])

    P_l = make_reference_projection(mtx_l)
    P_r = make_projection_matrix(mtx_r, R, T)

    Q = np.array(cal["Q"]) if "Q" in cal else None

    return P_l, P_r, mtx_l, mtx_r, Q