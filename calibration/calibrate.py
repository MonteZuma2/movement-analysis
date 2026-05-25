"""Calibration module — checkerboard detection, camera intrinsic calibration,
and stereo extrinsic (R, T) estimation between two cameras.

Run once before a session; reuse the calibration file across all sessions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


# -----------------------------------------------------------------------------
# COCO 17-keypoint joint definitions (for reference in clinical output)
# -----------------------------------------------------------------------------
JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# Subset used for gait lower-limb analysis
GAIT_JOINTS = {
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}


# -----------------------------------------------------------------------------
# Checkerboard utilities
# -----------------------------------------------------------------------------

def make_object_points(grid_w: int, grid_h: int, square_size: float):
    """N×M checkerboard world coordinates (z=0 plane)."""
    obj = np.zeros((grid_w * grid_h, 3), np.float32)
    obj[:, :2] = np.mgrid[:grid_w, :grid_h].T.reshape(-1, 2) * square_size
    return obj


def find_checkerboard_corners(image: np.ndarray, grid_w: int, grid_h: int):
    """Find inner checkerboard corners (not outer dots)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    ok, corners = cv2.findChessboardCorners(gray, (grid_w, grid_h), cv2.ADAPTIVE_THRESH)
    if not ok:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners.squeeze()


# -----------------------------------------------------------------------------
# Intrinsic calibration (per camera)
# -----------------------------------------------------------------------------

def calibrate_intrinsics(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
):
    """Calibrate a single camera's intrinsic parameters."""
    ok, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    if not ok:
        raise RuntimeError("Intrinsic calibration failed")
    return dict(mtx=mtx.tolist(), dist=dist.tolist(), rvecs=[r.tolist() for r in rvecs], tvecs=[t.tolist() for t in tvecs])


# -----------------------------------------------------------------------------
# Stereo calibration (relative pose between cameras)
# -----------------------------------------------------------------------------

def calibrate_stereo(
    object_points: list[np.ndarray],
    image_points_l: list[np.ndarray],
    image_points_r: list[np.ndarray],
    mtx_l, dist_l, mtx_r, dist_r,
    image_size: tuple[int, int],
):
    """Estimate rotation R and translation T from left camera to right camera."""
    ok, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        object_points, image_points_l, image_points_r,
        mtx_l, dist_l, mtx_r, dist_r,
        image_size,
        flags=cv2.CALIB_FIX_INTRINSICS,  # keep intrinsics fixed
    )
    if not ok:
        raise RuntimeError("Stereo calibration failed")
    return dict(R=R.tolist(), T=T.tolist(), E=E.tolist(), F=F.tolist())


# -----------------------------------------------------------------------------
# Rectification (optional — for epipolar-aligned view)
# -----------------------------------------------------------------------------

def compute_rectification_maps(mtx_l, dist_l, mtx_r, dist_r, R, T, image_size):
    """Compute rectification maps so rows are epipolar-aligned."""
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r,
        image_size, R, T,
        alpha=0,
    )
    map_l1, map_l2 = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, image_size, cv2.CV_16SC2)
    map_r1, map_r2 = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, image_size, cv2.CV_16SC2)
    return (map_l1, map_l2), (map_r1, map_r2), Q


# -----------------------------------------------------------------------------
# Calibration session — capture checkerboard views from both cameras
# -----------------------------------------------------------------------------

class CalibrationSession:
    """Interactive calibration capture from two sync'd cameras."""

    def __init__(
        self,
        left_cam: int | str,
        right_cam: int | str,
        grid_w: int = 9,
        grid_h: int = 6,
        square_size: float = 0.04,
    ):
        self.cap_l = cv2.VideoCapture(left_cam if isinstance(left_cam, int) else str(left_cam))
        self.cap_r = cv2.VideoCapture(right_cam if isinstance(right_cam, int) else str(right_cam))
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.square_size = square_size
        self.obj_points = make_object_points(grid_w, grid_h, square_size)

        self.objlist: list[np.ndarray] = []
        self.imglist_l: list[np.ndarray] = []
        self.imglist_r: list[np.ndarray] = []

    @property
    def image_size(self) -> tuple[int, int]:
        ok, frame = self.cap_l.read()
        if not ok:
            raise RuntimeError("Cannot read from left camera")
        self.cap_l.set(cv2.CAP_PROP_POS_FRAMES, 0)
        h, w = frame.shape[:2]
        return (w, h)

    def capture_frame(self) -> dict | None:
        """Grab one frame from each camera; return (corners_l, corners_r) or None."""
        ok_l, frame_l = self.cap_l.read()
        ok_r, frame_r = self.cap_r.read()
        if not (ok_l and ok_r):
            return None

        corners_l = find_checkerboard_corners(frame_l, self.grid_w, self.grid_h)
        corners_r = find_checkerboard_corners(frame_r, self.grid_w, self.grid_h)

        if corners_l is not None and corners_r is not None:
            self.objlist.append(self.obj_points)
            self.imglist_l.append(corners_l)
            self.imglist_r.append(corners_r)

        return dict(
            left_corners=corners_l,
            right_corners=corners_r,
            left_frame=frame_l,
            right_frame=frame_r,
            n_captured=len(self.imglist_l),
        )

    def calibrate(self) -> dict:
        """Run full calibration: intrinsic × 2, then stereo extrinsic."""
        h, w = self.image_size

        int_l = calibrate_intrinsics(self.objlist, self.imglist_l, (w, h))
        int_r = calibrate_intrinsics(self.objlist, self.imglist_r, (w, h))

        stereo = calibrate_stereo(
            self.objlist, self.imglist_l, self.imglist_r,
            np.array(int_l["mtx"]), np.array(int_l["dist"]),
            np.array(int_r["mtx"]), np.array(int_r["dist"]),
            (w, h),
        )

        (map_l1, map_l2), (map_r1, map_r2), Q = compute_rectification_maps(
            np.array(int_l["mtx"]), np.array(int_l["dist"]),
            np.array(int_r["mtx"]), np.array(int_r["dist"]),
            np.array(stereo["R"]), np.array(stereo["T"]),
            (w, h),
        )

        return dict(
            image_size=(w, h),
            intrinsics_left=int_l,
            intrinsics_right=int_r,
            extrinsics=stereo,
            rectification_map_left=(map_l1.tolist(), map_l2.tolist()),
            rectification_map_right=(map_r1.tolist(), map_r2.tolist()),
            Q=Q.tolist(),
            rms_error=None,  # filled after validation
        )

    def release(self):
        self.cap_l.release()
        self.cap_r.release()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Calibrate dual-camera setup")
    p.add_argument("--left", required=True, help="Left camera index (int) or device path")
    p.add_argument("--right", required=True, help="Right camera index (int) or device path")
    p.add_argument("--output", required=True, help="Output YAML calibration file")
    p.add_argument("--grid-w", type=int, default=9, help="Checkerboard columns")
    p.add_argument("--grid-h", type=int, default=6, help="Checkerboard rows")
    p.add_argument("--square-size", type=float, default=0.04, help="Square size in metres")
    p.add_argument("--n-views", type=int, default=20, help="Min checkerboard views to capture")
    return p


def main():
    args = build_parser().parse_args()

    left_idx = int(args.left) if args.left.isdigit() else args.left
    right_idx = int(args.right) if args.right.isdigit() else args.right

    session = CalibrationSession(left_idx, right_idx, args.grid_w, args.grid_h, args.square_size)
    n = 0

    print(f"Calibration capture — aim for {args.n_views}+ views across different positions")
    print("Press SPACE to capture a view | Q to finish early | ESC to cancel")
    print(f"Views captured: {n}/{args.n_views}")

    while True:
        ok, frame_l = session.cap_l.read()
        _, frame_r = session.cap_r.read()
        if not ok:
            break

        display_l = frame_l.copy()
        display_r = frame_r.copy()

        for i in range(n):
            corners = session.imglist_l[i]
            cv2.drawChessboardCorners(display_l, (args.grid_w, args.grid_h), corners, True)

        cv2.imshow("left (SPACE capture, Q finish, ESC cancel)", display_l)
        cv2.imshow("right", display_r)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            result = session.capture_frame()
            if result["left_corners"] is not None:
                n += 1
                print(f"  captured {n}/{args.n_views}")
            else:
                print("  no checkerboard detected — try repositioning")
        elif key == ord('q') or n >= args.n_views:
            break
        elif key == 27:
            print("Cancelled.")
            session.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()
    session.release()

    if n < 5:
        print(f"Only {n} views captured — insufficient for calibration. Need at least 5.")
        return

    print("Running calibration...")
    cal = session.calibrate()
    print(f"Calibration saved to {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(cal, f)


if __name__ == "__main__":
    main()