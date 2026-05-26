# Movement Analysis — Markerless Dual/Single-Camera Biomechanics Pipeline

**Core capability:** Video → 2D pose keypoints → (Optional stereo triangulation → 3D trajectories) → joint angle reports, C3D files, skeleton overlay videos, and per-session tracking quality assessments.

Supports **dual-camera** (3D, clinical-grade) and **single-camera** (2D sagittal, screening) modes for any moveable subject: **gait, squats, step-downs, jump-downs, lunges, sit-to-stand, balance tasks, and general functional movement**.

```
Dual:   Video × 2 → 2D Pose → Triangulation → 3D Trajectories → Angles → C3D
Single: Video × 1 → 2D Pose → 2D Sagittal Angles + skeleton overlay + quality report
```

**Pose backends (pluggable):**

| Backend | Description | Notes |
|---|---|---|
| **YOLO-Pose** (default) | Ultralytics single-stage; COCO-17 native; full-body tracking including lower limbs | `--pose-backend yolo` |
| **RTMPose** | Two-stage detector-crop-pose; most accurate for difficult movements | `--pose-backend rtmpose` |
| **MediaPipe** | Legacy fallback; automatic if others unavailable | `--pose-backend mediapipe` |

---

## Two modes

### Dual-camera (recommended for clinical work)
- Two sync'd cameras → pose estimation → DLT triangulation → true 3D trajectories
- Outputs: `session.c3d` + `angles.csv` + `trajectories_3d.npz` + `report.json`
- Produces clinical-grade hip/knee/ankle ROM, foot progression, gait speed, cadence

### Single-camera (screening / gross assessment)
- One camera in sagittal (side) view → pose estimation → 2D joint angles + skeleton overlay
- Outputs: `angles.csv` + `report.json` + `skeleton.mp4` + `tracking_quality.json` + optional `debug_pose_overlay.mp4`
- No 3D reconstruction, no C3D export
- Suitable for: screening, telehealth, squats, step-downs, jump landings, lunges

---

## Installation

```bash
cd /home/zuma/Documents/movement_analysis

python -m venv .venv
source .venv/bin/activate

# YOLO-Pose — recommended, easiest install, best lower-limb tracking
pip install ultralytics opencv-python numpy scipy pyyaml ezc3d

# RTMPose — preferred clinical backend (requires torch + mmpose)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install openmim mmengine "mmcv>=2.0" mmdet mmpose

# MediaPipe — legacy fallback
pip install mediapipe opencv-python numpy scipy pyyaml ezc3d
```

---

## Model tiers

Use `--pose-model-tier` to control accuracy/speed tradeoff:

| Flag | YOLO-Pose | RTMPose |
|---|---|---|
| `--pose-model-tier fast` | yolov8n-pose.pt | RTMPose-s (256×192) |
| `--pose-model-tier balanced` (default) | yolov8s-pose.pt | RTMPose-l (256×192) |
| `--pose-model-tier accurate` | yolov8m-pose.pt | RTMPose-x (384×288) |

Device: `--pose-device cpu` (default) or `cuda:0`.

---

## Quick start — single-camera (2D sagittal screening)

```bash
# YOLO-Pose is the recommended backend for full-body tracking
python run_session.py \
    --session sessions/lunge_001 \
    --calibration calibration/intrinsics_demo.yaml \
    --mode single \
    --video "path/to/video.mov" \
    --pose-backend yolo \
    --pose-model-tier balanced \
    --conf-threshold 0.25 \
    --debug-overlay
```

For the included demo video (lunge):
```bash
python run_session.py \
    --session sessions/lunge_test \
    --calibration calibration/intrinsics_demo.yaml \
    --mode single \
    --video "/home/zuma/Movement Tracking project/Ant View Lunge Christina.MOV" \
    --pose-backend yolo \
    --debug-overlay
```

---

## Calibration

### Dual-camera (full stereo calibration)

```bash
python -m movement_analysis.calibration.calibrate \
    --left 0 --right 1 \
    --output calibration/cal_001.yaml \
    --grid 9 6 --square-size 0.04 --n-views 20
```

Press SPACE to capture 20+ checkerboard views from both cameras simultaneously → saves intrinsics (K, dist) for each camera + relative pose (R, T) between cameras.

### Single-camera (intrinsics only)

```bash
python -m movement_analysis.calibration.calibrate \
    --left 0 --right 0 \
    --output calibration/intrinsics_only.yaml \
    --grid 9 6 --square-size 0.04 --n-views 15
```

Setting `--right 0` (same camera twice) skips stereo calibration and saves intrinsics only — sufficient for single-camera mode.

---

## Running a session

### Dual-camera (3D clinical analysis)

```bash
python run_session.py \
    --session sessions/session_001 \
    --calibration calibration/cal_001.yaml \
    --left videos/cam1.mp4 \
    --right videos/cam2.mp4 \
    --pose-backend rtmpose \
    --pose-model-tier balanced
```

### Single-camera (2D sagittal screening + debug overlay)

```bash
python run_session.py \
    --session sessions/squat_001 \
    --calibration calibration/intrinsics_only.yaml \
    --mode single \
    --video videos/squat_side.mov \
    --pose-backend yolo \
    --pose-model-tier balanced \
    --debug-overlay
```

---

## All CLI options

```
--session <path>              Session output directory (required)
--calibration <path>          Calibration YAML (required)
--mode single|dual           default: dual

# Single-camera
--video <path>               Video file for single-camera mode

# Dual-camera
--left <path>                 Left camera video
--right <path>                Right camera video

--fps <float>                 Override video FPS (auto-detect if omitted)
--conf-threshold 0.25         Min keypoint confidence (0–1)
--smooth-window 7             Savitzky-Golay smooth window (odd integer)
--output <path>               Override output directory
--log-level INFO             DEBUG | INFO | WARNING

# Pose backend
--pose-backend auto           auto | rtmpose | yolo | mediapipe (default: auto)
--pose-model-tier balanced    fast | balanced | accurate (default: balanced)
--pose-device cpu             cpu (default) or cuda:0
--debug-overlay               Generate debug_pose_overlay.mp4 with confidence colors
--quality-report              Produce tracking_quality.json (default: True)
```

---

## Outputs

### Dual-camera → `sessions/<name>/output/`

| File | Description |
|---|---|
| `session.c3d` | Full 3D trajectories in C3D format |
| `trajectories_3d.npz` | (17, 3, n_frames) compressed NPZ |
| `angles.csv` | Joint angles per frame, both legs |
| `report.json` | ROM, gait speed, cadence, tracking quality grade |
| `tracking_quality.json` | Per-frame + session quality metrics and clinical warnings |

### Single-camera → `sessions/<name>/output/`

| File | Description |
|---|---|
| `skeleton.mp4` | Skeleton overlay on original video frames |
| `angles.csv` | Sagittal-plane hip/knee/ankle angles per frame |
| `report.json` | ROM summary + tracking quality grade + clinical warnings |
| `tracking_quality.json` | Per-frame quality metrics + session grade |
| `debug_pose_overlay.mp4` | (if `--debug-overlay`) Confidence-colored debug view |

---

## Tracking quality grades

Every session receives a quality grade based on critical lower-limb frame usability:

| Grade | Criteria | Action |
|---|---|---|
| **excellent** | ≥90% usable critical frames | Use all metrics freely |
| **good** | 80–90% | Use metrics; no flag needed |
| **caution** | 65–80% | Report metrics with warning |
| **poor** | <65% | Suppress precise clinical metrics; require manual review |

---

## Clinical guardrails

- Every report includes `tracking_quality_grade` and `clinical_warnings`
- Poor tracking suppresses or flags downstream metrics
- Single-camera outputs are always labeled "2D screening only"
- Dual-camera outputs require valid calibration
- The system never implies diagnostic certainty from markerless pose alone

---

## Project structure

```
movement_analysis/
├── README.md
├── requirements.txt
├── run_session.py                # Unified CLI
├── config/
│   └── pose_tracking.yaml        # Model tier configs and checkpoint URLs
│
├── keypoints/
│   ├── __init__.py
│   ├── detect.py                 # PoseDetection dataclass + façade + backend router
│   ├── subject_selector.py       # Multi-person primary subject heuristic
│   ├── quality.py                # Per-frame + per-session tracking quality metrics
│   ├── visualize.py              # Debug overlay renderer
│   └── backends/
│       ├── mediapipe_backend.py  # Legacy fallback (manual path resolution for .tasks API)
│       ├── yolo_pose_backend.py   # Ultralytics YOLO-Pose (COCO-17 native, recommended)
│       └── rtmpose_backend.py    # OpenMMLab RTMPose (preferred clinical backend)
│
├── calibration/
│   ├── __init__.py
│   ├── calibrate.py              # Interactive checkerboard calibration
│   └── intrinsics_demo.yaml      # Demo intrinsics (no extrinsic/stereo)
│
├── geometry/
│   └── triangulate.py            # DLT triangulation, RANSAC, SG smoothing
│
├── kinematics/
│   ├── angles.py                 # 3D joint angles (dual-camera)
│   ├── angles_2d.py              # 2D sagittal angles (single-camera)
│   └── export.py                 # C3D, CSV, JSON report
│
├── utils/
│   └── visualize.py              # Skeleton overlay renderer
│
└── sessions/                     # Per-session output (gitignored)
```

---

*Last updated: May 2026 — v1.0*