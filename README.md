# Movement Analysis — Markerless Dual/Single-Camera Biomechanics Pipeline

Markerless 2D→3D movement analysis using MediaPipe Pose + stereo triangulation + C3D export.
Supports **dual-camera** (3D, clinical-grade) and **single-camera** (2D sagittal, screening) modes for any moveable subject — gait, exercise form, rehabilitation, sports motion.

```
Dual:   Video × 2 → 2D Keypoints → Triangulation → 3D Trajectories → Angles → C3D
Single: Video × 1 → 2D Keypoints → 2D Sagittal Angles → CSV / JSON
```

---

## Two modes

### Dual-camera (recommended for clinical work)
- Two sync'd cameras → MediaPipe Pose → DLT triangulation → true 3D trajectories
- Outputs: `session.c3d` + `angles.csv` + `trajectories_3d.npz` + `report.json`
- Produces clinical-grade hip/knee/ankle ROM, foot progression, gait speed, cadence

### Single-camera (screening / gross assessment)
- One camera in sagittal (side) view → MediaPipe Pose → 2D joint angles only
- Outputs: `angles.csv` + `report.json`
- No 3D reconstruction, no C3D export
- Suitable for: screening, telehealth, gross gait assessment
- **Not suitable for** clinical publication or research-grade analysis

---

## Installation

```bash
cd /home/zuma/Documents/movement_analysis

python -m venv .venv
source .venv/bin/activate

pip install mmpose mmdet mmengine opencv-python scipy numpy ezc3d pyyaml
```

MediaPipe Pose models are downloaded automatically on first inference (HRNet-w32 COCO).

---

## Calibration

### Dual-camera (full stereo calibration)

```bash
python -m movement_analysis.calibration.calibrate \
    --left 0 --right 1 \
    --output calibration/cal_001.yaml \
    --grid 9 6 --square-size 0.04 --n-views 20
```

Press SPACE to capture 20+ checkerboard views from both cameras simultaneously → saves
intrinsics (K, dist) for each camera + relative pose (R, T) between cameras.

### Single-camera (intrinsics only)

```bash
python -m movement_analysis.calibration.calibrate \
    --left 0 --right 0 \
    --output calibration/intrinsics_only.yaml \
    --grid 9 6 --square-size 0.04 --n-views 15
```

Setting `--right 0` (same camera twice) skips the stereo calibration step and saves
intrinsics only — sufficient for single-camera mode.

---

## Running a session

### Dual-camera (3D clinical analysis)

```bash
python run_session.py \
    --session sessions/session_001 \
    --calibration calibration/cal_001.yaml \
    --left videos/cam1.mp4 \
    --right videos/cam2.mp4
```

### Single-camera (2D sagittal screening)

```bash
python run_session.py \
    --session sessions/session_002 \
    --calibration calibration/intrinsics_only.yaml \
    --mode single \
    --video videos/cam1_side.mp4
```

### All CLI options

```bash
python run_session.py \
    --session <path>           # Session output directory (required)
    --calibration <path>      # Calibration YAML (required)
    --mode single|dual         # default: dual
    --video <path>             # [single] video file
    --left <path>              # [dual] left camera video
    --right <path>             # [dual] right camera video
    --fps <float>              # Override video FPS (auto-detect if None)
    --conf-threshold 0.3       # Min keypoint confidence (0-1)
    --smooth-window 7          # Savitzky-Golay smooth window (odd)
    --output <path>            # Override output directory
    --log-level INFO           # DEBUG | INFO | WARNING
```

---

## Outputs

### Dual-camera → `sessions/<name>/output/`

| File | Description | Use |
|---|---|---|
| `session.c3d` | Full 3D trajectories in C3D format | Open in Visual3D / Nexus / MATLAB |
| `trajectories_3d.npz` | (17, 3, n_frames) compressed | Reprocess / custom analysis |
| `angles.csv` | Joint angles per frame, both legs | Excel / Python |
| `report.json` | ROM, gait speed, cadence, foot progression | Clinical report |

### Single-camera → `sessions/<name>/output/`

| File | Description |
|---|---|
| `angles.csv` | Sagittal-plane hip/knee/ankle angles per frame |
| `report.json` | ROM summary + step count (no 3D metrics) |

---

## Clinical metrics

| Metric | Dual | Single |
|---|---|---|
| Hip/knee/ankle ROM | ✓ | ✓ (sagittal only) |
| Foot progression angle | ✓ 3D | ✗ |
| Gait speed (m/s) | ✓ | ✓ (with calibration) |
| Cadence (steps/min) | ✓ | ✓ |
| Stride length | ✓ | ✗ |
| True 3D joint angles | ✓ | ✗ |
| C3D export | ✓ | ✗ |

---

## Camera requirements

### Dual
- ≥100 fps (120+ recommended for running)
- ≥1920×1080 resolution
- Hardware sync between cameras (<1 frame drift)
- Overlapping fields of view

### Single
- ≥60 fps (100+ recommended)
- Side view (camera perpendicular to walking direction)
- Subject stays in sagittal plane (no significant out-of-plane rotation)

---

## Project structure

```
movement_analysis/
├── README.md
├── requirements.txt
├── run_session.py              # Unified CLI (single + dual)
│
├── calibration/
│   └── calibrate.py            # Checkerboard stereo calibration
│
├── keypoints/
│   └── detect.py               # MediaPipe Pose HRNet-w32 wrapper
│
├── geometry/
│   └── triangulate.py          # DLT triangulation, RANSAC, SG smoothing
│
├── kinematics/
│   ├── angles.py               # 3D joint angles (dual-camera)
│   ├── angles_2d.py            # 2D sagittal angles (single-camera)
│   └── export.py               # C3D, CSV, report JSON
```

*Last updated: May 2026*