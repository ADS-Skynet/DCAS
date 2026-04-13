# DCAS — Driver Control Assistance System

Real-time driver impairment detection using a laptop camera and MediaPipe face landmarking.
The system continuously scores two impairment states — **drowsy** and **drunk** — from facial signals and head movement, and displays live alerts.

## How it works

Each frame from the camera is processed by MediaPipe's Face Landmarker model, which provides:
- 478 facial landmark positions (including iris centres)
- Facial blendshape scores (eye blink, squint, etc.)
- 3D head transformation matrix (pitch / yaw / roll)

A 5-second rolling window of these values feeds an `ImpairmentScorer` that computes two scores (0–100):

### Drowsy score

| Signal | Weight | Description |
|---|---|---|
| PERCLOS | 45% | Proportion of frames where average eye closure > 50% |
| Consecutive closure | 35% | Longest unbroken run of closed-eye frames (microsleep detection) |
| Pitch droop | 20% | Forward pitch slope over the window — slow head-nod |

### Drunk score

| Signal | Weight | Description |
|---|---|---|
| Iris jitter | 30% | Iris position std-dev normalised by inter-ocular distance (nystagmus proxy) |
| Roll sway | 30% | Std-dev of head roll — side-to-side sway |
| Yaw sway | 20% | Std-dev of head yaw — left-right sway |
| Movement chaos | 20% | Mean absolute jerk (first-diff) across all three rotation axes |

Scores drive a status label: **Normal → Warning → Alert**, displayed as a HUD overlay on the camera feed.

## Requirements

- Python 3.10+
- `mediapipe >= 0.10`
- `opencv-python`
- `numpy`

Install dependencies:
```bash
pip install mediapipe opencv-python numpy
```

Download the MediaPipe Face Landmarker model:
```bash
wget -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## Usage

```bash
python Drowsy_Drunk.py
```

Press `q` to quit.

By default the system opens `/dev/video0`. If your camera is on a different index, change `CAMERA_INDEX` at the top of [Drowsy_Drunk.py](Drowsy_Drunk.py).

## Tuning

All thresholds are constants at the top of [Drowsy_Drunk.py](Drowsy_Drunk.py):

| Constant | Default | Meaning |
|---|---|---|
| `WINDOW_SECS` | 5 | Rolling window length in seconds |
| `EYE_CLOSED_THRESH` | 0.50 | Blink blendshape score above which eye counts as closed |
| `PERCLOS_THRESH` | 0.25 | PERCLOS value that saturates the drowsy contribution |
| `CAMERA_INDEX` | 0 | Camera device index |

The normalisation denominators inside `ImpairmentScorer._compute()` (e.g. `roll_std / 8.0`) should be calibrated against real test data.

## Limitations

- Severe drowsiness and severe intoxication produce overlapping signals (both cause microsleeps); early-stage detection is more reliable.
- Iris jitter quality depends on camera resolution and lighting.
- No per-user baseline calibration — individual variation (naturally hooded eyes, resting head angle) is not yet accounted for.
