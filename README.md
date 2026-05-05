# Driver Monitoring System (DMS)

A real-time driver monitoring system that detects drowsiness, impairment, and distraction using a combination of facial landmark analysis and vision-language model (VLM) classification.

---

## What is a DMS?

A Driver Monitoring System (DMS) continuously observes the driver through a cabin-facing camera and assesses their alertness and attention in real time. When signs of fatigue, impairment, or distraction are detected, the system issues warnings — and in severe cases, triggers a Minimum Risk Maneuver (MRM) to bring the vehicle to a safe stop.

This DMS combines two complementary approaches:

- **MediaPipe** for high-frequency facial analysis (every frame)
- **VLM (Vision-Language Model)** for image-level scene understanding (on-demand fallback)

---

## Components

### `dms.py` — MediaPipe-based monitoring

MediaPipe is Google's open-source framework for real-time perception tasks. This project uses the `FaceLandmarker` task in `VIDEO` mode, which provides per-frame:

- **478 3D face landmarks** — including iris positions
- **52 blendshape coefficients** — fine-grained facial action scores (e.g. `eyeBlinkLeft`, `eyeLookDownRight`)
- **Facial transformation matrix** — a 4×4 rigid-body matrix encoding head pose (pitch, yaw, roll)

**Why MediaPipe?**
- Runs on-device at 30 fps with low CPU/GPU overhead
- No network dependency — suitable for embedded or automotive environments
- Provides rich, structured outputs (landmarks, blendshapes, pose) in a single inference call
- `VIDEO` mode uses temporal context for more stable results than single-image inference

---

### `vlm.py` — VLM-based fallback classifier

`vlm.py` runs as a standalone ZMQ server hosting a **Qwen3-VL-2B-Instruct** vision-language model via a vLLM-compatible API endpoint.

**Why a VLM?**
MediaPipe requires a clearly visible face to produce reliable landmark data. When parameter values freeze for more than 2 seconds — which can happen due to camera obstruction, lens contamination, or driver collapse — landmark-based scoring becomes meaningless. A VLM can interpret the raw image directly and classify the scene without requiring facial structure.

**What it classifies:**

| Category | Description |
|---|---|
| `BLOCKED_LENS` | Entire image is blurred or out of focus — camera physically blocked |
| `FOGGY_LENS` | Hazy/milky overlay — condensation or dirty lens |
| `DRIVER_UNCONSCIOUS` | Image is clear but driver's head is drooping or slumped out of frame |

**Why Qwen3-VL-2B?**
- Small enough to run on edge hardware (2B parameters)
- Supports vision-language instruction following out of the box
- Returns structured JSON output suitable for programmatic parsing
- Low temperature (0.2) ensures deterministic classification

---

## How `dms.py` and `vlm.py` work together

```
Camera frame
    │
    ├──► MediaPipe (every frame)
    │         └── face detected → score drowsy / impairment / distraction
    │
    └──► FreezeDispatcher (monitors parameter stability)
              └── params unchanged for >2s → send frame to vlm.py via ZMQ
                        └── BLOCKED / FOGGY → UI alert
                            UNCONSCIOUS     → MRM immediately
```

`dms.py` monitors four scalar values each frame: pitch, yaw, blink_left, blink_right. If none of them change by more than `STOP_DELTA = 0.01` for 60 consecutive frames (2 seconds at 30 fps), `FreezeDispatcher` sends the current frame as JPEG over ZMQ to `vlm.py`. The VLM replies with a JSON result, which `dms.py` parses to decide on a camera alert or MRM activation.

---

## MediaPipe monitoring modes

### Drowsy score

Detects fatigue by measuring eye closure patterns and head drooping behaviour.

| Parameter | Weight | How it is computed |
|---|---|---|
| PERCLOS | 35 | Fraction of frames in the 5s window where `(blinkLeft + blinkRight) / 2 > 0.5` |
| Consecutive closed | 25 | Longest unbroken run of closed-eye frames in the window |
| Pitch droop | 15 | Linear regression slope of pitch over the window — positive slope means head falling forward |
| Pitch movement | 20 | Mean absolute frame-to-frame pitch change — low movement combined with droop signals nodding off |
| Roll drift | 10 | Mean absolute frame-to-frame roll change — slow lateral head tilt |

Each raw value is normalised to [0, 1] against a reference maximum, then multiplied by its weight to produce a weighted sum (max 105). The score is smoothed with an exponential moving average: `alpha = 0.1` when rising, `alpha = 0.03` when falling, so the score climbs quickly but recovers slowly.

---

### Impairment score

Detects alcohol or drug impairment by measuring involuntary, erratic movement patterns that differ from drowsiness.

| Parameter | Weight | How it is computed |
|---|---|---|
| Iris jitter | 45 | Std deviation of iris x/y positions normalised by inter-ocular distance — captures nystagmus-like tremor |
| Yaw movement | 40 | Fraction of frames with frame-to-frame yaw change > 1° — captures erratic lateral head oscillation |
| Roll oscillation | 45 | Rate of roll direction reversals (sign changes after filtering small noise) — captures involuntary rocking |

The key distinction from drowsiness: impairment produces **high-frequency, bi-directional** motion, whereas drowsiness produces **low-frequency, unidirectional** drift. The same EMA smoothing is applied.

---

### Long distraction (Euro NCAP)

Detects sustained gaze away from the road using **True Gaze** — a combined head pose + eye gaze estimate.

```
true_yaw   = head_yaw   + (look_right − look_left)  × 40
true_pitch = head_pitch + (look_up    − look_down)   × 50
```

The eye blendshape offsets allow the system to distinguish between a driver who turns their head slightly but keeps their eyes on the road, versus one who actually looks away.

**Eyes-off** is declared when:
- `|true_yaw| > 15°`, or
- `|true_pitch| > 15°`, or
- both eyes blink score > 0.3 simultaneously

The `score_long` is the current eyes-off timer as a percentage of the 3-second threshold (90 frames). A single frame with eyes back on the road resets the long-distraction timer; 6 seconds of continuous eyes-off triggers `EMERGENCY`.

---

### VATS score (Euro NCAP)

VATS (Visual Attention Time Sharing) measures the **total accumulated eyes-off time** within a rolling 30-second window, rather than any single continuous event.

```
score_vats = sum(eyes_off_frames in last 900 frames) / 300 × 100
```

Where 300 frames = 10 seconds (the NCAP maximum tolerated off-road time in 30s). If the driver looks back at the road and stays on-road for 2 continuous seconds, the VATS history is cleared.

This catches distracted drivers who take many short glances away — each glance too brief to trigger Long distraction, but cumulatively dangerous.

---

## Alert levels and outputs

| Condition | Score threshold | Outputs |
|---|---|---|
| Caution | any score ≥ 50 | UI caution popup |
| Warning | drowsy or impairment ≥ 70 | Alert sound + vehicle brake haptic + UI check popup |
| No driver response to warning | timeout | MRM activated, UI MRM popup |
| VLM: camera issue | — | UI "Check camera" popup |
| VLM: driver unconscious | — | MRM immediately, UI MRM popup |

---

## Running the system

Start the VLM server first, then the main DMS loop:

```bash
python vlm.py          # starts ZMQ server on port 5555
python dms.py          # starts camera capture and monitoring
```

Required environment variables for `vlm.py` (in `.env`):

```
VLLM_API_URL=http://<host>:<port>/v1/chat/completions
VLLM_API_KEY=<your_key>
```

Press `q` to quit, `m` to manually toggle MRM during testing.