# DCAS — Driver Condition Analysis System

Real-time driver monitoring that combines face landmark analysis with vision-language model inference.

---

## Files

### `dms.py` — Driver Monitoring System

Real-time camera-based driver monitor. Runs at 30 FPS and tracks drowsiness and Euro NCAP distraction metrics.

**What it detects**

| Feature | Description |
|---|---|
| PERCLOS | Percentage of frames with eyes closed (>50% blink score) |
| Consecutive eye closure | Longest unbroken run of closed-eye frames |
| Pitch droop | Downward head tilt trend (drowsy head drop) |
| Pitch movement | Frame-to-frame head pitch change rate |
| Roll drift | Lateral head roll drift |
| True Gaze | Head yaw/pitch combined with eye-look blendshapes |
| Long Distraction | Eyes off road for >3 s continuous |
| VATS | Total eyes-off time within a 30 s rolling window |
| Occlusion | No face detected for >10 s |

**Drowsy score** (0–100) is a weighted blend of the above signals, updated each frame with exponential smoothing.

**NCAP states**: NORMAL → CAUTION → WARNING → EMERGENCY → OCCLUDED

**HUD** (left panel overlay on the live window):
- Score bars for Long Distraction, VATS, and Drowsy
- Status label
- Full parameter table with live values
- FPS counter

**vLLM integration**

Every frame, four parameters are read: `pitch`, `yaw`, `eyeBlinkLeft`, `eyeBlinkRight` (all `0.0` when no face is detected). If none of these values change by more than `0.01` for 2 consecutive seconds (60 frames), one frame is sent to the vLLM server over ZMQ for visual analysis. The trigger resets once the parameters start changing again.

**Key constants** (top of file)

| Constant | Default | Meaning |
|---|---|---|
| `CAMERA_INDEX` | `0` | Camera device index |
| `FPS_TARGET` | `30` | Expected camera FPS |
| `EYE_CLOSED_THRESH` | `0.50` | Blink score threshold |
| `PERCLOS_THRESH` | `0.25` | PERCLOS alert threshold |
| `LONG_DISTRACTION_MAX` | 3 s | Long distraction warning threshold |
| `VATS_MAX` | 10 s | VATS warning threshold |
| `OCCLUSION_MAX` | 10 s | Occlusion alert threshold |
| `SAFE_YAW_LIMIT` | 15° | Max safe horizontal gaze deviation |
| `SAFE_PITCH_LIMIT` | 15° | Max safe vertical gaze deviation |
| `ZMQ_PORT` | `5555` | Port used to reach the vLLM server |
| `STOP_FRAMES` | 60 | Frames of no change before vLLM dispatch |
| `STOP_DELTA` | `0.01` | Min change to count as "still moving" |

**Run**

```bash
python dms.py
```

Press `q` in the video window to quit.

**Required file**: `face_landmarker.task` must be present in the working directory.

---

### `vLLM.py` — Vision-Language Model Server

ZMQ server that receives a JPEG frame from `dms.py`, sends it to a remote vLLM API, and returns the classification result.

**Categories returned**

| Category | Meaning |
|---|---|
| `BLOCKED_LENS` | Image is out of focus or motion-blurred — lens obstruction |
| `DRIVER_UNCONSCIOUS` | Image is clear but driver is slumped or head drooping |
| `FOGGY_LENS` | Hazy/milky overlay on the image — condensation or dirty lens |

Response is JSON:
```json
{"category": "DRIVER_UNCONSCIOUS", "confidence": 0.92}
```

The result is also printed to the terminal and written to `~/DCAS/driver_state.json`.

**Key constants** (top of file)

| Constant | Value | Meaning |
|---|---|---|
| `VLLM_API_URL` | `http://...` | Remote vLLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen3-VL-2B-Instruct` | Vision-language model |
| `MAX_IMAGE_SIDE` | `768` | Max pixel dimension before resizing |

**Run**

```bash
python vLLM.py           # listens on port 5555 (default)
python vLLM.py 5556      # custom port
```

---

## How they work together

```
dms.py (camera loop)
  │
  │  parameters unchanged for 2 s
  │
  ▼
FreezeDispatcher  ──── JPEG frame ────►  vLLM.py (ZMQ REP)
                                              │
                                              │  calls remote vLLM API
                                              │
                  ◄─── JSON result ───────────┘
                        printed to terminal
                        saved to driver_state.json
```

Start the vLLM server first, then run the DMS:

```bash
# Terminal 1
python vLLM.py

# Terminal 2
python dms.py
```

---

## Dependencies

```bash
pip install opencv-python mediapipe numpy requests Pillow pyzmq
```

Download the MediaPipe Face Landmarker model:
```bash
wget -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```
