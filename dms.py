# Drowsy_Drunk_NCAP.py
# Drowsy/Drunk detection (Drowsy_Drunk.py) +
# Euro NCAP distraction features (True Gaze, Long Distraction, VATS, Occlusion)

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
import numpy as np
from collections import deque
import time

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH   = 'face_landmarker.task'
CAMERA_INDEX = 0
WINDOW_SECS  = 5
FPS_TARGET   = 30
WINDOW_FRAMES = FPS_TARGET * WINDOW_SECS   # 150 frames

# Drowsy/Drunk thresholds
EYE_CLOSED_THRESH = 0.50
PERCLOS_THRESH    = 0.25

# Iris landmark indices
L_IRIS = 468
R_IRIS = 473

# ── Euro NCAP parameters ──────────────────────────────────────────────────────
# All timer thresholds in frames (assuming FPS_TARGET)
OCCLUSION_MAX        = int(10.0 * FPS_TARGET)   # 10s → occluded
LONG_DISTRACTION_MAX = int(3.0  * FPS_TARGET)   # 3s  → long distraction warning
UNRESPONSIVE_MAX     = int(6.0  * FPS_TARGET)   # 6s  → emergency

VATS_WINDOW          = int(30.0 * FPS_TARGET)   # 30s rolling window
VATS_MAX             = int(10.0 * FPS_TARGET)   # 10s total off-road in VATS window

EYES_ON_RESET        = int(2.0  * FPS_TARGET)   # 2s eyes-on resets VATS

# True Gaze: head pose + eye blendshape offset
EYE_YAW_WEIGHT   = 40.0
EYE_PITCH_WEIGHT = 50.0
SAFE_YAW_LIMIT   = 15.0
SAFE_PITCH_LIMIT = 15.0

BLINK_THRESHOLD  = 0.3   # for eyes-off detection (separate from PERCLOS)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_head_rotation(transformation_matrix):
    m = np.array(transformation_matrix.data).reshape(4, 4)
    r = m[:3, :3]
    pitch = np.degrees(np.arcsin(-r[2, 0]))
    yaw   = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    roll  = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
    return pitch, yaw, roll


def draw_landmarks_on_image(rgb_image, result):
    # drawing_utils requires an RGB writable array
    annotated = np.ascontiguousarray(rgb_image.copy())
    for face_landmarks in result.face_landmarks:
        drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
        drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())
        for conn in [
            vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
        ]:
            drawing_utils.draw_landmarks(
                image=annotated,
                landmark_list=face_landmarks,
                connections=conn,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
    return annotated


# ── Drowsy/Drunk Scoring Engine ───────────────────────────────────────────────
class ImpairmentScorer:
    def __init__(self, window: int = WINDOW_FRAMES):
        n = window
        self.pitch   = deque(maxlen=n)
        self.yaw     = deque(maxlen=n)
        self.roll    = deque(maxlen=n)
        self.blink_l = deque(maxlen=n)
        self.blink_r = deque(maxlen=n)
        # self.iris_lx = deque(maxlen=n)  # drunk (disabled)
        # self.iris_ly = deque(maxlen=n)  # drunk (disabled)
        # self.iris_rx = deque(maxlen=n)  # drunk (disabled)
        # self.iris_ry = deque(maxlen=n)  # drunk (disabled)

        self.drowsy_score: int = 0
        # self.drunk_score:  int = 0  # drunk (disabled)
        self.signals: dict     = {}

    def update(self, blendshapes, transform_matrix, face_landmarks):
        blink_l = next((b.score for b in blendshapes if b.category_name == 'eyeBlinkLeft'),  0.0)
        blink_r = next((b.score for b in blendshapes if b.category_name == 'eyeBlinkRight'), 0.0)
        pitch, yaw, roll = get_head_rotation(transform_matrix)

        self.pitch.append(pitch);    self.yaw.append(yaw);    self.roll.append(roll)
        self.blink_l.append(blink_l); self.blink_r.append(blink_r)

        # if face_landmarks and len(face_landmarks) > R_IRIS:  # drunk (disabled)
        #     self.iris_lx.append(face_landmarks[L_IRIS].x)
        #     self.iris_ly.append(face_landmarks[L_IRIS].y)
        #     self.iris_rx.append(face_landmarks[R_IRIS].x)
        #     self.iris_ry.append(face_landmarks[R_IRIS].y)

        self._compute()

    def _perclos(self) -> float:
        if len(self.blink_l) < 10: return 0.0
        avg = (np.array(self.blink_l) + np.array(self.blink_r)) / 2.0
        return float(np.mean(avg > EYE_CLOSED_THRESH))

    def _max_consec_closed(self) -> int:
        if len(self.blink_l) < 5: return 0
        avg = (np.array(self.blink_l) + np.array(self.blink_r)) / 2.0
        closed = avg > EYE_CLOSED_THRESH
        best = cur = 0
        for c in closed:
            cur  = cur + 1 if c else 0
            best = max(best, cur)
        return best

    def _pitch_droop(self) -> float:
        if len(self.pitch) < 10: return 0.0
        t = np.arange(len(self.pitch))
        slope = float(np.polyfit(t, self.pitch, 1)[0])
        return max(0.0, slope)

    def _pitch_movement(self) -> float:
        if len(self.pitch) < 10: return 0.0
        return float(np.abs(np.diff(self.pitch)).mean())

    def _roll_drift(self) -> float:
        if len(self.roll) < 10: return 0.0
        return abs(float(np.mean(np.diff(self.roll))))

    # def _roll_oscillation(self) -> float:  # drunk (disabled)
    #     if len(self.roll) < 10: return 0.0
    #     d = np.diff(self.roll)
    #     d = d[np.abs(d) > 0.2]
    #     if len(d) < 10: return 0.0
    #     rate = float(np.sum(np.sign(d[1:]) != np.sign(d[:-1]))) / (len(d) - 1)
    #     return max(0.0, min((rate - 0.5) / 0.5, 1.0))

    # def _yaw_movement(self) -> float:  # drunk (disabled)
    #     if len(self.yaw) < 30: return 0.0
    #     d = np.abs(np.diff(self.yaw))
    #     return float(np.mean(d > 1.0))

    # def _iris_jitter(self) -> float:  # drunk (disabled)
    #     if len(self.iris_lx) < 30: return 0.0
    #     iod = abs(float(np.mean(self.iris_lx)) - float(np.mean(self.iris_rx))) + 1e-6
    #     j = (np.std(self.iris_lx) + np.std(self.iris_ly)
    #        + np.std(self.iris_rx) + np.std(self.iris_ry))
    #     return max(0.0, float(j / iod) - 0.02)

    def _compute(self):
        perclos    = self._perclos()
        consec     = self._max_consec_closed()
        droop      = self._pitch_droop()
        pitch_mov  = self._pitch_movement()
        roll_drift = self._roll_drift()
        # roll_osc = self._roll_oscillation()  # drunk (disabled)
        # yaw_mov  = self._yaw_movement()        # drunk (disabled)
        # jitter   = self._iris_jitter()         # drunk (disabled)

        perclos_n    = min(perclos    / PERCLOS_THRESH, 1.0)
        consec_n     = min(consec     / 15.0,           1.0)
        droop_n      = min(droop      / 0.05,           1.0)
        pitch_mov_n  = min(pitch_mov  / 0.3,            1.0)
        roll_drift_n = min(roll_drift / 0.05,           1.0)
        # jitter_n = min(jitter / 0.12, 1.0)  # drunk (disabled)
        # yaw_mov_n = min(yaw_mov / 0.5, 1.0)  # drunk (disabled)

        alpha = 0.1

        new_drowsy = int(
            perclos_n    * 35 +
            consec_n     * 25 +
            droop_n      * 15 +
            pitch_mov_n  * 20 +
            roll_drift_n * 10
        )
        self.drowsy_score = int(alpha * new_drowsy + (1 - alpha) * self.drowsy_score)

        # new_drunk = int(               # drunk (disabled)
        #     jitter_n  * 45 +
        #     yaw_mov_n * 40 +
        #     roll_osc  * 45
        # )
        # self.drunk_score = int(alpha * new_drunk + (1 - alpha) * self.drunk_score)

        self.signals = dict(
            perclos=perclos, consec_closed=consec,
            pitch_droop=droop, pitch_mov=pitch_mov,
            roll_drift=roll_drift,
            # roll_osc=roll_osc,   # drunk (disabled)
            # yaw_mov=yaw_mov,     # drunk (disabled)
            # iris_jitter=jitter,  # drunk (disabled)
        )


# ── Euro NCAP Distraction Tracker ─────────────────────────────────────────────
class DistractionTracker:
    """
    Tracks Euro NCAP distraction metrics using VIDEO-mode blendshapes.
    Uses the same blendshape/transform_matrix data already available in Drowsy_Drunk.py.
    """
    def __init__(self):
        self.timer_occlusion        = 0
        self.timer_long_distraction = 0
        self.timer_eyes_on          = 0
        self.vats_history           = deque(maxlen=VATS_WINDOW)

        # Outputs
        self.is_occluded   = False
        self.is_eyes_off   = False
        self.score_long    = 0.0
        self.score_vats    = 0.0
        self.ncap_state    = "NORMAL"   # NORMAL / CAUTION / WARNING / EMERGENCY / OCCLUDED
        self.true_yaw      = 0.0
        self.true_pitch    = 0.0
        self.eye_yaw_off   = 0.0
        self.eye_pitch_off = 0.0

    def update(self, blendshapes, transform_matrix):
        """
        Call once per frame.
        Pass None for both args when no face is detected.
        """
        # ── Occlusion check ──────────────────────────────────────────────
        if blendshapes is None or transform_matrix is None:
            self.timer_occlusion += 1
            self.is_occluded = self.timer_occlusion >= OCCLUSION_MAX
            self._update_ncap_state()
            return

        self.timer_occlusion = 0
        self.is_occluded = False

        # ── Head pose (reuse same matrix as ImpairmentScorer) ────────────
        pitch, yaw, _ = get_head_rotation(transform_matrix)

        # ── Eye blendshape gaze offsets ──────────────────────────────────
        def _bs(name):
            return next((b.score for b in blendshapes if b.category_name == name), 0.0)

        down_l = _bs('eyeLookDownLeft');  down_r = _bs('eyeLookDownRight')
        up_l   = _bs('eyeLookUpLeft');    up_r   = _bs('eyeLookUpRight')
        in_l   = _bs('eyeLookInLeft');    in_r   = _bs('eyeLookInRight')
        out_l  = _bs('eyeLookOutLeft');   out_r  = _bs('eyeLookOutRight')

        look_left  = (out_l + in_r) / 2.0
        look_right = (in_l  + out_r) / 2.0
        look_up    = (up_l  + up_r)  / 2.0
        look_down  = (down_l + down_r) / 2.0

        self.eye_yaw_off   = (look_right - look_left) * EYE_YAW_WEIGHT
        self.eye_pitch_off = (look_up    - look_down) * EYE_PITCH_WEIGHT

        self.true_yaw   = yaw   + self.eye_yaw_off
        self.true_pitch = pitch + self.eye_pitch_off

        # ── Eyes-off decision ────────────────────────────────────────────
        blink_l = _bs('eyeBlinkLeft')
        blink_r = _bs('eyeBlinkRight')
        eyes_closed = (blink_l > BLINK_THRESHOLD and blink_r > BLINK_THRESHOLD)

        self.is_eyes_off = (
            abs(self.true_yaw)   > SAFE_YAW_LIMIT or
            abs(self.true_pitch) > SAFE_PITCH_LIMIT or
            eyes_closed
        )

        # ── NCAP timers ──────────────────────────────────────────────────
        if self.is_eyes_off:
            self.timer_long_distraction += 1
            self.timer_eyes_on = 0
            self.vats_history.append(1)
        else:
            self.timer_long_distraction = 0
            self.timer_eyes_on += 1
            self.vats_history.append(0)

        if self.timer_eyes_on >= EYES_ON_RESET:
            self.vats_history.clear()

        # ── Scores (0–100) ───────────────────────────────────────────────
        self.score_long = min(100.0, (self.timer_long_distraction / LONG_DISTRACTION_MAX) * 100.0)
        self.score_vats = min(100.0, (sum(self.vats_history) / VATS_MAX) * 100.0)

        self._update_ncap_state()

    def _update_ncap_state(self):
        if self.is_occluded:
            self.ncap_state = "OCCLUDED"
        elif self.timer_long_distraction >= UNRESPONSIVE_MAX:
            self.ncap_state = "EMERGENCY"
        elif self.score_long >= 100.0 or self.score_vats >= 100.0:
            self.ncap_state = "WARNING"
        elif self.score_long >= 50.0 or self.score_vats >= 50.0:
            self.ncap_state = "CAUTION"
        else:
            self.ncap_state = "NORMAL"


# ── HUD drawing ───────────────────────────────────────────────────────────────
def _score_color(score: int):
    if score < 50: return (0, 200, 100)
    if score < 70: return (0, 165, 255)
    return                (0,  50, 255)


def draw_score_bar(frame, x, y, label, score, bar_w=180, bar_h=16):
    color = _score_color(int(score))
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + int(bar_w * score / 100), y + bar_h), color, -1)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (160, 160, 160), 1)
    cv2.putText(frame, f"{label}: {int(score):3d}", (x + bar_w + 6, y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1)


def draw_hud(frame, scorer: ImpairmentScorer, tracker: DistractionTracker, fps: float):
    h, w = frame.shape[:2]

    # ── Left panel (unified) ──────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (310, h), (18, 18, 18), -1)
    frame[:] = cv2.addWeighted(overlay, 0.50, frame, 0.50, 0)

    # ── Score bars: Long / VATS / Drowsy ─────────────────────────────────
    draw_score_bar(frame, 8, 10, "Long  ", tracker.score_long)
    draw_score_bar(frame, 8, 34, "VATS  ", tracker.score_vats)
    draw_score_bar(frame, 8, 58, "DROWSY", scorer.drowsy_score)

    # ── Status label ──────────────────────────────────────────────────────
    ds = scorer.drowsy_score
    nc_state = tracker.ncap_state
    if tracker.is_occluded:
        status, sc = "CAMERA OCCLUDED",      (128, 128, 128)
    elif nc_state == "EMERGENCY":
        status, sc = "!! DISTRACTION EMERG", (255,   0, 255)
    elif nc_state == "WARNING":
        status, sc = "DISTRACTION WARNING",  (0,   50, 255)
    elif ds >= 70:
        status, sc = "!! DROWSY ALERT !!",   (0,   50, 255)
    elif nc_state == "CAUTION" or ds >= 45:
        status, sc = "CAUTION",              (0,  165, 255)
    else:
        status, sc = "Normal",               (0,  200, 100)
    cv2.putText(frame, status, (8, 90), cv2.FONT_HERSHEY_DUPLEX, 0.60, sc, 2)

    # ── Parameter table ───────────────────────────────────────────────────
    sig = scorer.signals
    gaze_col = (0, 0, 255) if tracker.is_eyes_off else (0, 255, 0)
    eyes_label = "EYES OFF" if tracker.is_eyes_off else "EYES ON "

    rows = [
        ("PERCLOS",      f"{sig.get('perclos', 0):.2f}",          (120, 220, 255)),
        ("Consec close", f"{sig.get('consec_closed', 0):2d} fr",  (120, 220, 255)),
        ("Pitch droop",  f"{sig.get('pitch_droop', 0):.3f} d/fr", (120, 220, 255)),
        ("Pitch mov",    f"{sig.get('pitch_mov', 0):.3f} d/fr",   (120, 220, 255)),
        ("Roll drift",   f"{sig.get('roll_drift', 0):.3f} d/fr",  (120, 220, 255)),
        ("TrueYaw",      f"{tracker.true_yaw:+6.1f} deg",           gaze_col),
        ("TruePitch",    f"{tracker.true_pitch:+6.1f} deg",         gaze_col),
        ("EyeYawOff",    f"{tracker.eye_yaw_off:+6.1f}",            (180, 255, 180)),
        ("EyePitOff",    f"{tracker.eye_pitch_off:+6.1f}",          (180, 255, 180)),
        (eyes_label,     "",                                         gaze_col),
        ("FPS",          f"{fps:.1f}",                               (180, 180, 180)),
    ]
    for i, (name, val, color) in enumerate(rows):
        y = 115 + i * 21
        cv2.putText(frame, f"{name:<14} {val}", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,          # VIDEO mode (timestamp required)
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    scorer   = ImpairmentScorer()
    tracker  = DistractionTracker()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}.")

    t0          = time.time()
    frame_count = 0
    fps         = 0.0

    print("Driver Monitor (Drowsy/Drunk + Euro NCAP) — press  q  to quit.")
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        frame_count += 1
        now   = time.time()
        fps   = frame_count / max(now - t0, 1e-6)
        ts_ms = int((now - t0) * 1000)

        rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = detector.detect_for_video(mp_image, ts_ms)

        display = frame_bgr.copy()

        # Extract data (None when no face detected)
        blendshapes  = result.face_blendshapes[0]             if result.face_blendshapes                else None
        transform_mx = result.facial_transformation_matrixes[0] if result.facial_transformation_matrixes else None
        landmarks    = result.face_landmarks[0]               if result.face_landmarks                  else None

        # Update Drowsy/Drunk scorer
        if blendshapes is not None and transform_mx is not None and landmarks is not None:
            scorer.update(blendshapes, transform_mx, landmarks)

        # Update Euro NCAP distraction tracker (handles None internally)
        tracker.update(blendshapes, transform_mx)

        draw_hud(display, scorer, tracker, fps)
        cv2.imshow('DMS Driver Monitoring System', display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == '__main__':
    main()