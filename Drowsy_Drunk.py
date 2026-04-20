import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
import numpy as np
from collections import deque
import time

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH    = 'face_landmarker.task'
CAMERA_INDEX  = 0          # /dev/video0  (try 1 if this fails)
WINDOW_SECS   = 5          # rolling-window length
FPS_TARGET    = 30
WINDOW_FRAMES = FPS_TARGET * WINDOW_SECS   # 150 frames

EYE_CLOSED_THRESH = 0.50   # blink blendshape score → eyes "closed"
PERCLOS_THRESH    = 0.25   # ≥25 % closed in window triggers full drowsy signal

# Iris centre landmark indices (MediaPipe 478-point model)
L_IRIS = 468
R_IRIS = 473


# ── Helpers ──────────────────────────────────────────────────────────────────
def get_head_rotation(transformation_matrix):
    m = np.array(transformation_matrix.data).reshape(4, 4)
    r = m[:3, :3]
    pitch = np.degrees(np.arcsin(-r[2, 0]))
    yaw   = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    roll  = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
    return pitch, yaw, roll


def draw_landmarks_on_image(rgb_image, result):
    annotated = np.copy(rgb_image)
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


# ── Scoring engine ────────────────────────────────────────────────────────────
class ImpairmentScorer:
    """
    Maintains rolling buffers and computes drowsy / drunk scores (0–100).

    Drowsy signals
    ──────────────
    • PERCLOS          – % of frames with eyes > 50 % closed
    • Sustained closure – longest run of consecutive closed frames (microsleeps)
    • Head nod (droop) – slow forward-pitch trend over the window

    Drunk signals
    ─────────────
    • Roll sway  – std-dev of head roll (side-to-side)
    • Yaw sway   – std-dev of head yaw (left-right)
    • Iris jitter – std-dev of iris centre position normalised by inter-ocular
                    distance  (nystagmus proxy)
    • Movement chaos – mean absolute jerk (first-diff of all three angles)
                       — drunk = erratic; drowsy = slow or absent
    """

    def __init__(self, window: int = WINDOW_FRAMES):
        n = window
        self.pitch   = deque(maxlen=n)
        self.yaw     = deque(maxlen=n)
        self.roll    = deque(maxlen=n)
        self.blink_l = deque(maxlen=n)
        self.blink_r = deque(maxlen=n)
        self.iris_lx = deque(maxlen=n)
        self.iris_ly = deque(maxlen=n)
        self.iris_rx = deque(maxlen=n)
        self.iris_ry = deque(maxlen=n)

        self.drowsy_score: int = 0
        self.drunk_score:  int = 0
        self.signals: dict     = {}

    # ── ingest ───────────────────────────────────────────────────────────
    def update(self, blendshapes, transform_matrix, face_landmarks):
        blink_l = next((b.score for b in blendshapes if b.category_name == 'eyeBlinkLeft'),  0.0)
        blink_r = next((b.score for b in blendshapes if b.category_name == 'eyeBlinkRight'), 0.0)
        pitch, yaw, roll = get_head_rotation(transform_matrix)

        self.pitch.append(pitch);    self.yaw.append(yaw);    self.roll.append(roll)
        self.blink_l.append(blink_l); self.blink_r.append(blink_r)

        if face_landmarks and len(face_landmarks) > R_IRIS:
            self.iris_lx.append(face_landmarks[L_IRIS].x)
            self.iris_ly.append(face_landmarks[L_IRIS].y)
            self.iris_rx.append(face_landmarks[R_IRIS].x)
            self.iris_ry.append(face_landmarks[R_IRIS].y)

        self._compute()

    # ── individual signals ────────────────────────────────────────────────
    def _perclos(self) -> float:
        """Proportion of frames where average eye-blink score exceeds threshold."""
        if len(self.blink_l) < 10:
            return 0.0
        avg = (np.array(self.blink_l) + np.array(self.blink_r)) / 2.0
        return float(np.mean(avg > EYE_CLOSED_THRESH))

    def _max_consec_closed(self) -> int:
        """Longest consecutive run of closed-eye frames (microsleep detector)."""
        if len(self.blink_l) < 5:
            return 0
        avg   = (np.array(self.blink_l) + np.array(self.blink_r)) / 2.0
        closed = avg > EYE_CLOSED_THRESH
        best = cur = 0
        for c in closed:
            cur  = cur + 1 if c else 0
            best = max(best, cur)
        return best

    def _pitch_droop(self) -> float:
        """
        Forward pitch slope in degrees/frame over the window.
        Positive = head tilting down (drowsy nod).
        """
        if len(self.pitch) < 10:
            return 0.0
        t = np.arange(len(self.pitch))
        slope = float(np.polyfit(t, self.pitch, 1)[0])
        return max(0.0, slope)

    def _pitch_movement(self) -> float:
        """Mean abs frame-to-frame pitch change — head nodding (drowsy)."""
        if len(self.pitch) < 10:
            return 0.0
        return float(np.abs(np.diff(self.pitch)).mean())

    def _roll_drift(self) -> float:
        """
        Sustained directional roll — head drooping to one side (drowsy).
        abs(mean of signed diffs) is large when roll consistently moves one way.
        """
        if len(self.roll) < 10:
            return 0.0
        return abs(float(np.mean(np.diff(self.roll))))

    def _roll_oscillation(self) -> float:
        """
        Direction-reversal rate in roll — left-right-left sway (drunk).
        Random noise ≈ 50 % reversals; perfect alternation = 100 %.
        Normalised so 50 % → 0 and 100 % → 1.
        """
        if len(self.roll) < 10:
            return 0.0
        d = np.diff(self.roll)
        d = d[np.abs(d) > 0.2]  # ignore tiny changes (noise)
        if len(d) < 10:
            return 0.0
        rate = float(np.sum(np.sign(d[1:]) != np.sign(d[:-1]))) / (len(d) - 1)
        normalized = (rate - 0.5) / 0.5  # scale so 0.5 → 0 and 1.0 → 1.0
        return max(0.0, min(normalized, 1.0))

    def _yaw_movement(self) -> float:
        """Mean abs frame-to-frame yaw change — erratic head turning (drunk)."""
        if len(self.yaw) < 30:
            return 0.0
        d = np.abs(np.diff(self.yaw))
        return float(np.mean(d > 1.0))

    def _iris_jitter(self) -> float:
        """
        Combined iris-position std-dev, normalised by inter-ocular distance.
        Approximates nystagmus: scale-invariant regardless of camera distance.
        """
        if len(self.iris_lx) < 30:
            return 0.0
        iod = abs(float(np.mean(self.iris_lx)) - float(np.mean(self.iris_rx))) + 1e-6
        j = (np.std(self.iris_lx) + np.std(self.iris_ly)
           + np.std(self.iris_rx) + np.std(self.iris_ry))
        return max(0.0, float(j / iod) - 0.02)

    # ── composite scorer ─────────────────────────────────────────────────
    def _compute(self):
        perclos    = self._perclos()
        consec     = self._max_consec_closed()
        droop      = self._pitch_droop()
        pitch_mov  = self._pitch_movement()
        roll_drift = self._roll_drift()
        roll_osc   = self._roll_oscillation()
        yaw_mov    = self._yaw_movement()
        jitter     = self._iris_jitter()

        perclos_n    = min(perclos    / PERCLOS_THRESH, 1.0)
        consec_n     = min(consec     / 15.0,           1.0)
        droop_n      = min(droop      / 0.05,           1.0)
        pitch_mov_n  = min(pitch_mov  / 0.3,            1.0)
        roll_drift_n = min(roll_drift / 0.05,           1.0)
        jitter_n     = min(jitter     / 0.12,           1.0)
        yaw_mov_n    = min(yaw_mov    / 0.5,            1.0)

        alpha_drowsy = 0.2  # EMA smoothing factor
        alpha_drunk = 0.2

        # ── Drowsy score (EMA) ────────────────────────────────────────────────
        new_drowsy = int(
            perclos_n    * 35 +
            consec_n     * 25 +
            droop_n      * 15 +
            pitch_mov_n  * 20 +
            roll_drift_n * 10
        )
        self.drowsy_score = int(alpha_drowsy * new_drowsy + (1 - alpha_drowsy) * self.drowsy_score)

        # ── Drunk score (EMA) ─────────────────────────────────────────────────
        new_drunk = int(
            jitter_n  * 45 +
            yaw_mov_n * 40 +
            roll_osc  * 45
        )
        self.drunk_score = int(alpha_drunk * new_drunk + (1 - alpha_drunk) * self.drunk_score)

        self.signals = dict(
            perclos=perclos, consec_closed=consec,
            pitch_droop=droop, pitch_mov=pitch_mov,
            roll_drift=roll_drift, roll_osc=roll_osc,
            yaw_mov=yaw_mov, iris_jitter=jitter,
        )


# ── HUD drawing ──────────────────────────────────────────────────────────────
def _score_color(score: int):
    if score < 50:  return (0, 200, 100)    # green
    if score < 70:  return (0, 165, 255)    # orange
    return                 (0,  50, 255)    # red


def draw_score_bar(frame, x: int, y: int, label: str, score: int):
    BAR_W, BAR_H = 180, 16
    color = _score_color(score)
    cv2.rectangle(frame, (x, y), (x + BAR_W, y + BAR_H), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + int(BAR_W * score / 100), y + BAR_H), color, -1)
    cv2.rectangle(frame, (x, y), (x + BAR_W, y + BAR_H), (160, 160, 160), 1)
    cv2.putText(frame, f"{label}: {score:3d}", (x + BAR_W + 6, y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1)


def draw_hud(frame, scorer: ImpairmentScorer, fps: float):
    # Semi-transparent left panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (310, frame.shape[0]), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)

    # Score bars
    draw_score_bar(frame, 8,  10, "DROWSY", scorer.drowsy_score)
    draw_score_bar(frame, 8,  34, "DRUNK ", scorer.drunk_score)

    # Status label
    ds, ks = scorer.drowsy_score, scorer.drunk_score
    if   ds >= 70:              status, sc = "!! DROWSY ALERT !!", (0,  50, 255)
    elif ks >= 70:              status, sc = "!! DRUNK  ALERT !!", (0,  50, 255)
    elif ds >= 45 and ds > ks:  status, sc = "Drowsy Warning",     (0, 165, 255)
    elif ks >= 45:              status, sc = "Drunk  Warning",      (0, 165, 255)
    else:                       status, sc = "Normal",              (0, 200, 100)
    cv2.putText(frame, status, (8, 70), cv2.FONT_HERSHEY_DUPLEX, 0.65, sc, 2)

    # Signal table
    sig = scorer.signals
    rows = [
        ("PERCLOS",      f"{sig.get('perclos', 0):.2f}",           "drowsy"),
        ("Consec close", f"{sig.get('consec_closed', 0):2d} fr",   "drowsy"),
        ("Pitch droop",  f"{sig.get('pitch_droop', 0):.3f} d/fr",  "drowsy"),
        ("Pitch mov",    f"{sig.get('pitch_mov', 0):.3f} d/fr",    "drowsy"),
        ("Roll drift",   f"{sig.get('roll_drift', 0):.3f} d/fr",   "drowsy"),
        ("Roll osc",     f"{sig.get('roll_osc', 0):.2f}",          "drunk"),
        ("Yaw  mov",     f"{sig.get('yaw_mov', 0):.3f} d/fr",      "drunk"),
        ("Iris jitter",  f"{sig.get('iris_jitter', 0):.4f}",       "drunk"),
        ("FPS",          f"{fps:.1f}",                              "info"),
    ]
    colors = {"drowsy": (120, 220, 255), "drunk": (200, 255, 120), "info": (180, 180, 180)}
    for i, (name, val, kind) in enumerate(rows):
        y = 95 + i * 21
        c = colors[kind]
        cv2.putText(frame, f"{name:<14} {val}", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, c, 1)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    scorer   = ImpairmentScorer()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}. Try CAMERA_INDEX=1.")

    t0          = time.time()
    frame_count = 0
    fps         = 0.0

    print("Driver Impairment Monitor — press  q  to quit.")
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        frame_count += 1
        now  = time.time()
        fps  = frame_count / max(now - t0, 1e-6)
        ts_ms = int((now - t0) * 1000)

        rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect_for_video(mp_image, ts_ms)

        # Annotate face mesh
        annotated_rgb = draw_landmarks_on_image(rgb, result)
        display       = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        # Score update (only when a face is detected)
        if (result.face_blendshapes
                and result.facial_transformation_matrixes
                and result.face_landmarks):
            scorer.update(
                result.face_blendshapes[0],
                result.facial_transformation_matrixes[0],
                result.face_landmarks[0],
            )

        draw_hud(display, scorer, fps)
        cv2.imshow('Driver Impairment Monitor', display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == '__main__':
    main()
