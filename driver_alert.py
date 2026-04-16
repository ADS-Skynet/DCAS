import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
import numpy as np
from collections import deque
import time
import os
import json
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH    = 'face_landmarker.task'
CAMERA_INDEX  = 0
WINDOW_SECS   = 5
FPS_TARGET    = 30
WINDOW_FRAMES = FPS_TARGET * WINDOW_SECS   # 150 frames

EYE_CLOSED_THRESH = 0.50
PERCLOS_THRESH    = 0.25

L_IRIS = 468
R_IRIS = 473

# ── Alert 설정 ────────────────────────────────────────────────────────────────
ALERT_SCORE_THRESH = 40    # 이 점수 이상이면 캡처 시작
SAMPLE_INTERVAL    = 11    # 30fps에서 11프레임마다 1장 샘플링 → 약 2.7fps
CAPTURE_FRAMES     = 16    # 저장할 프레임 수 (약 6초치)
CAPTURE_RESIZE     = (224, 168)   # VLM 전송용 해상도 (48토큰/장 × 16장 = 768토큰)


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


# ── Capture Manager ───────────────────────────────────────────────────────────
class CaptureManager:
    """
    alert_score 임계값 초과 시 프레임 + 파라미터를 저장.
    - 30fps 원본에서 15프레임마다 샘플링 → 2fps 효과
    - 10프레임(5초치) 슬라이딩 버퍼 유지
    - 임계값 초과 시 버퍼 내용을 폴더에 저장
    """

    def __init__(self, session_dir: str):
        self.session_dir = session_dir
        os.makedirs(session_dir, exist_ok=True)

        # 슬라이딩 버퍼: (frame_bgr, params_dict) 튜플
        self.frame_buffer: deque = deque(maxlen=CAPTURE_FRAMES)

        self.frame_count   = 0       # 전체 프레임 카운터 (샘플링 기준)
        self.event_count   = 0       # 저장된 이벤트 수
        self.capturing     = False   # 현재 저장 중인지
        self.last_save_time = 0.0   # 중복 저장 방지용

    def update(self, frame_bgr, params: dict, alert_score: int):
        self.frame_count += 1

        # 2fps 샘플링: 15프레임마다 1장만 버퍼에 추가
        if self.frame_count % SAMPLE_INTERVAL == 0:
            small = cv2.resize(frame_bgr, CAPTURE_RESIZE)
            self.frame_buffer.append((small.copy(), params.copy()))

        # 임계값 초과 + 5초 쿨다운 (같은 이벤트 중복 저장 방지)
        now = time.time()
        if (alert_score >= ALERT_SCORE_THRESH
                and len(self.frame_buffer) == CAPTURE_FRAMES
                and now - self.last_save_time > 5.0):
            self._save_event(alert_score)
            self.last_save_time = now

    def _save_event(self, alert_score: int):
        self.event_count += 1
        event_id  = f"event_{self.event_count:03d}"
        event_dir = os.path.join(self.session_dir, event_id)
        os.makedirs(event_dir, exist_ok=True)

        all_params = []

        for i, (frame, params) in enumerate(self.frame_buffer):
            # 이미지 저장
            img_path = os.path.join(event_dir, f"frame_{i:02d}.jpg")
            cv2.imwrite(img_path, frame)

            # 프레임별 파라미터 JSON 저장
            json_path = os.path.join(event_dir, f"frame_{i:02d}.json")
            with open(json_path, 'w') as f:
                json.dump(params, f, indent=2, ensure_ascii=False)

            all_params.append(params)

        # 이벤트 요약 저장
        meta = {
            "event_id":    event_id,
            "timestamp":   datetime.now().isoformat(),
            "alert_score": alert_score,
            "num_frames":  len(self.frame_buffer),
            "sample_fps":  FPS_TARGET / SAMPLE_INTERVAL,
            "window_secs": WINDOW_SECS,
            "frames":      all_params,
        }
        with open(os.path.join(event_dir, "event_meta.json"), 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"[CAPTURE] {event_id} saved — alert_score={alert_score}")


# ── Scoring engine ────────────────────────────────────────────────────────────
class ImpairmentScorer:
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
        self.alert_score:  int = 0   # ← 통합 점수 추가
        self.signals: dict     = {}

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

    def _perclos(self) -> float:
        if len(self.blink_l) < 10:
            return 0.0
        avg = (np.array(self.blink_l) + np.array(self.blink_r)) / 2.0
        return float(np.mean(avg > EYE_CLOSED_THRESH))

    def _max_consec_closed(self) -> int:
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
        if len(self.pitch) < 30:
            return 0.0
        t = np.arange(len(self.pitch))
        slope = float(np.polyfit(t, self.pitch, 1)[0])
        return max(0.0, slope)

    def _roll_std(self) -> float:
        return float(np.std(self.roll)) if len(self.roll) > 5 else 0.0

    def _yaw_std(self) -> float:
        return float(np.std(self.yaw))  if len(self.yaw)  > 5 else 0.0

    def _iris_jitter(self) -> float:
        if len(self.iris_lx) < 10:
            return 0.0
        iod = abs(float(np.mean(self.iris_lx)) - float(np.mean(self.iris_rx))) + 1e-6
        j = (np.std(self.iris_lx) + np.std(self.iris_ly)
           + np.std(self.iris_rx) + np.std(self.iris_ry))
        return float(j / iod)

    def _movement_chaos(self) -> float:
        if len(self.pitch) < 10:
            return 0.0
        dp = float(np.abs(np.diff(self.pitch)).mean())
        dy = float(np.abs(np.diff(self.yaw)).mean())
        dr = float(np.abs(np.diff(self.roll)).mean())
        return dp + dy + dr

    def _compute(self):
        perclos = self._perclos()
        consec  = self._max_consec_closed()
        droop   = self._pitch_droop()
        roll_s  = self._roll_std()
        yaw_s   = self._yaw_std()
        jitter  = self._iris_jitter()
        chaos   = self._movement_chaos()

        perclos_n = min(perclos / PERCLOS_THRESH, 1.0)
        consec_n  = min(consec / 15.0, 1.0)
        droop_n   = min(droop / 0.05, 1.0)

        self.drowsy_score = int(perclos_n * 45 + consec_n * 35 + droop_n * 20)

        roll_n   = min(roll_s  / 8.0,  1.0)
        yaw_n    = min(yaw_s   / 8.0,  1.0)
        jitter_n = min(jitter  / 0.05, 1.0)
        chaos_n  = min(chaos   / 1.0,  1.0)

        self.drunk_score = int(roll_n * 30 + yaw_n * 20 + jitter_n * 30 + chaos_n * 20)

        # ── 통합 alert_score ───────────────────────────────────────────────
        # 둘 중 높은 값을 기반으로 하되, 둘 다 높으면 가중합산
        self.alert_score = min(100, max(self.drowsy_score, self.drunk_score)
                               + int(min(self.drowsy_score, self.drunk_score) * 0.3))

        self.signals = dict(
            perclos=perclos, consec_closed=consec, pitch_droop=droop,
            roll_std=roll_s, yaw_std=yaw_s,
            iris_jitter=jitter, movement_chaos=chaos,
            drowsy_score=self.drowsy_score,
            drunk_score=self.drunk_score,
            alert_score=self.alert_score,
        )

    def get_params_snapshot(self) -> dict:
        """현재 프레임의 모든 파라미터를 딕셔너리로 반환 (저장용)"""
        sig = self.signals
        return {
            # 눈 관련
            "blink_left":     float(self.blink_l[-1]) if self.blink_l  else 0.0,
            "blink_right":    float(self.blink_r[-1]) if self.blink_r  else 0.0,
            "perclos":        round(sig.get("perclos", 0), 4),
            "consec_closed":  sig.get("consec_closed", 0),
            # 고개 관련
            "pitch":          round(float(self.pitch[-1]), 2) if self.pitch else 0.0,
            "yaw":            round(float(self.yaw[-1]),   2) if self.yaw   else 0.0,
            "roll":           round(float(self.roll[-1]),  2) if self.roll  else 0.0,
            "pitch_droop":    round(sig.get("pitch_droop", 0), 5),
            # 시선/움직임
            "iris_jitter":    round(sig.get("iris_jitter", 0), 5),
            "roll_std":       round(sig.get("roll_std", 0), 3),
            "yaw_std":        round(sig.get("yaw_std",  0), 3),
            "movement_chaos": round(sig.get("movement_chaos", 0), 3),
            # 점수
            "drowsy_score":   self.drowsy_score,
            "drunk_score":    self.drunk_score,
            "alert_score":    self.alert_score,
        }


# ── HUD drawing ──────────────────────────────────────────────────────────────
def _score_color(score: int):
    if score < 45:  return (0, 200, 100)
    if score < 70:  return (0, 165, 255)
    return                 (0,  50, 255)


def draw_score_bar(frame, x: int, y: int, label: str, score: int):
    BAR_W, BAR_H = 180, 16
    color = _score_color(score)
    cv2.rectangle(frame, (x, y), (x + BAR_W, y + BAR_H), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + int(BAR_W * score / 100), y + BAR_H), color, -1)
    cv2.rectangle(frame, (x, y), (x + BAR_W, y + BAR_H), (160, 160, 160), 1)
    cv2.putText(frame, f"{label}: {score:3d}", (x + BAR_W + 6, y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1)


def draw_hud(frame, scorer: ImpairmentScorer, fps: float, capturing: bool):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (310, frame.shape[0]), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)

    draw_score_bar(frame, 8,  10, "DROWSY", scorer.drowsy_score)
    draw_score_bar(frame, 8,  34, "DRUNK ", scorer.drunk_score)
    draw_score_bar(frame, 8,  58, "ALERT ", scorer.alert_score)   # ← 통합 점수 바

    ds, ks, als = scorer.drowsy_score, scorer.drunk_score, scorer.alert_score
    if   ds >= 70:              status, sc = "!! DROWSY ALERT !!", (0,  50, 255)
    elif ks >= 70:              status, sc = "!! DRUNK  ALERT !!", (0,  50, 255)
    elif als >= ALERT_SCORE_THRESH: status, sc = "Capturing...",   (0, 165, 255)
    else:                           status, sc = "Normal",         (0, 200, 100)
    cv2.putText(frame, status, (8, 88), cv2.FONT_HERSHEY_DUPLEX, 0.65, sc, 2)

    # 캡처 중 표시
    if capturing:
        cv2.circle(frame, (295, 15), 8, (0, 0, 255), -1)   # 빨간 점

    sig = scorer.signals
    rows = [
        ("PERCLOS",      f"{sig.get('perclos', 0):.2f}",           "drowsy"),
        ("Consec close", f"{sig.get('consec_closed', 0):2d} fr",   "drowsy"),
        ("Pitch droop",  f"{sig.get('pitch_droop', 0):.3f} d/fr",  "drowsy"),
        ("Roll std",     f"{sig.get('roll_std', 0):.1f} deg",      "drunk"),
        ("Yaw  std",     f"{sig.get('yaw_std', 0):.1f} deg",       "drunk"),
        ("Iris jitter",  f"{sig.get('iris_jitter', 0):.4f}",       "drunk"),
        ("Move chaos",   f"{sig.get('movement_chaos', 0):.2f}",    "drunk"),
        ("Alert score",  f"{sig.get('alert_score', 0)}",           "alert"),
        ("FPS",          f"{fps:.1f}",                              "info"),
    ]
    colors = {"drowsy": (120, 220, 255), "drunk": (200, 255, 120),
              "alert": (100, 100, 255),  "info": (180, 180, 180)}
    for i, (name, val, kind) in enumerate(rows):
        y = 110 + i * 21
        cv2.putText(frame, f"{name:<14} {val}", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, colors[kind], 1)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # 세션 디렉토리 생성
    session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    session_dir  = os.path.join("captures", session_name)
    capturer     = CaptureManager(session_dir)
    print(f"[INFO] Saving captures to: {session_dir}")

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
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}.")

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
        now   = time.time()
        fps   = frame_count / max(now - t0, 1e-6)
        ts_ms = int((now - t0) * 1000)

        rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect_for_video(mp_image, ts_ms)

        annotated_rgb = draw_landmarks_on_image(rgb, result)
        display       = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        if (result.face_blendshapes
                and result.facial_transformation_matrixes
                and result.face_landmarks):
            scorer.update(
                result.face_blendshapes[0],
                result.facial_transformation_matrixes[0],
                result.face_landmarks[0],
            )
            # 파라미터 스냅샷 + 캡처 매니저 업데이트
            params = scorer.get_params_snapshot()
            params["timestamp_ms"] = ts_ms
            capturer.update(frame_bgr, params, scorer.alert_score)

        capturing = scorer.alert_score >= ALERT_SCORE_THRESH
        draw_hud(display, scorer, fps, capturing)
        cv2.imshow('Driver Impairment Monitor', display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == '__main__':
    main()