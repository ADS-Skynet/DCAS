import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import time
from collections import deque

# ==========================================
# 음주 및 피로 통계 분석기 (2번 스크립트 방식)
# ==========================================
class ImpairmentScorer:

    def __init__(self, window_frames=150):  # 최근 5초간 프레임 고려
        self.n = window_frames
        self.pitch   = deque(maxlen=self.n)
        self.yaw     = deque(maxlen=self.n)
        self.roll    = deque(maxlen=self.n)
        self.blink   = deque(maxlen=self.n)
        self.iris_lx = deque(maxlen=self.n)
        self.iris_ly = deque(maxlen=self.n)
        self.iris_rx = deque(maxlen=self.n)
        self.iris_ry = deque(maxlen=self.n)

        self.drunk_score       = 0.0
        self.stat_drowsy_score = 0.0

    def update(self, pitch, yaw, roll, blink_avg, iris_lx, iris_ly, iris_rx, iris_ry):
        self.pitch.append(pitch)
        self.yaw.append(yaw)
        self.roll.append(roll)
        self.blink.append(blink_avg)
        self.iris_lx.append(iris_lx)
        self.iris_ly.append(iris_ly)
        self.iris_rx.append(iris_rx)
        self.iris_ry.append(iris_ry)
        self._compute()

    def _compute(self):
        if len(self.pitch) < 10:
            return

        alpha_drowsy = 0.2
        alpha_drunk  = 0.25

        # ── 음주 특징 추출 ──────────────────────────────────────────────────

        # 1. 시선 떨림 (Iris jitter, 안진)
        iod = abs(np.mean(self.iris_lx) - np.mean(self.iris_rx)) + 1e-6
        jitter = (np.std(self.iris_lx) + np.std(self.iris_ly)
                + np.std(self.iris_rx) + np.std(self.iris_ry)) / iod
        jitter_n = min(jitter / 0.12, 1.0)

        # 2. 불규칙한 고개 돌림
        yaw_mov   = np.abs(np.diff(self.yaw)).mean()
        yaw_mov_n = min(yaw_mov / 0.5, 1.0)

        # 3. 고개 비틀거림 (roll 방향 반전율)
        d_roll   = np.diff(self.roll)
        rate     = float(np.sum(np.sign(d_roll[1:]) != np.sign(d_roll[:-1]))) / max(len(d_roll) - 1, 1)
        roll_osc = min(rate / 1.2, 1.0)

        # 음주 점수 (EMA) - 원본 가중치 유지
        new_drunk = (jitter_n * 25) + (yaw_mov_n * 20) + (roll_osc * 25)
        self.drunk_score = alpha_drunk * new_drunk + (1 - alpha_drunk) * self.drunk_score

        # ── 피로 특징 추출 ──────────────────────────────────────────────────

        # 1. PERCLOS
        perclos   = np.mean(np.array(self.blink) > 0.3)
        perclos_n = min(perclos / 0.25, 1.0)

        # 2. 꾸벅임 (pitch 변화량)
        pitch_mov   = np.abs(np.diff(self.pitch)).mean()
        pitch_mov_n = min(pitch_mov / 0.3, 1.0)

        # 3. 고개 떨굼 (roll 지속 편향)
        roll_drift   = abs(float(np.mean(np.diff(self.roll))))
        roll_drift_n = min(roll_drift / 0.05, 1.0)

        # 피로 점수 (EMA) - 원본 가중치 유지
        new_drowsy = (perclos_n * 30) + (pitch_mov_n * 15) + (roll_drift_n * 15)
        self.stat_drowsy_score = alpha_drowsy * new_drowsy + (1 - alpha_drowsy) * self.stat_drowsy_score


# ==========================================
# 초기 설정 및 시스템 워밍업
# ==========================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("에러: 웹캠을 열 수 없습니다.")
    exit()

print("웹캠 워밍업 중...")
time.sleep(2)

# ==========================================
# MediaPipe FaceLandmarker 설정
# ==========================================
base_options = python.BaseOptions(
    model_asset_path='face_landmarker.task',
    delegate=python.BaseOptions.Delegate.CPU
)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# ==========================================
# Euro NCAP 파라미터 (30FPS 기준)
# ==========================================
FPS = 30.0

OCCLUSION_MAX        = 10.0 * FPS
LONG_DISTRACTION_MAX =  3.0 * FPS
FATIGUE_MAX          =  3.0 * FPS
UNRESPONSIVE_MAX     =  6.0 * FPS

VATS_WINDOW   = int(30.0 * FPS)
VATS_MAX      = 10.0 * FPS
EYES_ON_RESET =  2.0 * FPS

BLINK_THRESHOLD    = 0.3
PITCH_THRESHOLD    = 15.0
YAW_THRESHOLD      = 30.0
GAZE_DEV_THRESHOLD = 0.5

L_IRIS = 468
R_IRIS = 473

# ==========================================
# 타이머 및 분석기 초기화
# ==========================================
timer_occlusion        = 0
timer_long_distraction = 0
timer_fatigue          = 0
timer_eyes_on          = 0
vats_history = deque(maxlen=VATS_WINDOW)

stat_scorer = ImpairmentScorer(window_frames=int(5.0 * FPS))

# ==========================================
# 메인 루프
# ==========================================
try:
    while True:
        ret, bgr_image = cap.read()
        if not ret:
            print("프레임을 읽어올 수 없습니다.")
            break

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result    = detector.detect(mp_image)
        annotated_bgr = bgr_image.copy()

        # ==========================================
        # 가려짐 감지
        # ==========================================
        is_eyes_off = False
        is_closed   = False
        is_occluded = False

        if not result.facial_transformation_matrixes or not result.face_blendshapes or not result.face_landmarks:
            timer_occlusion += 1
            if timer_occlusion >= OCCLUSION_MAX:
                is_occluded = True
        else:
            timer_occlusion = 0

            rotation_matrix = result.facial_transformation_matrixes[0][:3, :3]
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
            pitch, yaw, roll = -angles[0], -angles[1], angles[2]

            blendshapes = result.face_blendshapes[0]

            blink_left  = next((i.score for i in blendshapes if i.category_name == 'eyeBlinkLeft'),  0.0)
            blink_right = next((i.score for i in blendshapes if i.category_name == 'eyeBlinkRight'), 0.0)
            blink_avg   = (blink_left + blink_right) / 2.0

            down_left  = next((i.score for i in blendshapes if i.category_name == 'eyeLookDownLeft'),  0.0)
            down_right = next((i.score for i in blendshapes if i.category_name == 'eyeLookDownRight'), 0.0)
            look_left  = (next((i.score for i in blendshapes if i.category_name == 'eyeLookOutLeft'),  0.0)
                        + next((i.score for i in blendshapes if i.category_name == 'eyeLookInRight'),  0.0)) / 2
            look_right = (next((i.score for i in blendshapes if i.category_name == 'eyeLookInLeft'),   0.0)
                        + next((i.score for i in blendshapes if i.category_name == 'eyeLookOutRight'), 0.0)) / 2

            landmarks        = result.face_landmarks[0]
            iris_lx, iris_ly = landmarks[L_IRIS].x, landmarks[L_IRIS].y
            iris_rx, iris_ry = landmarks[R_IRIS].x, landmarks[R_IRIS].y

            pose_dev = max(abs(yaw) / YAW_THRESHOLD, abs(pitch) / PITCH_THRESHOLD)
            gaze_dev = max(down_left, down_right, look_left, look_right)

            if max(pose_dev, gaze_dev) > GAZE_DEV_THRESHOLD:
                is_eyes_off = True

            if blink_avg > BLINK_THRESHOLD:
                is_closed = True

            # ImpairmentScorer 업데이트 (2번 스크립트 방식)
            stat_scorer.update(pitch, yaw, roll, blink_avg,
                               iris_lx, iris_ly, iris_rx, iris_ry)

        # ==========================================
        # Euro NCAP 기준 타이머 업데이트
        # ==========================================
        if not is_occluded and timer_occlusion == 0:

            if is_closed:
                timer_fatigue += 1
            else:
                timer_fatigue = 0

            if is_eyes_off:
                timer_long_distraction += 1
                timer_eyes_on = 0
                vats_history.append(1)
            else:
                timer_long_distraction = 0
                timer_eyes_on += 1
                vats_history.append(0)

            if timer_eyes_on >= EYES_ON_RESET:
                vats_history.clear()

        # ==========================================
        # 항목별 점수화 (0~100점)
        # ==========================================
        score_long  = min(100.0, (timer_long_distraction / LONG_DISTRACTION_MAX) * 100.0)
        score_vats  = min(100.0, (sum(vats_history) / VATS_MAX) * 100.0)

        # 피로/음주 점수는 ImpairmentScorer(2번 방식) 사용
        score_fatigue = min(100.0, max((timer_fatigue / FATIGUE_MAX) * 100.0,
                                       stat_scorer.stat_drowsy_score))
        score_drunk   = min(100.0, stat_scorer.drunk_score)

        # ==========================================
        # State Machine
        # ==========================================
        final_risk_score = max(score_long, score_vats, score_fatigue, score_drunk)
        dms_state, color = "NORMAL", (0, 255, 0)
        alert_msg = ""

        # 1순위 : 가려짐
        if is_occluded:
            final_risk_score = 0
            dms_state, color = "SYSTEM DEGRADED", (128, 128, 128)
            alert_msg = "CAMERA OCCLUDED (>10s)"

        # 2순위 : 무반응 운전자 (6초 이상)
        elif timer_long_distraction >= UNRESPONSIVE_MAX or timer_fatigue >= UNRESPONSIVE_MAX:
            final_risk_score = 100.0
            dms_state, color = "EMERGENCY", (255, 0, 255)
            alert_msg = "UNRESPONSIVE DRIVER"

        # 3순위 : 규정 위반
        elif final_risk_score == 100.0:
            dms_state, color = "WARNING", (0, 0, 255)
            if score_drunk == 100.0:
                alert_msg = "DRUNK / IMPAIRED DETECTED"
            elif score_fatigue == 100.0:
                alert_msg = "SLEEP DETECTED"
            elif score_long == 100.0:
                alert_msg = "LONG DISTRACTION"
            else:
                alert_msg = "SHORT DISTRACTION (VATS)"

        # 4순위 : VLM 호출 대기 구간
        elif final_risk_score >= 50.0:
            dms_state, color = "CAUTION", (0, 165, 255)

        # ==========================================
        # UI 대시보드 출력
        # ==========================================
        cv2.putText(annotated_bgr, f'STATE: {dms_state}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if alert_msg:
            cv2.putText(annotated_bgr, alert_msg, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        y_offset = 320
        scores_to_draw = [
            ("Long Dist", score_long,    (0,   0, 255)),
            ("VATS Acc",  score_vats,    (0, 165, 255)),
            ("Fatigue",   score_fatigue, (255,  0,   0)),
            ("Drunk",     score_drunk,   (200, 255, 120)),
        ]

        for i, (label, sc, c) in enumerate(scores_to_draw):
            y = y_offset + (i * 30)
            cv2.putText(annotated_bgr, f'{label}: {int(sc)}', (10, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(annotated_bgr, (140, y), (140 + int(sc * 2), y + 15), c, -1)
            cv2.rectangle(annotated_bgr, (140, y), (340, y + 15), (100, 100, 100), 1)

        cv2.putText(annotated_bgr, f'FINAL RISK: {int(final_risk_score)}', (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('DMS - Euro NCAP + Impairment Scorer', annotated_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()