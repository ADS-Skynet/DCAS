import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import time
from collections import deque

# ==========================================
# 초기 설정 및 시스템 워밍업
# ==========================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
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

# 시간 임계값
OCCLUSION_MAX = 10.0 * FPS
LONG_DISTRACTION_MAX = 3.0 * FPS
FATIGUE_MAX = 3.0 * FPS
UNRESPONSIVE_MAX = 6.0 * FPS

VATS_WINDOW = int(30.0 * FPS)
VATS_MAX = 10.0 * FPS
EYES_ON_RESET = 2.0 * FPS

# 인지 임계값
BLINK_THRESHOLD = 0.3
PITCH_THRESHOLD = 15.0
YAW_THRESHOLD = 30.0
DEV_THRESHOLD = 0.5 # 이 값이 어떤 값인가???

# 타이머 변수 초기화
timer_occlusion = 0
timer_long_distraction = 0
timer_fatigue = 0
timer_eyes_on = 0
vats_history = deque(maxlen = VATS_WINDOW)

try:
    while True:
        ret, bgr_image = cap.read()
        if not ret:
            print("프레임을 읽어올 수 없습니다.")
            break

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = detector.detect(mp_image)
        annotated_bgr = bgr_image.copy()

        # ==========================================
        # 가려짐 감지
        # ==========================================
        is_eyes_off = False
        is_closed = False
        is_occluded = False

        if not result.facial_transformation_matrixes or not result.face_blendshapes:
            timer_occlusion += 1
            if timer_occlusion >= OCCLUSION_MAX:
                is_occluded = True
        else:
            timer_occlusion = 0

            rotation_matrix = result.facial_transformation_matrixes[0][:3, :3]
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
            pitch, yaw, roll = -angles[0], -angles[1], angles[2]

            pose_dev = max(abs(yaw)/YAW_THRESHOLD, abs(pitch)/PITCH_THRESHOLD)

            blendshapes = result.face_blendshapes[0]
            down_left = next((item.score for item in blendshapes if item.category_name == 'eyeLookDownLeft'), 0.0)
            down_right = next((item.score for item in blendshapes if item.category_name == 'eyeLookDownRight'), 0.0)
            look_left = (next((item.score for item in blendshapes if item.category_name == 'eyeLookOutLeft'), 0.0) + next((item.score for item in blendshapes if item.category_name == 'eyeLookInRight'), 0.0)) / 2
            look_right = (next((item.score for item in blendshapes if item.category_name == 'eyeLookInLeft'), 0.0) + next((item.score for item in blendshapes if item.category_name == 'eyeLookOutRight'), 0.0)) / 2
            gaze_dev = max(down_left, down_right, look_left, look_right)

            if max(pose_dev, gaze_dev) > DEV_THRESHOLD:
                is_eyes_off = True


            blink_left = next((item.score for item in blendshapes if item.category_name == 'eyeBlinkLeft'), 0.0)
            blink_right = next((item.score for item in blendshapes if item.category_name == 'eyeBlinkRight'), 0.0)
            if blink_left > BLINK_THRESHOLD and blink_right > BLINK_THRESHOLD:
                is_closed = True

        # ==========================================
        # Euro NCAP 기준 타이머 업데이트
        # ==========================================
        if not is_occluded and timer_occlusion == 0:

            if is_closed:
                timer_fatigue += 1
            else:
                timer_fatigue = 0

            if is_eyes_off: # 시선 이탈
                timer_long_distraction += 1
                timer_eyes_on = 0
                vats_history.append(1)
            else: # 시선 정면
                timer_long_distraction = 0
                timer_eyes_on += 1
                vats_history.append(0)

            if timer_eyes_on >= EYES_ON_RESET:
                vats_history.clear()
                
        # ==========================================
        # 항목별 점수화 (0~100점)
        # ==========================================

        score_long = min(100.0, (timer_long_distraction / LONG_DISTRACTION_MAX) * 100.0)
        score_fatigue = min(100.0, (timer_fatigue / FATIGUE_MAX) * 100.0)
        score_vats = min(100.0, (sum(vats_history) / VATS_MAX) * 100.0)
        
        # ==========================================
        # State Machine
        # ==========================================

        final_risk_score = max(score_long, score_fatigue, score_vats)
        dms_state, color = "NORMAL", (0, 255, 0)
        alert_msg = ""

        # 1 순위 : 가려짐
        if is_occluded:
            final_risk_score = 0
            dms_state, color = "SYSTEM DEGRADED", (128, 128, 128)
            alert_msg = "CAMERA OCCLUDED (>10s)"

        # 2 순위 : 무반응 운전자 (6초 이상 이탈/수면)
        elif timer_long_distraction >= UNRESPONSIVE_MAX or timer_fatigue >= UNRESPONSIVE_MAX:
            final_risk_score = 100.0
            dms_state, color = "EMERGENCY", (255, 0, 255)
            alert_msg = "UNRESPONSIVE DRIVER"
        
        # 3 순위 : 규정 위반 (3초 이상 이탈/수면, or 누적 10초)
        elif final_risk_score == 100.0:
            dms_state, color = "WARNING", (0, 0, 255)
            if score_fatigue == 100.0:
                alert_msg = "SLEEP DETECTED"
            elif score_long == 100.0:
                alert_msg = "LONG DISTRACTION"
            else:
                alert_msg = "SHORT DISTRACTION (VATS)"

        # 4 순위 : VLM 호출 대기 구간
        elif final_risk_score >= 50.0:
            dms_state, color = "CAUTION", (0, 165, 255)

        # ==========================================
        # UI 대시보드 출력
        # ==========================================
        cv2.putText(annotated_bgr, f'STATE: {dms_state}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if alert_msg:
            cv2.putText(annotated_bgr, alert_msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if is_occluded:
            cv2.putText(annotated_bgr, "PLEASE UNBLOCK THE CAMERA", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        else:
            # 상태 게이지
            cv2.putText(annotated_bgr, f'Long Dist: {int(score_long)}', (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.rectangle(annotated_bgr, (150, 350), (150 + int(score_long*2), 365), (0,0,255), -1)
            
            cv2.putText(annotated_bgr, f'VATS (Acc): {int(score_vats)}', (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.rectangle(annotated_bgr, (150, 380), (150 + int(score_vats*2), 395), (0,165,255), -1)
            
            cv2.putText(annotated_bgr, f'Fatigue: {int(score_fatigue)}', (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.rectangle(annotated_bgr, (150, 410), (150 + int(score_fatigue*2), 425), (255,0,0), -1)

            cv2.putText(annotated_bgr, f'FINAL RISK: {int(final_risk_score)}', (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Euro NCAP Full Compliance', annotated_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    cap.release()
    cv2.destroyAllWindows()

