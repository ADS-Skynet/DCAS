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

EYE_YAW_WEIGHT = 40.0
EYE_PITCH_WEIGHT = 50.0

SAFE_YAW_LIMIT = 15.0
SAFE_PITCH_LIMIT = 15.0

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

        # 화면 출력을 위한 변수 초기화
        pitch = yaw = roll = 0.0
        in_l = in_r = out_l = out_r = up_l = up_r = down_l = down_r = 0.0
        true_yaw = true_pitch = 0.0

        if not result.facial_transformation_matrixes or not result.face_blendshapes:
            timer_occlusion += 1
            if timer_occlusion >= OCCLUSION_MAX:
                is_occluded = True
        else:
            timer_occlusion = 0

            # 1. 고개 각도 계산
            rotation_matrix = result.facial_transformation_matrixes[0][:3, :3]
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
            pitch, yaw, roll = -angles[0], -angles[1], angles[2]

            # 2. 눈동자 점수 계산 (화면 출력용 변수 저장)
            blendshapes = result.face_blendshapes[0]
            
            # 상/하
            down_l = next((item.score for item in blendshapes if item.category_name == 'eyeLookDownLeft'), 0.0)
            down_r = next((item.score for item in blendshapes if item.category_name == 'eyeLookDownRight'), 0.0)
            up_l = next((item.score for item in blendshapes if item.category_name == 'eyeLookUpLeft'), 0.0)
            up_r = next((item.score for item in blendshapes if item.category_name == 'eyeLookUpRight'), 0.0)
            
            # 좌/우 (In/Out)
            in_l = next((item.score for item in blendshapes if item.category_name == 'eyeLookInLeft'), 0.0)
            in_r = next((item.score for item in blendshapes if item.category_name == 'eyeLookInRight'), 0.0)
            out_l = next((item.score for item in blendshapes if item.category_name == 'eyeLookOutLeft'), 0.0)
            out_r = next((item.score for item in blendshapes if item.category_name == 'eyeLookOutRight'), 0.0)

            # 시선 이탈 로직 (기존 코드 유지)
            look_down = (down_l + down_r) / 2.0
            look_up = (up_l + up_r) / 2.0
            look_left = (out_l + in_r) / 2.0
            look_right = (in_l + out_r) / 2.0

            eye_yaw_offset = (look_right - look_left) * EYE_YAW_WEIGHT
            eye_pitch_offset = (look_up - look_down) * EYE_PITCH_WEIGHT

            true_yaw = yaw + eye_yaw_offset
            true_pitch = pitch + eye_pitch_offset

            if abs(true_yaw) > SAFE_YAW_LIMIT or abs(true_pitch) > SAFE_PITCH_LIMIT:
                is_eyes_off = True

            blink_left = next((item.score for item in blendshapes if item.category_name == 'eyeBlinkLeft'), 0.0)
            blink_right = next((item.score for item in blendshapes if item.category_name == 'eyeBlinkRight'), 0.0)
            if blink_left > BLINK_THRESHOLD and blink_right > BLINK_THRESHOLD:
                is_closed = True

        # ==========================================
        # Euro NCAP 기준 타이머 업데이트
        # ==========================================
        if not is_occluded and timer_occlusion == 0:
            if is_closed: timer_fatigue += 1
            else: timer_fatigue = 0

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
        score_long = min(100.0, (timer_long_distraction / LONG_DISTRACTION_MAX) * 100.0)
        score_fatigue = min(100.0, (timer_fatigue / FATIGUE_MAX) * 100.0)
        score_vats = min(100.0, (sum(vats_history) / VATS_MAX) * 100.0)
        
        # ==========================================
        # State Machine
        # ==========================================
        final_risk_score = max(score_long, score_fatigue, score_vats)
        dms_state, color, alert_msg = "NORMAL", (0, 255, 0), ""

        if is_occluded:
            final_risk_score = 0; dms_state, color, alert_msg = "SYSTEM DEGRADED", (128, 128, 128), "CAMERA OCCLUDED"
        elif timer_long_distraction >= UNRESPONSIVE_MAX or timer_fatigue >= UNRESPONSIVE_MAX:
            final_risk_score = 100.0; dms_state, color, alert_msg = "EMERGENCY", (255, 0, 255), "UNRESPONSIVE DRIVER"
        elif final_risk_score == 100.0:
            dms_state, color = "WARNING", (0, 0, 255)
            alert_msg = "SLEEP DETECTED" if score_fatigue == 100.0 else "LONG DISTRACTION" if score_long == 100.0 else "SHORT DISTRACTION"
        elif final_risk_score >= 50.0:
            dms_state, color = "CAUTION", (0, 165, 255)

        # ==========================================
        # UI 대시보드 출력
        # ==========================================
        cv2.putText(annotated_bgr, f'STATE: {dms_state}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if alert_msg: cv2.putText(annotated_bgr, alert_msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if is_occluded:
            cv2.putText(annotated_bgr, "PLEASE UNBLOCK THE CAMERA", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        else:
            # 왼쪽 하단: 상태 게이지
            cv2.putText(annotated_bgr, f'Long Dist: {int(score_long)}', (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.rectangle(annotated_bgr, (150, 350), (150 + int(score_long*2), 365), (0,0,255), -1)
            cv2.putText(annotated_bgr, f'VATS (Acc): {int(score_vats)}', (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.rectangle(annotated_bgr, (150, 380), (150 + int(score_vats*2), 395), (0,165,255), -1)
            cv2.putText(annotated_bgr, f'Fatigue: {int(score_fatigue)}', (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.rectangle(annotated_bgr, (150, 410), (150 + int(score_fatigue*2), 425), (255,0,0), -1)
            cv2.putText(annotated_bgr, f'FINAL RISK: {int(final_risk_score)}', (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 💡 오른쪽 상단: 시선 & 고개 Telemetry (HUD)
            overlay = annotated_bgr.copy()
            cv2.rectangle(overlay, (420, 10), (630, 280), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated_bgr, 0.4, 0, annotated_bgr)
            
            # [고개 각도]
            cv2.putText(annotated_bgr, "[ Head Pose (deg) ]", (430, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(annotated_bgr, f"Pitch : {pitch:5.1f}", (430, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(annotated_bgr, f"Yaw   : {yaw:5.1f}", (430, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # [안구 회전 점수]
            cv2.putText(annotated_bgr, "[ Eye Blendshapes ]", (430, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(annotated_bgr, f"In L  : {in_l:5.2f}", (430, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(annotated_bgr, f"In R  : {in_r:5.2f}", (530, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(annotated_bgr, f"Out L : {out_l:5.2f}", (430, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(annotated_bgr, f"Out R : {out_r:5.2f}", (530, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(annotated_bgr, f"Up L  : {up_l:5.2f}", (430, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(annotated_bgr, f"Up R  : {up_r:5.2f}", (530, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # [안구 보상 각도 (Offset)]
            cv2.putText(annotated_bgr, "[ Eye Offset (deg) ]", (430, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(annotated_bgr, f"Yaw Off  : {eye_yaw_offset:+5.1f}", (430, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)
            cv2.putText(annotated_bgr, f"Pitch Off: {eye_pitch_offset:+5.1f}", (430, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)

            # [종합 이탈 평가 (True Gaze)]
            gaze_color = (0, 0, 255) if is_eyes_off else (0, 255, 0)
            cv2.putText(annotated_bgr, "[ True Gaze ]", (430, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(annotated_bgr, f"True Yaw  : {true_yaw:5.1f}", (430, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.45, gaze_color, 1)

        cv2.imshow('Euro NCAP True Gaze Tracker', annotated_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    cap.release()
    cv2.destroyAllWindows()