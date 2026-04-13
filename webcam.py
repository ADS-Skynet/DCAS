import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import time
from collections import deque

# ==========================================
# 1. OpenCV 웹캠 실시간 스트림 설정
# ==========================================
cap = cv2.VideoCapture(0)

# 해상도 설정 (선택 사항)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("에러: 웹캠을 열 수 없습니다.")
    exit()

print("웹캠 워밍업 중...")
time.sleep(2) # 웹캠이 켜질 시간을 잠깐 줍니다.
print("실시간 모니터링 시작! (위험 점수 스코어링 시스템 가동)")

# ==========================================
# 2. MediaPipe FaceLandmarker 설정
# ==========================================
base_options = python.BaseOptions(
    model_asset_path='face_landmarker.task',
    delegate=python.BaseOptions.Delegate.CPU
)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

prev_time = time.time()

# 시스템 변수
WINDOW_TIME = 3.0

BLINK_THRESHOLD = 0.3
PITCH_THRESHOLD = 15.0
YAW_THRESHOLD = 30.0

blink_history = deque()

# 스코어링을 위한 변수
eyes_off_score_accum = 0.0 # 점진적 누적을 위한 실수형 변수
MAX_ACCUM_SCORE = 60.0 # 100점으로 환산하기 위한 기준 (약 2초 완전 이탈 시 만점)


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

        current_time = time.time()
        is_closed = 0

        pitch, yaw, roll = 0.0, 0.0, 0.0

        if result.facial_transformation_matrixes:
            transformation_matrix = result.facial_transformation_matrixes[0]
            rotation_matrix = transformation_matrix[:3, :3]
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

            pitch = -angles[0]
            yaw = -angles[1]
            roll = angles[2]
        
        # 블렌드 셰이프 추출을 위한 변수 초기화
        eye_look_down_left, eye_look_down_right = 0.0, 0.0
        blink_left, blink_right = 0.0, 0.0
        look_left, look_right = 0.0, 0.0

        if result.face_blendshapes:
            blendshapes = result.face_blendshapes[0]
            eye_blink_left = next((item.score for item in blendshapes if item.category_name == 'eyeBlinkLeft'), 0.0)
            eye_blink_right = next((item.score for item in blendshapes if item.category_name == 'eyeBlinkRight'), 0.0)
            
            # Gaze 관련 점수 추출 (다음 단계를 위해 미리 변수만 빼두었습니다)
            eye_look_down_left = next((item.score for item in blendshapes if item.category_name == 'eyeLookDownLeft'), 0.0)
            eye_look_down_right = next((item.score for item in blendshapes if item.category_name == 'eyeLookDownRight'), 0.0)
            
            if pitch >= -4.0:
                is_l_closed = (eye_blink_left >= BLINK_THRESHOLD) and ()

            if eye_blink_left >= BLINK_THRESHOLD and eye_blink_right >= BLINK_THRESHOLD:
                is_closed = 1
                continuous_open_frames = 0
            else:
                is_closed = 0
                continuous_open_frames += 1

            cv2.putText(annotated_bgr, f'L Blink: {eye_blink_left:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(annotated_bgr, f'R Blink: {eye_blink_right:.2f}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.putText(annotated_bgr, f'L Down: {eye_look_down_left:.2f}', (150, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(annotated_bgr, f'R Down: {eye_look_down_right:.2f}', (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        

        is_eyes_off = False

        # 조건 1, 2 : 고개가 상하/좌우로 크게 돌아갔는가?
        if abs(yaw) > YAW_THRESHOLD or abs(pitch) > PITCH_THRESHOLD:
            is_eyes_off = True
        
        # 조건 3 : 고개는 정면이더라도 눈동자만 극단적으로 아래를 향하는가?
        elif eye_look_down_left > LOOK_DOWN_THRESHOLD and eye_look_down_right > LOOK_DOWN_THRESHOLD:
            is_eyes_off = True

        if is_eyes_off:
            eyes_off_frames += 1
        else:
            eyes_off_frames = 0

        # ==========================================
        # 4. 큐 관리 (절대 clear 하지 않음)
        # ==========================================
        
        blink_history.append((current_time, is_closed))

        while blink_history and (current_time - blink_history[0][0]) > WINDOW_TIME:
            blink_history.popleft()

        # ==========================================
        # 5. 순수 PERCLOS 계산
        # ==========================================

        if len(blink_history) > 30:
            closed_frames = sum(state for timestamp, state in blink_history)
            perclos = closed_frames / len(blink_history)
        else:
            perclos = 0.0

        # ==========================================
        # 6. 스마트 상태 판별 (보수적 알람 & 빠른 복귀)
        # ==========================================
        # 조건 1: PERCLOS >= 60% (약 1.8~2초 연속 눈 감김)
        # 조건 2: continuous_open_frames < 20 (눈을 번쩍 뜨면 즉시 알람 해제)
        if perclos >= PERCLOS_THRESHOLD and continuous_open_frames < 20:
            dms_state = "WARNING: Drowsiness"
            color = (0, 0, 255)
        elif eyes_off_frames > MAX_EYES_OFF_FRAMES:
            dms_state = "WARNING: DISTRACTION"
            color = (0, 165, 255)
        else:
            dms_state = "OK: Normal"
            color = (0, 255, 0)

        cv2.putText(annotated_bgr, f'State: {dms_state}', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(annotated_bgr, f'PERCLOS: {perclos*100:.1f}%', (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(annotated_bgr, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        cv2.imshow('Real-time DMS', annotated_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("종료되었습니다.")