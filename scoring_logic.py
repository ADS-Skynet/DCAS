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

####### 수정 가능성 #######
# 졸음(PERCLOS) 계산할 때, 과거 몇 초 동안의 데이터를 볼 것인지 결정함.
WINDOW_TIME = 3.0 # 3초

# 0.0 ~ 1.0 사이로 나오는 눈 감김 점수. 1.0이면 완전 감은 것임.
BLINK_THRESHOLD = 0.3
# 앞, 뒤로 고개 꺾임 임계값
PITCH_THRESHOLD = 15.0
# 좌우로 고개 돌림 임계값
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
        eye_blink_left, eye_blink_right = 0.0, 0.0
        look_left, look_right = 0.0, 0.0

        if result.face_blendshapes:
            blendshapes = result.face_blendshapes[0]
            # 눈 감김 점수 추출
            eye_blink_left = next((item.score for item in blendshapes if item.category_name == 'eyeBlinkLeft'), 0.0)
            eye_blink_right = next((item.score for item in blendshapes if item.category_name == 'eyeBlinkRight'), 0.0)
            
            # Gaze 관련 점수 추출
            # 시선 아래
            eye_look_down_left = next((item.score for item in blendshapes if item.category_name == 'eyeLookDownLeft'), 0.0)
            eye_look_down_right = next((item.score for item in blendshapes if item.category_name == 'eyeLookDownRight'), 0.0)

            # 시선 좌/우
            look_left = (next((item.score for item in blendshapes if item.category_name == 'eyeLookOutLeft'), 0.0) + next((item.score for item in blendshapes if item.category_name == 'eyeLookInRight'), 0.0)) / 2
            look_right = (next((item.score for item in blendshapes if item.category_name == 'eyeLookInLeft'), 0.0) + next((item.score for item in blendshapes if item.category_name == 'eyeLookOutRight'), 0.0)) / 2

            # 양 눈의 감김 점수가 모두 임계값 넘으면 감은 것으로 간주
            if eye_blink_left > BLINK_THRESHOLD and eye_blink_right > BLINK_THRESHOLD:
                is_closed = 1

        # 점진적 페널티 로직
        # 1. 현재 프레임의 이탈 강도를 계산함.

        # 운전자가 고개를 얼마나 좌우/위아래로 회전했는지 비율로 계산
        ####### 수정 가능성 #######
        pose_dev = max(abs(yaw)/YAW_THRESHOLD, abs(pitch)/PITCH_THRESHOLD)
        # 시선 얼마나 이동했는지 강도 확인
        gaze_dev = max(eye_look_down_left, eye_look_down_right, look_left, look_right)

        # 고개와 시선 이탈 중 더 큰 값을 현재 이탈 강도로 결정
        current_deviation = max(pose_dev, gaze_dev)

        # 2. 누적 및 감쇄
        ####### 수정 가능성 #######
        # 이탈 강도가 50% 이상이면 유의미한 이탈로 보고 누적을 시작함
        if current_deviation > 0.5:
            eyes_off_score_accum = min(MAX_ACCUM_SCORE, eyes_off_score_accum + current_deviation)
        else:
            eyes_off_score_accum = max(0.0, eyes_off_score_accum - 1.5)

        # 3. PERCLOS 계산
        blink_history.append((current_time, is_closed))
        while blink_history and (current_time - blink_history[0][0]) > WINDOW_TIME:
            blink_history.popleft()
        # 30 프레임 이상 데이터가 쌓이면 PERCLOS 계산 시작
        if len(blink_history) > 30:
            closed_frames = sum(state for timestamp, state in blink_history)
            perclos = closed_frames / len(blink_history)
        else:
            perclos = 0.0

        # 위험 점수 및 상태 판별
        ####### 수정 가능성 #######
        f_distraction = (eyes_off_score_accum / MAX_ACCUM_SCORE) * 100.0
        f_fatigue = min(100.0, (perclos / 0.6) * 100.0)
        f_inactivity = 100.0 if (pitch < -10 or abs(roll) > 20) else 0.0

        # 시선 이탈(50%) + 졸음(30%) + 자세 무너짐(20%)
        risk_score = (0.5 * f_distraction) + (0.3 * f_fatigue) + (0.2 * f_inactivity)

        risk_score = min(100.0, max(0.0, risk_score))

        # 상태 및 색상 결정
        if risk_score >= 70: dms_state, color = "WARNING", (0, 0, 255)
        elif risk_score >= 50: dms_state, color = "CAUTION", (0, 165, 255)
        else: dms_state, color = "NORMAL", (0, 255, 0)


        # Pose 수치
        cv2.putText(annotated_bgr, f'Pitch: {pitch:.1f}', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
        cv2.putText(annotated_bgr, f'Yaw: {yaw:.1f}', (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
        cv2.putText(annotated_bgr, f'Roll: {roll:.1f}', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
        
        # 시선(Gaze) 수치 (추가)
        cv2.putText(annotated_bgr, f'Gaze Down: {max(eye_look_down_left, eye_look_down_right):.2f}', (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(annotated_bgr, f'Gaze Left: {look_left:.2f}', (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(annotated_bgr, f'Gaze Right: {look_right:.2f}', (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 상태 및 스코어 바
        cv2.putText(annotated_bgr, f'STATE: {dms_state}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        bar_w = int((risk_score / 100.0) * 300)
        cv2.rectangle(annotated_bgr, (10, 50), (10 + bar_w, 70), color, -1)
        cv2.putText(annotated_bgr, f'Risk: {int(risk_score)}', (320, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('DMS Progressive Scoring', annotated_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    cap.release()
    cv2.destroyAllWindows()