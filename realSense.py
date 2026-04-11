import pyrealsense2 as rs
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import time
from collections import deque

# ==========================================
# 1. RealSense 실시간 스트림 설정
# ==========================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

print("카메라 워밍업 중...")
for _ in range(10):
    pipeline.wait_for_frames()
print("실시간 모니터링 시작! (약 10~20초 정도 켜두었다가 'q'를 눌러 종료하세요)")

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


WINDOW_TIME = 1.5
PERCLOS_THRESHOLD = 0.3
BLINK_THRESHOLD = 0.3

blink_history = deque()

continuous_open_frames = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame: continue

        bgr_image = np.asanyarray(color_frame.get_data())
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        result = detector.detect(mp_image)
        annotated_bgr = bgr_image.copy()

        current_time = time.time()
        is_closed = 0

        if result.face_blendshapes:
            blendshapes = result.face_blendshapes[0]
            eye_blink_left = next((item.score for item in blendshapes if item.category_name == 'eyeBlinkLeft'), 0.0)
            eye_blink_right = next((item.score for item in blendshapes if item.category_name == 'eyeBlinkRight'), 0.0)
            
            if eye_blink_left >= BLINK_THRESHOLD and eye_blink_right >= BLINK_THRESHOLD:
                is_closed = 1
                continuous_open_frames = 0
            else:
                is_closed = 0
                continuous_open_frames += 1

            cv2.putText(annotated_bgr, f'L Blink: {eye_blink_left:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(annotated_bgr, f'R Blink: {eye_blink_right:.2f}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        pitch, yaw, roll = 0.0, 0.0, 0.0

        if result.facial_transformation_matrixes:
            # 1. 4*4 변환 행렬 가져오기
            transformation_matrix = result.facial_transformation_matrixes[0]

            # 2. 회전 행렬 부분만 추출
            rotation_matrix = transformation_matrix[:3, :3]

            # 3. OpenCV를 사용하여 오일러 각도(Euler angles)로 변환
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

            pitch = -angles[0]
            yaw = -angles[1]
            roll = angles[2]

            cv2.putText(annotated_bgr, f'Pitch: {pitch:.1f}', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
            cv2.putText(annotated_bgr, f'Yaw: {yaw:.1f}', (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
            cv2.putText(annotated_bgr, f'Roll: {roll:.1f}', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)

        blink_history.append((current_time, is_closed))

        while blink_history and (current_time - blink_history[0][0]) > WINDOW_TIME:
            blink_history.popleft()

        if continuous_open_frames > 10:
            blink_history.clear()
            perclos = 0.0
        else:
            if len(blink_history) > 0:
                closed_frames = sum(state for timestamp, state in blink_history)
                perclos = closed_frames / len(blink_history)
            else:
                perclos = 0.0

        if perclos >= PERCLOS_THRESHOLD:
            dms_state = "WARNING: Drowsiness"
            color = (0, 0, 255)
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
    pipeline.stop()
    cv2.destroyAllWindows()
    print("종료되었습니다.")
