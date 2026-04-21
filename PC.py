import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import time
import zmq
import base64
import json

# ZMQ PUB 소켓 설정
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://127.0.0.1:5555") # 로컬호스트 5555번 포트 사용
print("ZMQ Publisher가 5555번 포트에서 대기 중입니다...")

# 카메라 및 MediaPipe 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("에러: 웹캠을 열 수 없습니다.")
    exit()

print("웹캠 워밍업 중...")
time.sleep(2)

# MediaPipe FaceLandmarker 설정
base_options = python.BaseOptions(model_asset_path='face_landmarker.task', delegate=python.BaseOptions.Delegate.CPU)
options = vision.FaceLandmarkerOptions(
    base_options=base_options, 
    running_mode=vision.RunningMode.VIDEO, 
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True, 
    num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

start_time = time.time()


try:
    while True:
        ret, bgr_image = cap.read()
        if not ret:
            print("에러: 프레임을 읽을 수 없습니다.")
            break

        ts_ms = int((time.time() - start_time) * 1000)

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        result = detector.detect_for_video(mp_image, ts_ms)
        face_detected = False

        pitch = yaw = roll = 0.0
        if result.facial_transformation_matrixes and len(result.facial_transformation_matrixes) > 0:
            face_detected = True
            rotation_matrix = result.facial_transformation_matrixes[0][:3, :3]
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
            pitch = -angles[0]
            yaw = -angles[1]
            roll = angles[2]

        if result.face_blendshapes and len(result.face_blendshapes) > 0:
            blendshapes = result.face_blendshapes[0]
            blink_left = next((item.score for item in blendshapes if item.category_name == "eyeBlinkLeft"), 0.0)
            blink_right = next((item.score for item in blendshapes if item.category_name == "eyeBlinkRight"), 0.0)
            eye_blink = (blink_left + blink_right) / 2.0

        _, buffer = cv2.imencode('.jpg', bgr_image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        payload = {
            "timestamp": time.time(),
            "face_detected": face_detected,
            "scores": {
                "pitch": round(pitch, 2),
                "yaw": round(yaw, 2),
                "roll": round(roll, 2),
                "eye_blink": round(eye_blink, 2)
            },
            "image": image_base64
        }

        socket.send_json(payload)

        cv2.imshow('MediaPipe PUB', bgr_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    socket.close()
    context.term()
    print("캡처 종료, ZMQ 소켓 닫힘")