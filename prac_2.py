import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MPLBACKEND"] = "Agg"  # headless matplotlib, no GUI

import pyrealsense2 as rs
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
print("Camera started OK")

# Force CPU, no GPU/EGL for MediaPipe
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
print("Detector created OK")

frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
print("Got first frame OK")

bgr = np.asanyarray(color_frame.get_data())
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
print("mp.Image created OK")

result = detector.detect(mp_image)
print(f"Detection OK — faces found: {len(result.face_landmarks)}")

pipeline.stop()
print("All OK")