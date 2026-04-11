import pyrealsense2 as rs
import numpy as np
import cv2

# Test 1: just camera alone
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
print("Camera started OK")

frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
print("Got first frame OK")

bgr = np.asanyarray(color_frame.get_data())
print(f"Frame shape: {bgr.shape}")

pipeline.stop()
print("Camera stopped OK")