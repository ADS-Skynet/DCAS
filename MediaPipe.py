import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 시각화 함수 정의 (질문하신 코드와 동일)
# ==========================================
def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes, ax):
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  for score, patch in zip(face_blendshapes_scores, bar.patches):
    ax.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")

# ==========================================
# 2. 실제 실행 코드 (새로 추가된 부분)
# ==========================================

# 테스트할 이미지 파일 경로 (현재 폴더에 test_image.jpg가 있어야 합니다)
IMAGE_FILE = 'test_image.jpg' 

# 이전 단계에서 다운받은 모델 파일 로드 및 설정
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# 이미지 불러오기
image = mp.Image.create_from_file(IMAGE_FILE)

# 얼굴 랜드마크 탐지 실행
detection_result = detector.detect(image)

# 원본 이미지에 랜드마크 그리기
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

# 결과 사진 + 얼굴 표정(Blendshapes) 그래프를 한 번에 띄우기
fig, axes = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [1.1, 1]})

axes[0].imshow(annotated_image)
axes[0].axis('off')
axes[0].set_title("Face Landmarks")

if detection_result.face_blendshapes:
  plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0], axes[1])
else:
  axes[1].text(0.5, 0.5, 'No face blendshapes detected', ha='center', va='center')
  axes[1].axis('off')

plt.tight_layout()
plt.show()