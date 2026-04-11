import os
os.environ["MPLBACKEND"] = "Agg"

import pyrealsense2 as rs
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io

# ==========================================
# 1. RealSense — 프레임 한 장만 캡처
# ==========================================
pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

print("카메라 워밍업 중...")
# 처음 몇 프레임은 노출/화이트밸런스가 안정되지 않으므로 버림
for _ in range(10):
    pipeline.wait_for_frames()

# 안정된 프레임 한 장 캡처
frames      = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
bgr_image   = np.asanyarray(color_frame.get_data())

pipeline.stop()
print("캡처 완료, 카메라 종료")

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

# ==========================================
# 3. 분석 실행
# ==========================================
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
result    = detector.detect(mp_image)
print(f"감지된 얼굴 수: {len(result.face_landmarks)}")

# ==========================================
# 4. 랜드마크 그리기
# ==========================================
def draw_landmarks_on_image(rgb_image, detection_result):
    annotated = np.copy(rgb_image)
    for face_landmarks in detection_result.face_landmarks:
        drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
        drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())
        drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
        drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
    return annotated

# ==========================================
# 5. 블렌드셰이프 그래프 → BGR numpy 배열
# ==========================================
def render_blendshape_graph(face_blendshapes, width=1000, height=700, columns=2):
    # 전체 항목을 유지하면서 라벨 겹침을 줄이기 위해 여러 컬럼으로 나눠 표시합니다.
    all_shapes = sorted(face_blendshapes, key=lambda c: c.score, reverse=True)
    columns = max(1, min(columns, len(all_shapes)))

    chunks = np.array_split(all_shapes, columns)

    dpi = 150
    fig, axes = plt.subplots(1, columns, figsize=(width / dpi, height / dpi), dpi=dpi)
    if columns == 1:
        axes = [axes]

    for idx, (ax, chunk) in enumerate(zip(axes, chunks), start=1):
        names = [c.category_name for c in chunk]
        scores = [c.score for c in chunk]
        ranks = range(len(names))

        bars = ax.barh(ranks, scores)
        ax.set_yticks(ranks)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.tick_params(axis='x', labelsize=10)
        ax.grid(axis='x', alpha=0.25)
        for score, patch in zip(scores, bars.patches):
            ax.text(
                patch.get_x() + patch.get_width() + 0.01,
                patch.get_y() + patch.get_height() / 2,
                f"{score:.3f}",
                va="center",
                fontsize=8,
            )
        ax.set_xlabel('Score', fontsize=11)
        ax.set_title(f"Group {idx}", fontsize=12)

    fig.suptitle("Face Blendshapes (All Features)", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_arr  = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    graph_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return cv2.resize(graph_bgr, (width, height))

# ==========================================
# 6. 결과 표시
# ==========================================
if not result.face_landmarks:
    print("얼굴이 감지되지 않았습니다.")
    # 원본 이미지만 표시
    cv2.imshow('Result — no face detected', bgr_image)
    cv2.waitKey(0)
else:
    annotated_rgb = draw_landmarks_on_image(rgb_image, result)
    annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

    panel_height = 700
    face_width = int(panel_height * annotated_bgr.shape[1] / annotated_bgr.shape[0])
    face_panel = cv2.resize(annotated_bgr, (face_width, panel_height), interpolation=cv2.INTER_LINEAR)
    graph_bgr = render_blendshape_graph(result.face_blendshapes[0], width=1000, height=panel_height, columns=2)

    combined = np.hstack((face_panel, graph_bgr))
    cv2.imshow('Face Analysis Result  |  press any key to close', combined)
    cv2.waitKey(0)  # 아무 키나 누르면 종료

cv2.destroyAllWindows()
print("완료.")