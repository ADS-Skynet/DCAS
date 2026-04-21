# DCAS
Driver Control Assistance Systems project

eyeBlinkLeft/Right : 눈꺼풀이 닫힌 정도. 1.0이면 완전히 감은 상태

<좌우 시선 방향>
eyeLookInLeft/Right : 눈동자가 안쪽을 향하는 정도를 나타냄. 1.0일수록 눈동자가 안쪽을 향해서 쏠린 상태를 의미함. 0일수록 눈동자가 수평 정면을 향하고 있음
eyeLookOutLeft/Right : 눈동자가 바깥쪽을 향하는 정도를 나타냄. 1.0일수록 눈동자가 바깥쪽을 향해 쏠린 상태를 의미함. 0일수록 눈동자가 수평 정면을 향하고 있음

<위 아래 시선 방향>
eyeLookUpLeft/Right : 눈동자가 위를 향하는 정도를 나타냄. 1.0일수록 눈동자가 위쪽을 향해 최대한 올라간 상태 의미함. 0일수록 눈동자가 정면을 향하고 있음
eyeLookDownLeft/Right : 눈동자가 아래를 향하는 움직임. 1.0일수록 눈동자가 아래쪽을 향해 내려간 상태를 의미함. 0일수록 눈동자가 수평 정면을 향하고 있음

eyeSquintLeft/Right : 눈 찡그림 정도를 나타냄. 1.0일수록 눈을 가늘게 찡그린 상태

eyeWideLeft/Right : 눈을 크게 뜬 정도를 나타냄. 1.0일수록 눈을 크게 부릅뜬 상태


# 🚗 DCAS Driver Monitoring Pipeline (Prototype)

이 레포지토리는 운전자의 상태(집중, 휴대폰 사용, 졸음 등)를 실시간으로 모니터링하고 분석하기 위한 파이프라인 프로토타입입니다. 

엣지 디바이스(PC)의 부하를 줄이고 시스템 안정성을 높이기 위해, 실시간 인지(MediaPipe) 파트와 고수준 맥락 분석(VLM) 파트를 분리하여 **ZeroMQ(ZMQ)**를 통해 통신하도록 설계되었습니다.

## 🏗️ 아키텍처 개요 (Architecture)

시스템은 크게 두 개의 독립된 프로세스로 동작합니다:

1. **`PC.py` (인지 레이어 / Publisher)**: 웹캠 영상을 캡처하고 MediaPipe를 이용해 1차적인 운전자 상태(고개 각도, 눈 깜빡임 등)를 추출하여 ZMQ로 송출합니다.
2. **`VLM.py` (분석 레이어 / Subscriber)**: ZMQ를 통해 최신 프레임과 데이터를 수신받고, 외부 VLM(Vision Language Model) 서버에 API를 요청하여 고수준의 행동 맥락을 분석합니다.

```text
[ 웹캠 ] 
   ↓
[ PC.py (MediaPipe) ] -- (ZeroMQ / tcp://127.0.0.1:5555) --> [ VLM.py ] -- (HTTP POST) --> [ KMU VLM Server (Qwen3-VL) ]
```

---

## 🛠️ 환경 설정 (Prerequisites)

이 프로젝트를 실행하기 위해 아래의 파이썬 패키지들이 필요합니다.

```bash
pip install mediapipe opencv-python pyzmq requests numpy
```
* **참고:** MediaPipe 구동을 위해 `face_landmarker.task` 모델 파일이 `PC.py`와 같은 디렉토리에 위치해야 합니다.

---

## 📄 파일별 핵심 로직 설명

### 1. `PC.py` (MediaPipe & ZMQ Publisher)
운전자의 얼굴 랜드마크를 실시간으로 추적하고 데이터를 발행(Publish)하는 역할을 합니다.

* **MediaPipe Video Mode:** 웹캠의 연속적인 프레임을 부드럽게 추적하기 위해 `RunningMode.VIDEO`를 사용합니다.
* **데이터 추출:** 얼굴의 Pitch, Yaw, Roll 각도 및 좌우 눈 깜빡임(`eyeBlink`) 점수를 실시간으로 계산합니다.
* **이미지 경량화:** VLM 서버 통신 용량을 줄이기 위해 OpenCV 이미지를 JPEG 형식으로 압축한 뒤 Base64 문자열로 인코딩합니다.
* **비동기 송신:** 추출된 센서 데이터와 인코딩된 이미지를 JSON 형태의 `payload`로 묶어 `tcp://127.0.0.1:5555` 포트로 데이터를 발행합니다.

### 2. `VLM.py` (ZMQ Subscriber & VLM API Client)
PC에서 보내는 데이터를 받아 VLM 서버에 맥락 분석을 요청하는 역할을 합니다.

* **ZeroMQ CONFLATE 적용:** VLM 서버의 응답을 기다리는 동안 큐(Queue)에 과거 프레임이 쌓이는 것을 방지하기 위해, 항상 **가장 최신 프레임 1개**만 가져옵니다.
* **VLM 분석:** 수신된 센서 데이터와 이미지를 바탕으로 `Qwen3-VL-2B-Instruct` 모델에 분석을 요청합니다.

---

## ⚙️ VLM 모델 튜닝 및 커스터마이징 (`call_vllm_with_image`)

`VLM.py` 내의 `call_vllm_with_image` 함수는 모델의 응답 성능과 정확도를 결정하는 핵심 부분입니다. 사용자는 필요에 따라 아래 항목들을 수정하여 튜닝할 수 있습니다.

### 1. 프롬프트 엔지니어링 (Prompt Tuning)
`prompt` 변수를 수정하여 모델이 더 정확한 판단을 내리도록 유도할 수 있습니다.
* **예:** "운전자가 전방을 보지 않고 아래를 보고 있다면 '휴대폰 사용'으로 간주해줘."와 같이 구체적인 규칙 추가 가능.

### 2. 하이퍼파라미터 조절
`payload` 내의 값을 변경하여 모델의 출력 특성을 제어합니다.
* **`temperature`**: 값이 낮을수록(0.1~0.2) 일관되고 정형화된 답변을 내놓으며, 높을수록 창의적인 답변을 합니다. (현재 0.2로 설정됨)
* **`max_tokens`**: 모델이 답변할 최대 길이를 제한합니다. (현재 256으로 설정됨)

### 3. 메시지 구조 변경 (System Message 활용)
`messages` 리스트에 `role: system`을 추가하여 모델에게 고정된 역할을 부여할 수 있습니다.
* **예:** "너는 차량 안전 시스템의 분석 전문가야. 오직 주어진 보기 중 하나만 골라서 답변해."

---

## 🚀 실행 방법 (How to Run)

ZMQ 통신을 위해 두 개의 터미널을 열고 아래 순서대로 실행합니다.

**Terminal 1 (데이터 송신부 실행)**
```bash
python PC.py
```

**Terminal 2 (데이터 수신 및 분석부 실행)**
```bash
python VLM.py
```