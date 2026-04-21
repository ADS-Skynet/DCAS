import zmq
import base64
import numpy as np
import cv2
import time
import json
import requests
import threading

VLLM_API_URL = "http://210.121.152.22:9000/v1/chat/completions"
API_KEY = "81cbe888efea8c89da139c5cc8194393c1ead203e11e85a9a5a721428c5a2517"
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

def call_vllm_with_image(image_base64: str, prompt: str) -> str:
    """Send image + prompt to vLLM Chat Completions API."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            # data URL 형식 필수
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                ],
            }
        ],
        "max_tokens" : 256,
        "temperature": 0.2,
    }

    response = requests.post(
        VLLM_API_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=60,
    )
    response.raise_for_status()

    result = response.json()
    if "choices" not in result:
        if "error" in result and isinstance(result["error"], dict):
            error_message = result["error"].get("message", "Unknown vLLM error.")
            raise ValueError(f"vLLM API error: {error_message}")
        raise ValueError(f"Unexpected API response format: {result}")

    return result["choices"][0]["message"]["content"]


# ZMQ SUB 소켓 설정
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://127.0.0.1:5555")
socket.setsockopt_string(zmq.SUBSCRIBE, "")

socket.setsockopt(zmq.CONFLATE, 1)

print("ZMQ Subscriber 시작됨. 데이터 기다림...")

try:
    while True:
        payload = socket.recv_json()

        timestamp = payload["timestamp"]
        scores = payload["scores"]
        image_base64 = payload["image"]
        face_detected = payload["face_detected"]

        print("-" * 50)
        print("[1] ZMQ 사진 수신 완료! VLM 서버로 전송합니다...")

        start_t = time.time()
        prompt = "운전자의 사진을 보고 상태를 하나만 골라줘: 1. 졸음 2. 폰 사용 3. 집중"

        try:
            result = call_vllm_with_image(image_base64, prompt)
            print(f"[2] VLM 응답 도착 (소요시간: {time.time() - start_t:.2f}초)")
            print(f"👉 분석 결과: {result.strip()}")
        except Exception as e:
            print(f"❌ API 호출 에러: {e}")

        print("다음 프레임을 기다립니다...\n")

except KeyboardInterrupt:
    print("\n테스트 종료")
finally:
    socket.close()
    context.term()