#!/usr/bin/env python3
import sys
import os
import base64
import json
import requests
from io import BytesIO
from PIL import Image

VLLM_API_URL = "http://210.121.152.22:9000/v1/chat/completions"
API_KEY = "81cbe888efea8c89da139c5cc8194393c1ead203e11e85a9a5a721428c5a2517"
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
MAX_IMAGE_SIDE = 768


def preprocess_image_bytes(image_bytes: bytes) -> bytes:
    resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    with Image.open(BytesIO(image_bytes)) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        if max(img.size) > MAX_IMAGE_SIDE:
            img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), resample)
        output = BytesIO()
        img.save(output, format="JPEG", quality=85, optimize=True)
        return output.getvalue()


def call_vllm_with_image(image_base64: str, prompt: str) -> str:
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
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ],
            }
        ],
        "max_tokens": 256,
        "temperature": 0.2,
    }
    response = requests.post(VLLM_API_URL, headers=headers, data=json.dumps(payload), timeout=60)
    response.raise_for_status()
    result = response.json()
    if "choices" not in result:
        if "error" in result and isinstance(result["error"], dict):
            raise ValueError(f"vLLM API error: {result['error'].get('message', 'Unknown error.')}")
        raise ValueError(f"Unexpected API response format: {result}")
    return result["choices"][0]["message"]["content"]


def run_server(port: int = 5555):
    import zmq
    context = zmq.Context()
    socket  = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    print(f"vLLM ZMQ server listening on tcp://*:{port}  (Ctrl-C to stop)")

    prompt = """
        Analyze this image and choose exactly ONE category.

        A: BLOCKED_LENS
        - The entire image is out of focus or motion-blurred
        - Details are indistinct but no fog/film overlay
        - Background edges are soft and undefined

        B: DRIVER_UNCONSCIOUS
        - Image is clear
        - Driver is present but head is drooping downward or slumped
        - Only the top of head or back of head visible due to drooping

        C: FOGGY_LENS
        - Entire image has a foggy, hazy, or milky overlay
        - Looks like condensation or dirt on the lens
        - Background exists but is visible through the haze

        KEY DISTINCTIONS:
        - Whole image soft/blurry → A
        - Whole image hazy/foggy overlay → C
        - Image clear, person present but collapsed → B

        Reply in JSON only, no other text:
        {"category": "BLOCKED_LENS" or "FOGGY_LENS" or "DRIVER_UNCONSCIOUS", "confidence": 0.0~1.0}
        """

    while True:
        try:
            frame_bytes = socket.recv()
            img_b64     = base64.b64encode(preprocess_image_bytes(frame_bytes)).decode("utf-8")
            result      = call_vllm_with_image(img_b64, prompt)
            socket.send_string(result)
            print(f"[vLLM] {result.strip()}")
            json_path = os.path.expanduser("~/DCAS/driver_state.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"driver_state": result.strip()}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            socket.send_string(f"[Error] {e}")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5555
    run_server(port)
