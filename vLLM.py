#!/usr/bin/env python3
import sys
import os
import base64
import json
import requests
from io import BytesIO
from urllib.parse import urlparse
from urllib.request import urlopen, Request
from PIL import Image

# =========================
# vLLM API 설정
# =========================
VLLM_API_URL = "http://210.121.152.22:9000/v1/chat/completions"
API_KEY = "81cbe888efea8c89da139c5cc8194393c1ead203e11e85a9a5a721428c5a2517"
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
MAX_IMAGE_SIDE = 768



def is_url(string: str) -> bool:
    """Check if the string is a valid URL."""
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except Exception:
        return False    


def get_image_base64(image_input: str) -> str:
    """Load image from URL or local path and return base64 string."""
    if is_url(image_input):
        req = Request(image_input, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req) as response:
            image_bytes = response.read()
    else:
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file '{image_input}' not found.")
        with open(image_input, "rb") as f:
            image_bytes = f.read()

    image_bytes = preprocess_image_bytes(image_bytes)
    return base64.b64encode(image_bytes).decode("utf-8")



def preprocess_image_bytes(image_bytes: bytes) -> bytes:
    """Resize/compress image to reduce multimodal tokens for small-context models."""
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



def main():
    if len(sys.argv) < 2:
        print("Usage: python describe_image_vllm.py <image_file_or_url>")
        sys.exit(1)

    image_input = sys.argv[1]
    try:
        image_base64 = get_image_base64(image_input)

        prompt = """
        운전자의 사진을 보고 상태를 아래 형식중에서 가장 적합한 답 하나만 골라서 답변해줘.
        - 운전자는 졸고 있습니다.
        - 운전자는 핸드폰을 보고 있습니다.
        - 운전자는 운전에 집중하고 있습니다.
        """

        description = call_vllm_with_image(
            image_base64=image_base64,
            prompt=prompt,
        )

        print(description)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except requests.HTTPError as e:
        print(f"HTTP Error: {e.response.text}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
