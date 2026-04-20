"""
vlm_analyze.py
VLM call functions for DCAS driver state classification.

No file I/O — all data is passed in-memory.
Input:  one snapshot frame (BGR numpy array) + list of per-frame signal dicts
Output: one of two Korean strings:
            "운전자는 졸고 있습니다."
            "운전자는 음주운전으로 추정됩니다."
"""

import base64
import requests
import numpy as np
from io import BytesIO
from PIL import Image

# ── vLLM endpoint ─────────────────────────────────────────────────────────────
VLLM_API_URL = "http://210.121.152.22:9000/v1/chat/completions"
API_KEY      = "81cbe888efea8c89da139c5cc8194393c1ead203e11e85a9a5a721428c5a2517"
MODEL_NAME   = "Qwen/Qwen3-VL-2B-Instruct"

# Token budget: 224×168 ≈ 48 tokens/image × 1 image = 48 + ~200 prompt + 32 output ≈ 280 tokens
IMAGE_W = 224
IMAGE_H = 168


# ── image util ────────────────────────────────────────────────────────────────
def frame_to_base64(frame_bgr: np.ndarray) -> str:
    """Encode a BGR numpy frame to a base64 JPEG string entirely in memory."""
    rgb = frame_bgr[:, :, ::-1]          # BGR → RGB
    img = Image.fromarray(rgb.astype(np.uint8))
    img = img.resize((IMAGE_W, IMAGE_H), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── prompt builder ────────────────────────────────────────────────────────────
def build_prompt(all_params: list) -> str:
    """
    Build the VLM prompt from a list of per-frame signal dicts.

    Expected keys in each dict (from ImpairmentScorer.signals + extras):
        perclos, consec_closed, pitch_droop, pitch_mov,
        roll_drift, roll_osc, yaw_mov, iris_jitter,
        drowsy_score, drunk_score, blink_left, blink_right
    """
    if not all_params:
        return "데이터없음. 판단불가."

    avg  = lambda k: sum(p.get(k, 0) for p in all_params) / len(all_params)
    mmax = lambda k: max(p.get(k, 0) for p in all_params)

    perclos_val  = avg("perclos")
    consec_val   = mmax("consec_closed")
    pitch_val    = avg("pitch_droop")
    roll_val     = avg("roll_drift")
    yaw_val      = avg("yaw_mov")
    jitter_val   = avg("iris_jitter")
    chaos_val    = avg("pitch_mov")      # pitch movement as movement-chaos proxy
    drowsy_score = mmax("drowsy_score")
    drunk_score  = mmax("drunk_score")

    # Eye-closed timeline: '1' = closed, '0' = open, one char per collected frame
    eye_seq = "".join(
        "1" if (p.get("blink_left", 0) + p.get("blink_right", 0)) / 2 > 0.5 else "0"
        for p in all_params
    )

    def flag(val, threshold):
        return "↑" if val >= threshold else "-"

    stats = (
        f"PERCLOS={perclos_val:.2f}{flag(perclos_val, 0.25)}(임계0.25)"
        f"|연속눈감김={consec_val}f{flag(consec_val, 15)}(임계15f)"
        f"|눈감김시계열={eye_seq}"
        f"|pitch(앞뒤)={pitch_val:.3f}{flag(pitch_val, 0.05)}(임계0.05)"
        f"|roll(좌우기울)={roll_val:.3f}{flag(roll_val, 0.05)}(임계0.05)"
        f"|yaw(좌우회전)={yaw_val:.3f}{flag(yaw_val, 0.5)}(임계0.5)"
        f"|눈동자떨림={jitter_val:.3f}{flag(jitter_val, 0.05)}(임계0.05)"
        f"|움직임혼란={chaos_val:.2f}{flag(chaos_val, 0.3)}(임계0.3)"
        f"|졸음점수={drowsy_score}/100"
        f"|음주점수={drunk_score}/100"
    )

    return (
        "당신은 차량내 운전자 모니터링 시스템(DMS)입니다.\n"
        "연속 이미지와 센서수치를 종합 판단하세요.\n\n"
        f"[센서수치] {stats}\n\n"
        "[졸음판단 - 높은비중 순서]\n"
        "①PERCLOS↑+눈감김시계열에1많음: 눈을반복적으로감음\n"
        "②연속눈감김↑: 15f이상지속=마이크로수면\n"
        "③pitch↑: 고개가앞으로천천히숙여짐\n"
        "④이미지확인: 눈반감김/고개아래처짐\n\n"
        "[음주판단 - 높은비중 순서]\n"
        "①yaw↑+roll↑: 고개가좌우로불규칙흔들림\n"
        "②눈동자떨림↑: 안진=눈동자불안정\n"
        "③움직임혼란↑: 전체적으로불규칙움직임\n"
        "④이미지확인: 고개좌우흔들림/눈동자불안정\n\n"
        "[출력] 반드시 아래 둘 중 하나만, 다른말없이:\n"
        "운전자는 졸고 있습니다.\n"
        "운전자는 음주운전으로 추정됩니다."
    )


# ── vLLM call ─────────────────────────────────────────────────────────────────
def call_vllm(image_base64_list: list, prompt: str) -> str:
    """
    Send images + prompt to the vLLM endpoint and return the model's response.

    Args:
        image_base64_list: list of base64-encoded JPEG strings
        prompt:            text prompt (from build_prompt)

    Returns:
        Stripped response string from the model.
    """
    content = [{"type": "text", "text": prompt}]
    for b64 in image_base64_list:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    response = requests.post(
        VLLM_API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model":       MODEL_NAME,
            "messages":    [{"role": "user", "content": content}],
            "max_tokens":  32,
            "temperature": 0.0,
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    if "choices" not in data:
        raise ValueError(f"Unexpected API response: {data}")
    return data["choices"][0]["message"]["content"].strip()
