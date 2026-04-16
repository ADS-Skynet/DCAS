#!/usr/bin/env python3
"""
VLM_analyze.py
세션/이벤트 폴더를 받아 운전자 상태를 분석합니다.

Usage:
    python VLM_analyze.py captures/session_20260414_135500
    python VLM_analyze.py captures/session_20260414_135500/event_001

토큰 설계 (한도 1000):
    이미지 224x168 = 48토큰/장 x 16장 = 768토큰
    프롬프트 ~200토큰
    출력 32토큰
    합계 ~1000토큰
"""

import sys
import os
import base64
import json
import requests
from io import BytesIO
from PIL import Image

# ── vLLM 설정 ─────────────────────────────────────────────────────────────────
VLLM_API_URL = "http://210.121.152.22:9000/v1/chat/completions"
API_KEY      = "81cbe888efea8c89da139c5cc8194393c1ead203e11e85a9a5a721428c5a2517"
MODEL_NAME   = "Qwen/Qwen3-VL-2B-Instruct"

IMAGE_W = 224   # 실측 기준: 320x240=97토큰/장, 224x168=약 48토큰/장
IMAGE_H = 168
MAX_IMAGES = 7  # 프롬프트200 + 이미지7장x48 = ~536토큰 (여유있게 한도내)


# ── 이미지 유틸 ───────────────────────────────────────────────────────────────
def image_to_base64(image_path: str) -> str:
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((IMAGE_W, IMAGE_H), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── 프롬프트 빌더 ─────────────────────────────────────────────────────────────
def build_prompt(all_params: list) -> str:
    if not all_params:
        return "데이터없음. 판단불가."

    avg  = lambda k: sum(p.get(k, 0) for p in all_params) / len(all_params)
    mmax = lambda k: max(p.get(k, 0) for p in all_params)

    perclos_val  = avg('perclos')
    consec_val   = mmax('consec_closed')
    pitch_val    = avg('pitch_droop')
    roll_val     = avg('roll_std')
    yaw_val      = avg('yaw_std')
    jitter_val   = avg('iris_jitter')
    chaos_val    = avg('movement_chaos')
    drowsy_score = mmax('drowsy_score')
    drunk_score  = mmax('drunk_score')

    # 눈 감김 시계열 (0=뜸 1=감김)
    eye_seq = "".join(
        "1" if (p.get("blink_left", 0) + p.get("blink_right", 0)) / 2 > 0.5 else "0"
        for p in all_params
    )

    def flag(val, threshold):
        return "↑" if val >= threshold else "-"

    stats = (
        f"PERCLOS={perclos_val:.2f}{flag(perclos_val,0.25)}(임계0.25)"
        f"|연속눈감김={consec_val}f{flag(consec_val,15)}(임계15f)"
        f"|눈감김시계열={eye_seq}"
        f"|pitch(앞뒤)={pitch_val:.3f}{flag(pitch_val,0.05)}(임계0.05)"
        f"|roll(좌우기울)={roll_val:.1f}deg{flag(roll_val,8)}(임계8)"
        f"|yaw(좌우회전)={yaw_val:.1f}deg{flag(yaw_val,8)}(임계8)"
        f"|눈동자떨림={jitter_val:.3f}{flag(jitter_val,0.05)}(임계0.05)"
        f"|움직임혼란={chaos_val:.2f}{flag(chaos_val,1.0)}(임계1.0)"
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


# ── vLLM 호출 ────────────────────────────────────────────────────────────────
def call_vllm(image_base64_list: list, prompt: str) -> str:
    content = [{"type": "text", "text": prompt}]
    for b64 in image_base64_list:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
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
    result = response.json()
    if "choices" not in result:
        raise ValueError(f"Unexpected API response: {result}")
    return result["choices"][0]["message"]["content"].strip()


# ── 이벤트 로더 ───────────────────────────────────────────────────────────────
def load_event(event_dir: str):
    meta_path = os.path.join(event_dir, "event_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"event_meta.json not found in {event_dir}")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    images_b64, all_params = [], []
    total_frames = meta.get("num_frames", 10)

    # 저장된 프레임에서 MAX_IMAGES장을 균등 샘플링
    if MAX_IMAGES == 1:
        indices = [total_frames // 2]  # 중간 프레임 1장
    else:
        indices = [round(i * (total_frames - 1) / (MAX_IMAGES - 1)) for i in range(MAX_IMAGES)]
    indices = sorted(set(indices))

    for i in indices:
        img_path  = os.path.join(event_dir, f"frame_{i:02d}.jpg")
        json_path = os.path.join(event_dir, f"frame_{i:02d}.json")

        if not os.path.exists(img_path):
            print(f"  [WARN] {img_path} not found, skipping")
            continue

        images_b64.append(image_to_base64(img_path))
        if os.path.exists(json_path):
            with open(json_path, encoding="utf-8") as f:
                all_params.append(json.load(f))
        else:
            all_params.append({})

    return images_b64, all_params, meta


# ── 결과 저장 ─────────────────────────────────────────────────────────────────
def save_result(event_dir: str, result_text: str, meta: dict):
    output = {
        "event_id":     meta.get("event_id"),
        "timestamp":    meta.get("timestamp"),
        "alert_score":  meta.get("alert_score"),
        "driver_state": result_text,
    }
    with open(os.path.join(event_dir, "vlm_result.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    state_path = os.path.expanduser("~/DCAS/driver_state.json")
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump({"driver_state": result_text}, f, ensure_ascii=False, indent=2)


# ── 이벤트 폴더 목록 추출 ────────────────────────────────────────────────────
def get_event_dirs(path: str) -> list:
    if os.path.exists(os.path.join(path, "event_meta.json")):
        return [path]

    events = sorted([
        os.path.join(path, d)
        for d in os.listdir(path)
        if d.startswith("event_") and os.path.isdir(os.path.join(path, d))
    ])

    if not events:
        raise ValueError(f"event_* 폴더를 찾을 수 없습니다: {path}")

    return events


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: python VLM_analyze.py <session_dir | event_dir>")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.isdir(input_path):
        print(f"Error: '{input_path}' is not a directory.")
        sys.exit(1)

    event_dirs = get_event_dirs(input_path)
    total      = len(event_dirs)
    print(f"[INFO] {total}개 이벤트 분석 시작\n")

    results = []

    for idx, event_dir in enumerate(event_dirs, 1):
        event_name = os.path.basename(event_dir)
        print(f"[{idx}/{total}] {event_name} 분석 중...")

        try:
            images_b64, all_params, meta = load_event(event_dir)
            prompt = build_prompt(all_params)
            result = call_vllm(images_b64, prompt)
            save_result(event_dir, result, meta)

            print(f"  → {result}\n")
            results.append({"event": event_name, "state": result, "error": None})

        except Exception as e:
            print(f"  [ERROR] {e}\n")
            results.append({"event": event_name, "state": None, "error": str(e)})

    print("=" * 45)
    print("분석 완료 요약")
    print("=" * 45)
    for r in results:
        status = r["state"] if r["state"] else f"실패: {r['error']}"
        print(f"  {r['event']} : {status}")
    print("=" * 45)


if __name__ == "__main__":
    main()