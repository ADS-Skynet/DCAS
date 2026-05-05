# DCAS
Driver Control Assistance Systems project

## VLM (Vision Language Model)

DCAS uses a Vision Language Model to analyze camera frames and detect abnormal driver states. The model receives an image and returns a structured classification used by the rest of the system.

### vLLM.py

Sends a camera image to a remote **Qwen3-VL-2B-Instruct** model hosted via a vLLM server and classifies the driver state into one of three categories:

| Category | Description |
|---|---|
| `BLOCKED_LENS` | Image is blurry or out of focus — camera may be obstructed |
| `FOGGY_LENS` | Image has a hazy/milky overlay — lens condensation or dirt |
| `DRIVER_UNCONSCIOUS` | Image is clear but driver is slumped/drooping |

**Usage:**
```bash
python vLLM.py <image_file_or_url>
```

**What it does:**
1. Accepts a local image path or URL
2. Resizes the image to at most 768px on the longest side to reduce token usage
3. Encodes it as base64 and sends it to the vLLM Chat Completions API
4. Prints the JSON result and writes it to `~/DCAS/driver_state.json`

**Output format (`driver_state.json`):**
```json
{
  "driver_state": "{\"category\": \"DRIVER_UNCONSCIOUS\", \"confidence\": 0.95}"
}
```
