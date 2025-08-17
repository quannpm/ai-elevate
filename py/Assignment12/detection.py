"""
Satellite Cloud Detection Demo (OpenAI / OpenAI-compatible endpoint version)

Differences vs Azure version:
- Uses standard OpenAI (or compatible) endpoint with optional custom base_url.
- Manual JSON parsing (no LangChain structured output).
Environment variables (set in .env or system):
  OPENAI_API_KEY   (required)
  OPENAI_MODEL     (required, e.g. gpt-4o-mini)
  OPENAI_BASE_URL  (optional, if using a proxy gateway)

Install dependencies:
  pip install streamlit python-dotenv openai pillow

Run the app:
  streamlit run detection.py
"""

import os
import io
import csv
import json
import base64
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(
    page_title="Satellite Cloud Detection (OpenAI)",
    page_icon="☁️",
    layout="wide"
)

st.title("☁️ Satellite Image Cloud Detection (OpenAI API)")
st.caption("Prototype – LLM vision reasoning (Cloudy / Clear)")

# ---------------------------
# Config & constants
# ---------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # optional
LOG_FILE = "cloud_detection_log.csv"
MAX_FILE_MB = 5
ALLOWED_TYPES = ("jpg", "jpeg", "png")

missing = []
if not OPENAI_API_KEY:
    missing.append("OPENAI_API_KEY")
if not OPENAI_MODEL:
    missing.append("OPENAI_MODEL")

if missing:
    st.error("Missing environment variables: " + ", ".join(missing))
    st.stop()

# ---------------------------
# Initialize OpenAI-compatible client
# ---------------------------
try:
    if OPENAI_BASE_URL:
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    else:
        client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize OpenAI client: {e}")
    st.stop()

# ---------------------------
# CSV helpers
# ---------------------------
def ensure_log_file():
    """Create CSV log with header if absent."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Timestamp", "Filename", "Prediction", "Confidence"])

def append_log(filename: str, prediction: str, confidence: float):
    """Append classification result."""
    ensure_log_file()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.utcnow().isoformat(),
            filename,
            prediction,
            f"{confidence:.4f}"
        ])

def load_log() -> List[Dict[str, Any]]:
    """Return list of log lines."""
    if not os.path.exists(LOG_FILE):
        return []
    rows: List[Dict[str, Any]] = []
    with open(LOG_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

# ---------------------------
# Image overlay utility
# ---------------------------
def overlay_label(img: Image.Image, label: str) -> Image.Image:
    """Draw label rectangle top-left."""
    new_img = img.copy()
    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    pad = 8
    width = draw.textlength(label, font=font)
    height = 14
    draw.rectangle([0, 0, width + pad * 2, height + pad * 2], fill=(0, 0, 0, 190))
    draw.text((pad, pad), label, fill=(255, 255, 255), font=font)
    return new_img

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("Options")
    add_overlay = st.toggle("Overlay label", value=True)
    show_raw = st.toggle("Show raw model text", value=False)
    st.markdown("---")
    st.markdown("**Disclaimer:** Confidence is self-reported heuristic.")
    if st.button("Show Log File Path"):
        st.info(os.path.abspath(LOG_FILE))

# ---------------------------
# File uploader
# ---------------------------
uploads = st.file_uploader(
    "Upload satellite image(s)",
    type=ALLOWED_TYPES,
    accept_multiple_files=True
)

col_left, col_right = st.columns([3, 2])

# ---------------------------
# Classification (OpenAI chat)
# ---------------------------
SYSTEM_PROMPT = """
You are an assistant performing binary cloud cover classification on a single satellite image.
Return STRICT JSON ONLY like:
{
  "result": "Clear" | "Cloudy",
  "confidence": <float 0..1>
}
Definitions:
- Clear: Little or no visible clouds; surface mostly unobstructed.
- Cloudy: Noticeable cloud formations obscuring a significant portion.
Rules:
- Output ONLY JSON. No markdown, no extra explanation.
- Confidence is a heuristic internal estimate (not calibrated).
"""

USER_INSTRUCTION = "Classify the image as 'Clear' or 'Cloudy' and return JSON only."

def classify_image(pil_img: Image.Image) -> dict:
    """Send image to the model and parse JSON response."""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_INSTRUCTION},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        }
    ]

    # Build kwargs safely (some proxy models reject temperature != default)
    chat_kwargs = {
        "model": OPENAI_MODEL,
        "messages": messages
    }
    env_temp = os.getenv("OPENAI_TEMPERATURE")
    if env_temp:
        try:
            chat_kwargs["temperature"] = float(env_temp)
        except:
            pass

    try:
        resp = client.chat.completions.create(**chat_kwargs)
    except Exception as e:
        # Retry once without temperature if that's the cause
        if "temperature" in chat_kwargs:
            chat_kwargs.pop("temperature", None)
            resp = client.chat.completions.create(**chat_kwargs)
        else:
            raise e

    raw_text = resp.choices[0].message.content.strip()

    # Attempt JSON parse
    parsed = {}
    try:
        parsed = json.loads(raw_text)
    except Exception:
        # Fallback: naive repair (remove trailing text)
        try:
            first_brace = raw_text.find("{")
            last_brace = raw_text.rfind("}")
            if first_brace != -1 and last_brace != -1:
                snippet = raw_text[first_brace:last_brace+1]
                parsed = json.loads(snippet)
        except Exception:
            pass

    # Validate & normalize
    result = parsed.get("result", "").strip()
    confidence = parsed.get("confidence", 0.0)
    if result not in ("Clear", "Cloudy"):
        # Basic heuristic fallback (NOT robust)
        lc = raw_text.lower()
        result = "Cloudy" if "cloudy" in lc else "Clear"
    try:
        confidence = float(confidence)
        if not (0 <= confidence <= 1):
            confidence = 0.0
    except:
        confidence = 0.0

    return {"raw_text": raw_text, "result": result, "confidence": confidence}

# ---------------------------
# Main action
# ---------------------------
if uploads and st.button("🔍 Run Cloud Detection"):
    ensure_log_file()
    display_records = []
    with st.spinner("Classifying images..."):
        for up in uploads:
            size_mb = len(up.getbuffer()) / (1024 * 1024)
            if size_mb > MAX_FILE_MB:
                st.warning(f"Skip {up.name}: {size_mb:.2f}MB > {MAX_FILE_MB}MB limit.")
                continue
            try:
                pil_img = Image.open(up).convert("RGB")
            except Exception as e:
                st.warning(f"Skip {up.name}: cannot open ({e}).")
                continue

            try:
                res = classify_image(pil_img)
            except Exception as e:
                st.error(f"Fail classify {up.name}: {e}")
                continue

            append_log(up.name, res["result"], res["confidence"])

            label = f"{res['result']} ({res['confidence']*100:.1f}%)"
            annotated = overlay_label(pil_img, label) if add_overlay else pil_img

            display_records.append({
                "filename": up.name,
                "image": annotated,
                "prediction": res["result"],
                "confidence": res["confidence"],
                "raw": res["raw_text"]
            })

    if display_records:
        st.subheader("Results")
        grid_cols = st.columns(min(3, len(display_records)))
        for i, rec in enumerate(display_records):
            c = grid_cols[i % len(grid_cols)]
            with c:
                st.image(rec["image"], caption=rec["filename"], use_container_width=True)
                st.markdown(f"**Prediction:** {rec['prediction']}")
                st.markdown(f"**Confidence:** {rec['confidence']*100:.1f}%")
                if show_raw:
                    st.code(rec["raw"], language="json")

# ---------------------------
# Log viewer
# ---------------------------
with col_right:
    st.subheader("📄 Log")
    rows = load_log()
    if rows:
        st.dataframe(rows, use_container_width=True, height=350)
        st.download_button(
            "Download Log CSV",
            data=open(LOG_FILE, "rb").read(),
            file_name=LOG_FILE,
            mime="text/csv"
        )
    else:
        st.info("No log entries yet.")

st.markdown("---")
st.caption("© 2025 Cloud Detection – OpenAI / compatible endpoint demo")