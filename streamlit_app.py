"""
VocabLearn — Streamlit Edition
===============================
Real-time vocabulary learning powered by YOLOv8 object detection.
Point your webcam at everyday objects and learn their English names!

Run:  streamlit run streamlit_app.py
"""

import streamlit as st
import cv2
import numpy as np
import time
import os
import math
import json
import base64
from io import BytesIO
from datetime import datetime
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────
#  Page Configuration
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VocabLearn — AI Vocabulary Trainer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────
MODEL_NAME = "yolov8n.pt"
# Use local file if available, otherwise Ultralytics will auto-download
_local_model = os.path.join(os.path.dirname(__file__), MODEL_NAME)
MODEL_PATH = _local_model if os.path.exists(_local_model) else MODEL_NAME
SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "screenshots")

# ──────────────────────────────────────────────────────────────
#  Custom CSS — Premium Dark Theme
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Global ── */
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: #1a1a25;
        --bg-card-hover: #22222f;
        --accent-orange: #f5a623;
        --accent-green: #4ecdc4;
        --accent-coral: #ff6b6b;
        --accent-purple: #a855f7;
        --accent-blue: #3b82f6;
        --text-primary: #f0f0f5;
        --text-secondary: #a0a0b0;
        --text-dim: #6b6b80;
        --border-subtle: #2a2a3a;
        --gradient-warm: linear-gradient(135deg, #f5a623, #ff6b6b);
        --gradient-cool: linear-gradient(135deg, #4ecdc4, #3b82f6);
        --gradient-purple: linear-gradient(135deg, #a855f7, #6366f1);
        --shadow-card: 0 4px 24px rgba(0,0,0,0.4);
        --shadow-glow: 0 0 30px rgba(245, 166, 35, 0.15);
    }

    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Remove default padding ── */
    .block-container {
        padding-top: 1rem !important;
        max-width: 100% !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: var(--text-secondary) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--text-primary) !important;
    }

    /* ── Hero Header ── */
    .hero-header {
        background: linear-gradient(135deg, #12121a 0%, #1a1028 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(245,166,35,0.06) 0%, transparent 50%),
                    radial-gradient(circle at 70% 80%, rgba(78,205,196,0.04) 0%, transparent 50%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: var(--gradient-warm);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(245,166,35,0.12);
        border: 1px solid rgba(245,166,35,0.25);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.75rem;
        color: var(--accent-orange);
        font-weight: 600;
        margin-top: 0.8rem;
    }

    /* ── Stat Cards ── */
    .stat-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-card);
    }
    .stat-card:hover {
        background: var(--bg-card-hover);
        border-color: rgba(245,166,35,0.3);
        box-shadow: var(--shadow-glow);
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 800;
        background: var(--gradient-warm);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-value.green {
        background: var(--gradient-cool);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-value.purple {
        background: var(--gradient-purple);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        color: var(--text-dim);
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-top: 4px;
    }

    /* ── Vocab Table ── */
    .vocab-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: var(--shadow-card);
    }
    .vocab-card-title {
        color: var(--accent-orange);
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .vocab-word {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.6rem 0.8rem;
        border-radius: 8px;
        margin-bottom: 4px;
        transition: background 0.15s ease;
    }
    .vocab-word:hover {
        background: rgba(245,166,35,0.06);
    }
    .vocab-word-name {
        color: var(--text-primary);
        font-weight: 500;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .vocab-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    .vocab-count {
        color: var(--text-dim);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
    }
    .vocab-bar {
        height: 4px;
        border-radius: 2px;
        background: var(--border-subtle);
        flex: 1;
        margin: 0 12px;
        overflow: hidden;
    }
    .vocab-bar-fill {
        height: 100%;
        border-radius: 2px;
        background: var(--gradient-warm);
    }

    /* ── Camera Feed Card ── */
    .camera-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 0;
        overflow: hidden;
        box-shadow: var(--shadow-card);
    }
    .camera-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.8rem 1.2rem;
        background: var(--bg-secondary);
        border-bottom: 1px solid var(--border-subtle);
    }
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        color: var(--accent-coral);
        font-weight: 600;
        font-size: 0.85rem;
    }
    .live-dot {
        width: 8px;
        height: 8px;
        background: var(--accent-coral);
        border-radius: 50%;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(255,107,107,0.7); }
        50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(255,107,107,0); }
    }

    /* ── Buttons ── */
    .stButton > button {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.2s ease !important;
        width: 100%;
    }
    .stButton > button:hover {
        border-color: var(--accent-orange) !important;
        background: rgba(245,166,35,0.08) !important;
        box-shadow: 0 0 20px rgba(245,166,35,0.1) !important;
    }

    /* ── Slider ── */
    .stSlider > div > div > div > div {
        background: var(--accent-orange) !important;
    }

    /* ── Detection log ── */
    .detection-log {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 1.2rem;
        max-height: 300px;
        overflow-y: auto;
    }
    .detection-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 6px 0;
        border-bottom: 1px solid rgba(42,42,58,0.5);
        font-size: 0.85rem;
    }
    .detection-time {
        color: var(--text-dim);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        min-width: 65px;
    }
    .detection-name {
        color: var(--text-primary);
        font-weight: 500;
    }
    .detection-conf {
        color: var(--accent-green);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        margin-left: auto;
    }

    /* ── Info Cards ── */
    .info-card {
        background: linear-gradient(135deg, rgba(245,166,35,0.08), rgba(78,205,196,0.05));
        border: 1px solid rgba(245,166,35,0.15);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    .info-card h4 {
        color: var(--accent-orange);
        margin: 0 0 0.5rem 0;
    }
    .info-card p {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin: 0;
    }

    /* ── Screenshot Gallery ── */
    .screenshot-gallery {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 12px;
    }
    .screenshot-item {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid var(--border-subtle);
        transition: transform 0.2s;
    }
    .screenshot-item:hover { transform: scale(1.02); }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: var(--text-dim);
        font-size: 0.8rem;
        border-top: 1px solid var(--border-subtle);
        margin-top: 2rem;
    }

    /* ── Hide Streamlit Defaults ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
#  Model Loading (cached)
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load YOLOv8 model (cached across reruns)."""
    model = YOLO(MODEL_PATH)
    return model


# ──────────────────────────────────────────────────────────────
#  Helper Functions
# ──────────────────────────────────────────────────────────────
def get_class_color(class_id: int):
    """Deterministic vibrant color per class using golden-angle hue."""
    hue = (class_id * 137.508) % 360
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue / 360, 0.75, 0.9)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def get_class_color_bgr(class_id: int):
    """BGR color for OpenCV drawing."""
    hue = int((class_id * 137.508) % 180)
    hsv = np.uint8([[[hue, 200, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_detection_on_frame(frame, label, conf, x1, y1, x2, y2, class_id):
    """Draw a stylish detection box onto the frame."""
    color = get_class_color_bgr(class_id)

    # Corner bracket style
    cl = min(22, (x2 - x1) // 3, (y2 - y1) // 3)
    t = 2

    # Top-left
    cv2.line(frame, (x1, y1), (x1 + cl, y1), color, t, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + cl), color, t, cv2.LINE_AA)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - cl, y1), color, t, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + cl), color, t, cv2.LINE_AA)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + cl, y2), color, t, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - cl), color, t, cv2.LINE_AA)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - cl, y2), color, t, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - cl), color, t, cv2.LINE_AA)

    # Semi-transparent fill
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

    # Label badge
    badge_text = f"{label.upper()} {conf:.0%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(badge_text, font, 0.55, 1)
    pad = 6
    cv2.rectangle(frame, (x1, y1 - th - pad * 2), (x1 + tw + pad * 2, y1), color, -1)
    cv2.putText(frame, badge_text, (x1 + pad, y1 - pad), font, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)


def frame_to_base64(frame):
    """Convert an OpenCV frame to a base64 PNG for display."""
    _, buffer = cv2.imencode('.png', frame)
    return base64.b64encode(buffer).decode('utf-8')


def pil_to_base64(image):
    """Convert PIL Image to base64."""
    buf = BytesIO()
    image.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# ──────────────────────────────────────────────────────────────
#  Session State Initialization
# ──────────────────────────────────────────────────────────────
if "vocab" not in st.session_state:
    st.session_state.vocab = defaultdict(int)
if "detection_log" not in st.session_state:
    st.session_state.detection_log = []
if "total_detections" not in st.session_state:
    st.session_state.total_detections = 0
if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.now()
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "screenshots" not in st.session_state:
    st.session_state.screenshots = []
if "captured_frame" not in st.session_state:
    st.session_state.captured_frame = None
if "last_frame_detections" not in st.session_state:
    st.session_state.last_frame_detections = []


# ──────────────────────────────────────────────────────────────
#  Sidebar — Controls & Settings
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size: 2.5rem;">📚</div>
        <h2 style="margin: 0.3rem 0; background: linear-gradient(135deg, #f5a623, #ff6b6b);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-weight: 800;">VocabLearn</h2>
        <p style="color: #6b6b80; font-size: 0.85rem;">AI-Powered Vocabulary Trainer</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Detection Settings ──
    st.markdown("### ⚙️ Detection Settings")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.1, max_value=0.95, value=0.45, step=0.05,
        help="Minimum confidence score for object detection. Higher = more accurate but fewer detections."
    )

    camera_index = st.selectbox(
        "Camera Source",
        options=[0, 1, 2],
        format_func=lambda x: f"Camera {x}" + (" (Default)" if x == 0 else ""),
        help="Select which camera to use."
    )

    st.markdown("---")

    # ── Session Controls ──
    st.markdown("### 🎮 Controls")

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        if st.button("🗑️ Clear Vocab", use_container_width=True):
            st.session_state.vocab = defaultdict(int)
            st.session_state.detection_log = []
            st.session_state.total_detections = 0
            st.rerun()

    with col_ctrl2:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.vocab = defaultdict(int)
            st.session_state.detection_log = []
            st.session_state.total_detections = 0
            st.session_state.session_start = datetime.now()
            st.session_state.screenshots = []
            st.session_state.captured_frame = None
            st.session_state.last_frame_detections = []
            st.rerun()

    st.markdown("---")

    # ── Session Info ──
    st.markdown("### 📊 Session Info")
    session_duration = datetime.now() - st.session_state.session_start
    mins = int(session_duration.total_seconds() // 60)
    secs = int(session_duration.total_seconds() % 60)
    st.markdown(f"""
    <div class="info-card">
        <p>⏱️ <strong>Duration:</strong> {mins}m {secs}s</p>
        <p>📸 <strong>Screenshots:</strong> {len(st.session_state.screenshots)}</p>
        <p>🔍 <strong>Model:</strong> YOLOv8 Nano</p>
        <p>🎯 <strong>Threshold:</strong> {confidence:.0%}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Export ──
    st.markdown("### 💾 Export Data")
    if st.session_state.vocab:
        vocab_data = json.dumps(dict(st.session_state.vocab), indent=2)
        st.download_button(
            "📥 Download Vocabulary (JSON)",
            data=vocab_data,
            file_name=f"vocablearn_vocab_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.markdown('<p style="color: #6b6b80; font-size: 0.85rem;">No vocabulary data yet.</p>',
                    unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
#  Main Content — Hero Header
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">📚 VocabLearn</h1>
    <p class="hero-subtitle">Point your webcam at everyday objects and learn their English names in real-time using AI-powered object detection.</p>
    <div class="hero-badge">
        <span>⚡</span> Powered by YOLOv8 • Real-time Detection
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
#  Stats Row
# ──────────────────────────────────────────────────────────────
stat_cols = st.columns(4)

with stat_cols[0]:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{len(st.session_state.vocab)}</div>
        <div class="stat-label">Words Learned</div>
    </div>
    """, unsafe_allow_html=True)

with stat_cols[1]:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value green">{st.session_state.total_detections}</div>
        <div class="stat-label">Total Detections</div>
    </div>
    """, unsafe_allow_html=True)

with stat_cols[2]:
    top_word = ""
    if st.session_state.vocab:
        top_word = max(st.session_state.vocab, key=st.session_state.vocab.get).capitalize()
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value purple">{top_word if top_word else '—'}</div>
        <div class="stat-label">Most Detected</div>
    </div>
    """, unsafe_allow_html=True)

with stat_cols[3]:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{len(st.session_state.screenshots)}</div>
        <div class="stat-label">Screenshots</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
#  Main Layout — Camera + Vocabulary
# ──────────────────────────────────────────────────────────────
main_col, vocab_col = st.columns([3, 1.2])

with main_col:
    # Camera card header
    st.markdown("""
    <div class="camera-card">
        <div class="camera-header">
            <div class="live-indicator">
                <span class="live-dot"></span> LIVE CAMERA FEED
            </div>
            <span style="color: #6b6b80; font-size: 0.8rem;">YOLOv8 Nano</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Camera feed placeholder
    camera_placeholder = st.empty()

    # Camera control buttons
    btn_cols = st.columns([1, 1, 1, 1])

    with btn_cols[0]:
        start_btn = st.button("▶️ Start Detection", use_container_width=True, key="start")
    with btn_cols[1]:
        stop_btn = st.button("⏹️ Stop Detection", use_container_width=True, key="stop")
    with btn_cols[2]:
        screenshot_btn = st.button("📸 Screenshot", use_container_width=True, key="screenshot")
    with btn_cols[3]:
        upload_btn = st.button("📁 Upload Image", use_container_width=True, key="upload_toggle")

    # Handle screenshot button
    if screenshot_btn and st.session_state.captured_frame is not None:
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SCREENSHOT_DIR, f"vocablearn_{ts}.png")
        cv2.imwrite(path, st.session_state.captured_frame)
        st.session_state.screenshots.append(path)
        st.toast(f"📸 Screenshot saved!", icon="✅")

    # Handle stop button
    if stop_btn:
        st.session_state.is_running = False

    # Handle start button
    if start_btn:
        st.session_state.is_running = True

    # ── Upload Image Mode ──
    if upload_btn or ("show_upload" in st.session_state and st.session_state.show_upload):
        st.session_state.show_upload = True
        uploaded_file = st.file_uploader(
            "Upload an image to detect objects",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="file_uploader"
        )
        if uploaded_file is not None:
            model = load_model()
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            results = model(img, verbose=False, conf=confidence)
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    label = model.names[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    st.session_state.vocab[label] += 1
                    st.session_state.total_detections += 1
                    draw_detection_on_frame(img, label, conf_val, x1, y1, x2, y2, cls_id)
                    st.session_state.detection_log.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "label": label,
                        "confidence": conf_val,
                    })

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(img_rgb, use_container_width=True)

    # ── Live Camera Mode ──
    if st.session_state.is_running:
        st.session_state.show_upload = False
        model = load_model()
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            st.error("❌ Could not open webcam. Please check your camera connection and try again.")
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            stop_placeholder = st.empty()

            while st.session_state.is_running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Failed to read frame from camera.")
                    break

                frame = cv2.flip(frame, 1)

                # Run YOLO detection
                results = model(frame, verbose=False, conf=confidence)
                current_detections = []

                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf_val = float(box.conf[0])
                        label = model.names[cls_id]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Update vocabulary
                        st.session_state.vocab[label] += 1
                        st.session_state.total_detections += 1
                        current_detections.append({
                            "label": label,
                            "confidence": conf_val,
                            "class_id": cls_id
                        })

                        # Add to detection log (keep last 50)
                        st.session_state.detection_log.append({
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "label": label,
                            "confidence": conf_val,
                        })
                        if len(st.session_state.detection_log) > 50:
                            st.session_state.detection_log = st.session_state.detection_log[-50:]

                        # Draw detection on frame
                        draw_detection_on_frame(frame, label, conf_val, x1, y1, x2, y2, cls_id)

                st.session_state.captured_frame = frame.copy()
                st.session_state.last_frame_detections = current_detections

                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(frame_rgb, use_container_width=True)

                # Small delay to prevent overwhelming
                time.sleep(0.03)

            cap.release()

    # If not running and no upload, show placeholder
    if not st.session_state.is_running and not st.session_state.get("show_upload", False):
        if st.session_state.captured_frame is not None:
            frame_rgb = cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, use_container_width=True)
        else:
            camera_placeholder.markdown("""
            <div style="background: #12121a; border: 2px dashed #2a2a3a; border-radius: 14px;
                        padding: 5rem 2rem; text-align: center;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📷</div>
                <h3 style="color: #a0a0b0; font-weight: 600;">Camera Not Active</h3>
                <p style="color: #6b6b80;">Click <strong>▶️ Start Detection</strong> to begin learning vocabulary,
                or <strong>📁 Upload Image</strong> to analyze a photo.</p>
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
#  Vocabulary Panel (Right Side)
# ──────────────────────────────────────────────────────────────
with vocab_col:
    if st.session_state.vocab:
        sorted_vocab = sorted(st.session_state.vocab.items(), key=lambda x: -x[1])
        max_count = sorted_vocab[0][1] if sorted_vocab else 1

        vocab_html = '<div class="vocab-card"><div class="vocab-card-title">📖 Vocabulary List</div>'

        for word, count in sorted_vocab[:20]:
            color = get_class_color(hash(word) % 80)
            bar_pct = (count / max_count) * 100

            vocab_html += f"""
            <div class="vocab-word">
                <span class="vocab-word-name">
                    <span class="vocab-dot" style="background: {color};"></span>
                    {word.capitalize()}
                </span>
                <div class="vocab-bar"><div class="vocab-bar-fill" style="width: {bar_pct}%;"></div></div>
                <span class="vocab-count">×{count}</span>
            </div>
            """

        if len(sorted_vocab) > 20:
            vocab_html += f'<p style="color: #6b6b80; text-align:center; margin-top:8px; font-size:0.8rem;">+{len(sorted_vocab)-20} more words...</p>'

        vocab_html += '</div>'
        st.markdown(vocab_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="vocab-card" style="text-align: center; padding: 2rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📖</div>
            <div class="vocab-card-title" style="justify-content: center;">Vocabulary List</div>
            <p style="color: #6b6b80; font-size: 0.9rem;">
                No words detected yet.<br>Start the camera to begin learning!
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Detection Log ──
    st.markdown("""
    <div class="vocab-card">
        <div class="vocab-card-title">📋 Detection Log</div>
    """, unsafe_allow_html=True)

    if st.session_state.detection_log:
        log_html = ""
        for entry in reversed(st.session_state.detection_log[-15:]):
            log_html += f"""
            <div class="detection-item">
                <span class="detection-time">{entry['time']}</span>
                <span class="detection-name">{entry['label'].capitalize()}</span>
                <span class="detection-conf">{entry['confidence']:.0%}</span>
            </div>
            """
        st.markdown(log_html, unsafe_allow_html=True)
    else:
        st.markdown('<p style="color: #6b6b80; font-size: 0.85rem; text-align:center;">No detections yet.</p>',
                    unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
#  Screenshot Gallery
# ──────────────────────────────────────────────────────────────
if st.session_state.screenshots:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="vocab-card">
        <div class="vocab-card-title">🖼️ Screenshot Gallery</div>
    </div>
    """, unsafe_allow_html=True)

    gallery_cols = st.columns(min(len(st.session_state.screenshots), 4))
    for i, path in enumerate(st.session_state.screenshots[-4:]):
        with gallery_cols[i]:
            if os.path.exists(path):
                img = Image.open(path)
                st.image(img, use_container_width=True, caption=os.path.basename(path))

                # Download button for each screenshot
                buf = BytesIO()
                img.save(buf, format="PNG")
                st.download_button(
                    f"📥 Download",
                    data=buf.getvalue(),
                    file_name=os.path.basename(path),
                    mime="image/png",
                    key=f"dl_{i}",
                    use_container_width=True,
                )


# ──────────────────────────────────────────────────────────────
#  How It Works Section
# ──────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

with st.expander("ℹ️ How It Works", expanded=False):
    how_cols = st.columns(3)

    with how_cols[0]:
        st.markdown("""
        <div class="info-card">
            <h4>📷 1. Capture</h4>
            <p>Your webcam captures a live video feed that is processed frame-by-frame in real-time.</p>
        </div>
        """, unsafe_allow_html=True)

    with how_cols[1]:
        st.markdown("""
        <div class="info-card">
            <h4>🤖 2. Detect</h4>
            <p>YOLOv8, a state-of-the-art object detection model, identifies objects in each frame with high accuracy.</p>
        </div>
        """, unsafe_allow_html=True)

    with how_cols[2]:
        st.markdown("""
        <div class="info-card">
            <h4>📝 3. Learn</h4>
            <p>Each detected object's English name is displayed with a confidence score, building your vocabulary over time.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    **Supported Objects:** VocabLearn can detect **80+ everyday objects** including people, animals,
    vehicles, furniture, electronics, food items, and more — all from the COCO dataset.
    """)


# ──────────────────────────────────────────────────────────────
#  Footer
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <p>Built with ❤️ using <strong>Streamlit</strong> + <strong>YOLOv8</strong> by Ultralytics</p>
    <p>VocabLearn © 2026 • AI-Powered Vocabulary Learning</p>
</div>
""", unsafe_allow_html=True)
