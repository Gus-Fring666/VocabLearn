"""
VocabLearn — Real-time Vocabulary Learning with Your Webcam
===========================================================
Uses YOLOv8 to detect everyday objects through your webcam and teaches
you their English names in a fun, interactive overlay.

Controls:
    Q / ESC  — Quit
    SPACE    — Pause / Resume detection
    S        — Screenshot (saved to ./screenshots/)
    C        — Clear the vocabulary list
    R        — Reset score
"""

import cv2
import numpy as np
import time
import os
import math
from collections import defaultdict
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.45
MODEL_NAME = "yolov8n.pt"  # nano model — fast & lightweight
WINDOW_NAME = "VocabLearn - Webcam Vocabulary Trainer"

# ──────────────────────────────────────────────────────────────
#  Color Palette  (BGR format for OpenCV)
# ──────────────────────────────────────────────────────────────
COLORS = {
    "bg_dark":      (30,  30,  30),
    "bg_panel":     (40,  40,  45),
    "accent":       (250, 160, 50),    # warm orange
    "accent2":      (100, 220, 120),   # green
    "accent3":      (255, 100, 100),   # coral
    "text_white":   (255, 255, 255),
    "text_light":   (200, 200, 210),
    "text_dim":     (130, 130, 140),
    "box_border":   (250, 160, 50),
    "box_fill":     (250, 160, 50),
    "shadow":       (15,  15,  18),
}

# Generate distinct colors for different object classes
def _class_color(class_id: int):
    """Deterministic vibrant color per class id using golden-angle hue spacing."""
    hue = int((class_id * 137.508) % 180)  # golden angle in degrees / 2
    hsv = np.uint8([[[hue, 200, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ──────────────────────────────────────────────────────────────
#  Drawing Helpers
# ──────────────────────────────────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1, alpha=0.85):
    """Draw a filled rounded rectangle with transparency."""
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    # Draw filled rounded rectangle on overlay
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, thickness)
    cv2.circle(overlay, (x1 + r, y1 + r), r, color, thickness)
    cv2.circle(overlay, (x2 - r, y1 + r), r, color, thickness)
    cv2.circle(overlay, (x1 + r, y2 - r), r, color, thickness)
    cv2.circle(overlay, (x2 - r, y2 - r), r, color, thickness)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_label_badge(img, text, pos, bg_color, text_color=(255, 255, 255),
                     font_scale=0.6, padding=8):
    """Draw a text label with a rounded background badge."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)
    x, y = pos
    pt1 = (x, y - th - padding)
    pt2 = (x + tw + padding * 2, y + padding)
    draw_rounded_rect(img, pt1, pt2, bg_color, radius=6, alpha=0.9)
    cv2.putText(img, text, (x + padding, y), font, font_scale, text_color, 1,
                cv2.LINE_AA)


def draw_fancy_box(img, x1, y1, x2, y2, color, thickness=2, corner_len=20):
    """Draw a detection box with stylish corner brackets instead of a full rectangle."""
    cl = min(corner_len, (x2 - x1) // 3, (y2 - y1) // 3)
    t = thickness

    # Top-left
    cv2.line(img, (x1, y1), (x1 + cl, y1), color, t, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + cl), color, t, cv2.LINE_AA)
    # Top-right
    cv2.line(img, (x2, y1), (x2 - cl, y1), color, t, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + cl), color, t, cv2.LINE_AA)
    # Bottom-left
    cv2.line(img, (x1, y2), (x1 + cl, y2), color, t, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - cl), color, t, cv2.LINE_AA)
    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - cl, y2), color, t, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - cl), color, t, cv2.LINE_AA)

    # Subtle dashed center lines
    dash_color = (color[0] // 3, color[1] // 3, color[2] // 3)
    for i in range(x1 + cl, x2 - cl, 8):
        cv2.line(img, (i, y1), (i + 4, y1), dash_color, 1, cv2.LINE_AA)
        cv2.line(img, (i, y2), (i + 4, y2), dash_color, 1, cv2.LINE_AA)
    for i in range(y1 + cl, y2 - cl, 8):
        cv2.line(img, (x1, i), (x1, i + 4), dash_color, 1, cv2.LINE_AA)
        cv2.line(img, (x2, i), (x2, i + 4), dash_color, 1, cv2.LINE_AA)


def draw_progress_bar(img, x, y, w, h, progress, color, bg_color=(60, 60, 65)):
    """Draw a sleek progress bar."""
    cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)
    fill_w = int(w * min(progress, 1.0))
    if fill_w > 0:
        cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 85), 1)


# ──────────────────────────────────────────────────────────────
#  HUD Panels
# ──────────────────────────────────────────────────────────────
def draw_header(img, fps, paused, total_detected, frame_w):
    """Top header bar with app title, FPS, and status."""
    h = 52
    draw_rounded_rect(img, (0, 0), (frame_w, h), COLORS["bg_dark"],
                      radius=0, alpha=0.75)

    # Title
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "VOCABLEARN", (16, 35), font, 0.85,
                COLORS["accent"], 2, cv2.LINE_AA)

    # Webcam icon (small circle)
    cx = 195
    pulse = int(4 * abs(math.sin(time.time() * 3)))
    cv2.circle(img, (cx, 28), 6 + pulse, COLORS["accent3"] if not paused
               else COLORS["text_dim"], -1, cv2.LINE_AA)
    status = "PAUSED" if paused else "LIVE"
    cv2.putText(img, status, (cx + 14, 35), font, 0.5,
                COLORS["accent3"] if not paused else COLORS["text_dim"],
                1, cv2.LINE_AA)

    # FPS
    fps_text = f"FPS: {fps:.0f}"
    cv2.putText(img, fps_text, (frame_w - 130, 35), font, 0.5,
                COLORS["accent2"], 1, cv2.LINE_AA)

    # Object count
    count_text = f"Objects: {total_detected}"
    cv2.putText(img, count_text, (frame_w - 280, 35), font, 0.5,
                COLORS["text_light"], 1, cv2.LINE_AA)


def draw_vocab_panel(img, vocab_dict, frame_h):
    """Right-side vocabulary panel showing discovered words."""
    panel_w = 220
    panel_x = img.shape[1] - panel_w - 10
    panel_y = 62
    panel_h = min(40 + len(vocab_dict) * 30, frame_h - 80)

    draw_rounded_rect(img, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h),
                      COLORS["bg_panel"], radius=10, alpha=0.8)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Panel title
    cv2.putText(img, "Vocabulary", (panel_x + 12, panel_y + 25),
                font, 0.55, COLORS["accent"], 1, cv2.LINE_AA)
    cv2.line(img, (panel_x + 12, panel_y + 32),
             (panel_x + panel_w - 12, panel_y + 32),
             COLORS["text_dim"], 1, cv2.LINE_AA)

    # Word list (sorted by count descending, show top items)
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: -x[1])
    max_items = (panel_h - 50) // 30
    for i, (word, count) in enumerate(sorted_vocab[:max_items]):
        y_pos = panel_y + 55 + i * 30
        # Colored dot
        dot_color = _class_color(hash(word) % 80)
        cv2.circle(img, (panel_x + 20, y_pos - 4), 5, dot_color, -1,
                   cv2.LINE_AA)
        # Word
        cv2.putText(img, word.capitalize(), (panel_x + 32, y_pos),
                    font, 0.45, COLORS["text_white"], 1, cv2.LINE_AA)
        # Count badge
        cv2.putText(img, f"x{count}", (panel_x + panel_w - 45, y_pos),
                    font, 0.38, COLORS["text_dim"], 1, cv2.LINE_AA)

    if len(sorted_vocab) > max_items:
        cv2.putText(img, f"+{len(sorted_vocab) - max_items} more...",
                    (panel_x + 32, panel_y + panel_h - 12),
                    font, 0.35, COLORS["text_dim"], 1, cv2.LINE_AA)


def draw_controls_hint(img, frame_w, frame_h):
    """Bottom bar with keyboard controls."""
    bar_h = 32
    bar_y = frame_h - bar_h
    draw_rounded_rect(img, (0, bar_y), (frame_w, frame_h),
                      COLORS["bg_dark"], radius=0, alpha=0.7)

    font = cv2.FONT_HERSHEY_SIMPLEX
    hints = "Q:Quit  SPACE:Pause  S:Screenshot  C:Clear  R:Reset"
    cv2.putText(img, hints, (16, frame_h - 10), font, 0.38,
                COLORS["text_dim"], 1, cv2.LINE_AA)


def draw_detection_overlay(img, label, conf, x1, y1, x2, y2, class_id):
    """Draw detection box + label for a single object."""
    color = _class_color(class_id)

    # Fancy bracket box
    draw_fancy_box(img, x1, y1, x2, y2, color, thickness=2, corner_len=22)

    # Semi-transparent fill inside the box
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.08, img, 0.92, 0, img)

    # Label badge above the box
    badge_text = f"{label.upper()}  {conf:.0%}"
    draw_label_badge(img, badge_text, (x1, y1 - 4), color,
                     text_color=(255, 255, 255), font_scale=0.55, padding=6)


def draw_welcome_splash(img, frame_w, frame_h):
    """Full screen welcome message when no objects are detected yet."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    msg1 = "Point your camera at everyday objects!"
    msg2 = "VocabLearn will identify them for you."
    (tw1, _), _ = cv2.getTextSize(msg1, font, 0.7, 1)
    (tw2, _), _ = cv2.getTextSize(msg2, font, 0.5, 1)

    cx = frame_w // 2
    cy = frame_h // 2

    # Pulsing circle
    radius = 40 + int(8 * math.sin(time.time() * 2.5))
    cv2.circle(img, (cx, cy - 50), radius, COLORS["accent"], 2, cv2.LINE_AA)
    cv2.putText(img, "?", (cx - 12, cy - 35), font, 1.2,
                COLORS["accent"], 2, cv2.LINE_AA)

    cv2.putText(img, msg1, (cx - tw1 // 2, cy + 30), font, 0.7,
                COLORS["text_white"], 1, cv2.LINE_AA)
    cv2.putText(img, msg2, (cx - tw2 // 2, cy + 60), font, 0.5,
                COLORS["text_dim"], 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────
#  Main Application
# ──────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  VocabLearn — Real-time Vocabulary Trainer")
    print("=" * 55)
    print(f"\n  Loading YOLO model ({MODEL_NAME})...")

    model = YOLO(MODEL_NAME)

    print("  Model loaded successfully!")
    print("  Opening webcam...\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [ERROR] Could not open webcam. Check your camera connection.")
        return

    # Try to set a decent resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # State
    vocab = defaultdict(int)       # word → times detected
    paused = False
    frame_count = 0
    fps = 0.0
    fps_timer = time.time()
    ever_detected = False
    screenshot_dir = os.path.join(os.path.dirname(__file__), "screenshots")

    print("  ✓ Webcam is live! Controls:")
    print("    Q / ESC  — Quit")
    print("    SPACE    — Pause / Resume")
    print("    S        — Screenshot")
    print("    C        — Clear vocabulary")
    print("    R        — Reset score\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("  [ERROR] Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror
        frame_h, frame_w = frame.shape[:2]
        display = frame.copy()

        # ── FPS calculation ──
        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 0.5:
            fps = frame_count / elapsed
            frame_count = 0
            fps_timer = time.time()

        # ── Run YOLO detection ──
        current_detections = 0
        if not paused:
            results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Update vocabulary
                    vocab[label] += 1
                    current_detections += 1
                    ever_detected = True

                    # Draw detection
                    draw_detection_overlay(display, label, conf,
                                           x1, y1, x2, y2, cls_id)

        # ── HUD ──
        draw_header(display, fps, paused, current_detections, frame_w)

        if vocab:
            draw_vocab_panel(display, vocab, frame_h)

        draw_controls_hint(display, frame_w, frame_h)

        if not ever_detected:
            draw_welcome_splash(display, frame_w, frame_h)

        # ── Show frame ──
        cv2.imshow(WINDOW_NAME, display)

        # ── Keyboard input ──
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # Q or ESC
            break
        elif key == ord(' '):
            paused = not paused
            print(f"  {'⏸ Paused' if paused else '▶ Resumed'}")
        elif key == ord('s'):
            os.makedirs(screenshot_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(screenshot_dir, f"vocablearn_{ts}.png")
            cv2.imwrite(path, display)
            print(f"  📸 Screenshot saved: {path}")
        elif key == ord('c'):
            vocab.clear()
            ever_detected = False
            print("  🗑  Vocabulary cleared!")
        elif key == ord('r'):
            vocab = defaultdict(int)
            print("  🔄 Score reset!")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Print final vocab summary
    if vocab:
        print("\n" + "=" * 55)
        print("  Your Vocabulary Session Summary")
        print("=" * 55)
        for word, count in sorted(vocab.items(), key=lambda x: -x[1]):
            bar = "█" * min(count, 30)
            print(f"    {word.capitalize():20s} {bar} ({count})")
        print(f"\n  Total unique words learned: {len(vocab)}")
    print("\n  Thanks for using VocabLearn! 👋\n")


if __name__ == "__main__":
    main()
