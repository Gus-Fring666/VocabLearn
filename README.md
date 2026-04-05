<![CDATA[<div align="center">

# 📚 VocabLearn

### AI-Powered Real-Time Vocabulary Trainer

*Point your webcam at everyday objects and learn their English names instantly using state-of-the-art object detection.*

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)

---

</div>

## 🌟 Overview

**VocabLearn** is an interactive vocabulary learning application that leverages the power of **YOLOv8** (You Only Look Once) — a cutting-edge real-time object detection model — to identify everyday objects through your webcam or uploaded images. Each detected object is labeled with its English name and confidence score, helping you build your vocabulary naturally and intuitively.

The app features a premium **dark-themed UI** built with **Streamlit**, complete with live camera feeds, vocabulary tracking, detection logging, screenshot capture, and data export capabilities.

---

## ✨ Features

### 🎥 Real-Time Detection
- **Live webcam feed** with frame-by-frame YOLOv8 inference
- **Stylish detection boxes** with corner-bracket overlays and semi-transparent fills
- **Confidence scores** displayed for every detected object
- Support for **multiple camera sources** (built-in, external, USB cameras)

### 📖 Vocabulary Tracking
- **Automatic word collection** — every detected object is added to your vocabulary list
- **Detection count** per word with visual progress bars
- **Color-coded entries** for easy visual distinction
- **Top 20 most-detected** words displayed in a sleek side panel

### 📊 Session Statistics
- **Words Learned** — unique vocabulary count
- **Total Detections** — cumulative detection counter
- **Most Detected** — your most frequently seen object
- **Screenshot Count** — number of captures taken
- **Session Duration** — live timer tracking

### 📸 Screenshot System
- **One-click capture** — save the current detection frame instantly
- **Screenshot gallery** — view all captures in a grid layout
- **Individual downloads** — download any screenshot as PNG
- Automatic file naming with timestamps

### 📁 Image Upload Mode
- **Upload any image** (JPG, PNG, BMP, WebP) for offline detection
- Full YOLO analysis with the same detection visualization
- Perfect for analyzing images when a webcam isn't available

### 💾 Data Export
- **Export vocabulary as JSON** — download your word list with detection counts
- Timestamped filenames for easy organization
- Ready for integration with other learning tools or flashcard apps

### ⚙️ Adjustable Settings
- **Confidence threshold slider** (10%–95%) to fine-tune detection sensitivity
- **Camera source selector** — easily switch between connected cameras
- All settings update in real-time without restarting

---

## 🤖 Object Detection

VocabLearn uses the **YOLOv8 Nano** model (`yolov8n.pt`), which can detect **80 everyday object categories** from the COCO dataset:

<details>
<summary><strong>📋 Click to see all 80 detectable objects</strong></summary>

| Category | Objects |
|----------|---------|
| **People & Animals** | person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe |
| **Vehicles** | bicycle, car, motorcycle, airplane, bus, train, truck, boat |
| **Outdoor** | traffic light, fire hydrant, stop sign, parking meter, bench |
| **Food** | banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake |
| **Kitchen** | bottle, wine glass, cup, fork, knife, spoon, bowl |
| **Electronics** | TV/monitor, laptop, mouse, remote, keyboard, cell phone |
| **Furniture** | chair, couch, bed, dining table, toilet |
| **Indoor** | potted plant, book, clock, vase, scissors, teddy bear, hair dryer, toothbrush |
| **Sports** | frisbee, skis, snowboard, sports ball, kite, baseball bat/glove, skateboard, surfboard, tennis racket |
| **Accessories** | backpack, umbrella, handbag, tie, suitcase |

</details>

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9** or higher
- A **webcam** (built-in or external) for live detection
- **pip** package manager

### Installation

1. **Clone or download the repository:**
   ```bash
   git clone https://github.com/yourusername/VocabLearn.git
   cd VocabLearn
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install streamlit ultralytics opencv-python numpy Pillow
   ```

3. **Download the YOLO model** (auto-downloads on first run):
   
   The `yolov8n.pt` model file will be automatically downloaded by Ultralytics on the first launch. If you already have it, place it in the project root.

### Running the Application

#### 🖥️ Streamlit App (Recommended)
```bash
streamlit run streamlit_app.py
```
The app will open in your browser at `http://localhost:8501`.

#### 🎮 OpenCV Desktop App (Original)
```bash
python app.py
```
Uses OpenCV's native window system with keyboard controls.

---

## 🎮 Usage Guide

### Live Detection Mode
1. Click **▶️ Start Detection** to activate your webcam
2. Point the camera at everyday objects around you
3. Watch as YOLOv8 identifies objects in real-time with bounding boxes
4. Your vocabulary list builds automatically on the right panel
5. Click **⏹️ Stop Detection** to pause the camera feed

### Image Upload Mode
1. Click **📁 Upload Image** to switch to upload mode
2. Drag & drop or browse for an image file
3. The app will analyze the image and display all detected objects
4. Detected words are added to your vocabulary list

### Taking Screenshots
1. While the camera is running, click **📸 Screenshot**
2. The current frame (with detections) is saved to the `screenshots/` folder
3. View all screenshots in the **Screenshot Gallery** section
4. Download individual screenshots using the **📥 Download** buttons

### Adjusting Settings
- Use the **Confidence Threshold** slider in the sidebar to control detection sensitivity
  - **Higher values** (0.7+): Fewer detections, but more accurate
  - **Lower values** (0.2–0.4): More detections, but may include false positives
  - **Recommended**: 0.45 (default) for a good balance

### Exporting Data
- Click **📥 Download Vocabulary (JSON)** in the sidebar to save your word list
- The exported JSON file contains each word and its detection count

---

## 📁 Project Structure

```
VocabLearn/
├── streamlit_app.py      # 🖥️  Streamlit web application (main)
├── app.py                # 🎮  Original OpenCV desktop application
├── requirements.txt      # 📦  Python dependencies
├── README.md             # 📖  This documentation file
├── yolov8n.pt            # 🤖  YOLOv8 Nano model weights
└── screenshots/          # 📸  Saved screenshot captures
    └── vocablearn_*.png
```

---

## 🛠️ Technology Stack

| Technology | Purpose |
|------------|---------|
| **[Streamlit](https://streamlit.io)** | Web application framework with reactive UI |
| **[YOLOv8](https://ultralytics.com)** | State-of-the-art real-time object detection |
| **[OpenCV](https://opencv.org)** | Computer vision, video capture & image processing |
| **[NumPy](https://numpy.org)** | Numerical computations and array operations |
| **[Pillow](https://pillow.readthedocs.io)** | Image handling and format conversion |
| **[Python](https://python.org)** | Core programming language (3.9+) |

---

## ⚙️ Configuration

### Model Options
You can switch to a more accurate (but slower) YOLO model by changing the `MODEL_NAME` in the source:

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `yolov8n.pt` | 6 MB | ⚡ Fastest | Good |
| `yolov8s.pt` | 22 MB | 🔥 Fast | Better |
| `yolov8m.pt` | 52 MB | ⚡ Medium | Great |
| `yolov8l.pt` | 87 MB | 🐢 Slower | Excellent |
| `yolov8x.pt` | 131 MB | 🐌 Slowest | Best |

### Camera Configuration
- Default camera index is `0` (built-in webcam)
- Switch cameras using the **Camera Source** dropdown in the sidebar
- Resolution defaults to 1280×720 (adjusts automatically if unsupported)

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Camera not opening** | Check if another application is using the camera. Try a different camera index. |
| **Slow performance** | Ensure you're using `yolov8n.pt` (Nano). Close other resource-heavy applications. |
| **No detections** | Lower the confidence threshold in the sidebar. Ensure objects are well-lit and in frame. |
| **Model download fails** | Check your internet connection. The model auto-downloads on first run. |
| **Import errors** | Run `pip install -r requirements.txt` to install all dependencies. |
| **Black/frozen camera feed** | Try restarting the app. Some cameras need a warm-up period. |

---

## 📝 Keyboard Shortcuts (Desktop App)

The original `app.py` desktop version supports these keyboard controls:

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit the application |
| `SPACE` | Pause / Resume detection |
| `S` | Take a screenshot |
| `C` | Clear vocabulary list |
| `R` | Reset score counter |

---

## 🤝 Contributing

Contributions are welcome! Here are some ideas for improvements:

- 🌍 **Multi-language support** — Translate detected object names into other languages
- 🎯 **Quiz mode** — Test your knowledge by naming objects before the AI reveals them
- 📊 **Analytics dashboard** — Learning progress over multiple sessions
- 🔊 **Text-to-speech** — Pronounce detected words aloud
- 🏷️ **Custom model training** — Train on domain-specific objects (medical, culinary, etc.)
- 📱 **Mobile optimization** — Responsive design for phone/tablet browsers

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

**Built with ❤️ using Streamlit + YOLOv8**

*VocabLearn © 2026 — AI-Powered Vocabulary Learning*

</div>
]]>
