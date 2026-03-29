# 🏏 Bowling Biomechanics Analyzer

> Real-time cricket bowling action analysis using **YOLOv8 + MediaPipe Pose** — detects bowling phases, joint angles, Center of Mass trajectory, and angular velocity from video input.

---

## 📸 Demo graph

![Bowling Analyzer Demo](biomechanics_graphs.png)

## 🎥 Demo Video

## 🎥 Demo Video
[![Watch Demo]](https://drive.google.com/file/d/1FgkleuZ9AzwVmzmXVdarXZXqdmTd2dgt/view?usp=sharing)

---

## 🚀 Features

- **YOLOv8 Person Tracking** — automatically locks onto the bowler using ByteTrack, no manual ROI selection needed
- **MediaPipe Pose Estimation** — 33-landmark full-body skeleton detection on cropped bowler region for maximum accuracy
- **4-Phase Bowling Detection** — automatically detects and labels:
  - 🟡 Run-Up
  - 🟠 Load / Coil
  - 🔴 Delivery Stride
  - 🟢 Follow-Through
- **Joint Angle Analysis** — real-time knee, arm, and trunk angles with visual arcs
- **Weighted Center of Mass (COM)** — 8-landmark weighted COM with EMA smoothing and trail
- **Arm Angular Velocity** — frame-by-frame angular velocity calculation
- **Phase-Annotated Graphs** — 6 biomechanics plots with phase shading using Savitzky-Golay filtering
- **CSV Export** — full frame-by-frame data export for further analysis
- **Professional HUD** — phase badge, angle panel, frame counter, live timeline bar

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [YOLOv8](https://github.com/ultralytics/ultralytics) | Person detection & tracking |
| [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) | 33-point pose estimation |
| [OpenCV](https://opencv.org/) | Video processing & visualization |
| [SciPy](https://scipy.org/) | Savitzky-Golay signal filtering |
| [Matplotlib](https://matplotlib.org/) | Biomechanics graphs |
| Python 3.11 | Core language |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bowling-biomechanics-analyzer.git
cd bowling-biomechanics-analyzer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install opencv-python mediapipe ultralytics scipy matplotlib numpy
```

---

## ▶️ Usage

```bash
# Place your bowling video as bowling.mp4 in the project folder
python3 bowling_analyzer_yolo.py
```

### Output Files
| File | Description |
|------|-------------|
| `output.mp4` | Annotated video with skeleton, phases, HUD |
| `biomechanics_graphs.png` | 6-panel phase-annotated biomechanics plots |
| `biomechanics_data.csv` | Frame-by-frame data export |

---

## 📊 Biomechanics Metrics

| Metric | Description |
|--------|-------------|
| Knee Angle | Hip → Knee → Ankle |
| Arm Angle | Shoulder → Elbow → Wrist |
| Trunk Angle | Shoulder → Hip → Knee |
| Arm Angular Velocity | Rate of arm angle change (°/s) |
| COM X | Horizontal center of mass movement |
| COM Y | Vertical center of mass movement |

---

## 🧠 System Architecture

```
Video Frame
    │
    ▼
YOLOv8 (ByteTrack)
    │  Detects & tracks bowler
    ▼
Crop Bowler Region
    │
    ▼
MediaPipe Pose
    │  33 landmark detection
    ▼
Angle Calculation + Phase Detection
    │
    ▼
EMA Smoothing + COM Calculation
    │
    ▼
HUD Overlay + Output Video + CSV + Graphs
```

---

## 📁 Project Structure

```
bowling-biomechanics-analyzer/
├── bowling_analyzer_yolo.py    # Main analysis script
├── bowling.mp4                 # Input video (not included)
├── output.mp4                  # Generated annotated video
├── biomechanics_graphs.png     # Generated graphs
├── biomechanics_data.csv       # Generated CSV data
└── README.md
```

---

## 📈 Phase Detection Logic

| Phase | Trigger Condition |
|-------|------------------|
| Run-Up → Load | Trunk angle < 155° AND Knee angle < 160° |
| Load → Delivery | Arm angle > 140° |
| Delivery → Follow-Through | Arm angle < 110° AND phase duration > 0.3s |

---

## 🔧 Configuration

You can tune these parameters in the script:

```python
# YOLO tracking
BowlerTracker(pad=45)           # padding around detected person box

# Smoothing
EMA(alpha=0.25)                 # COM smoothing (lower = smoother)
EMA(alpha=0.15)                 # Box smoothing

# MediaPipe
min_detection_confidence=0.3
min_tracking_confidence=0.3
model_complexity=2              # 0=fast, 1=balanced, 2=accurate
```

---

## 📋 Requirements

```
opencv-python>=4.8.0
mediapipe>=0.10.0
ultralytics>=8.0.0
scipy>=1.11.0
matplotlib>=3.7.0
numpy>=1.24.0
```

---

## 👤 Author

**Azhar Khan**  
B.Tech — IIT Indore  
Project: Cricket Bowling Biomechanics Analysis  

---

## 📄 License

This project is licensed under the MIT License.
