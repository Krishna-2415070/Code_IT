<div align="center">

<h1>🚦 Traffic Intelligence System</h1>

<p><strong>AI-Powered Helmet Violation Detection & Traffic Monitoring</strong></p>

<p>
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLO-Ultralytics-FF6F00?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-Latest-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
</p>

<p>
  <img src="https://img.shields.io/badge/GPU-CUDA Supported-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/CPU-Auto Fallback-0078D4?style=for-the-badge&logo=intel&logoColor=white" />
  <img src="https://img.shields.io/badge/Platform-Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white" />
</p>

<br/>

> A real-time traffic surveillance system that detects helmet violations using custom-trained YOLO models, with automatic evidence capture, analytics dashboards, and multi-format report generation.

<br/>

</div>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🛠️ Technologies](#️-technologies)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
  - [Method 1 — Run from Source](#method-1--run-from-source)
  - [Method 2 — Standalone Executable](#method-2--standalone-executable)
  - [Method 3 — Web Version](#method-3--web-version)
- [⚙️ Hardware Support](#️-hardware-support)
- [👥 Team](#-team)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🪖 **Helmet Violation Detection** | AI-powered real-time detection of riders without helmets |
| 🖼️ **Image Analysis** | Upload and analyze static images for violations |
| 🎥 **Video Analysis** | Process pre-recorded traffic footage frame by frame |
| 📡 **Live Webcam Monitoring** | Real-time detection through any connected camera |
| 📸 **Evidence Capture** | Automatically saves snapshots of detected violations |
| 📊 **Analytics Dashboard** | Visual summary of detections, statistics, and trends |
| 📄 **PDF Report Generation** | Export detailed violation reports in PDF format |
| 🌐 **HTML Report Generation** | Web-friendly reports for browser-based viewing |
| ⚡ **GPU Acceleration** | CUDA support for high-speed inference on NVIDIA GPUs |

---

## 🛠️ Technologies

<table>
<tr>
  <td><strong>Language</strong></td>
  <td>Python 3.8+</td>
</tr>
<tr>
  <td><strong>Object Detection</strong></td>
  <td>YOLO v8 (Ultralytics)</td>
</tr>
<tr>
  <td><strong>Computer Vision</strong></td>
  <td>OpenCV</td>
</tr>
<tr>
  <td><strong>Deep Learning</strong></td>
  <td>PyTorch</td>
</tr>
<tr>
  <td><strong>GUI Framework</strong></td>
  <td>Tkinter</td>
</tr>
<tr>
  <td><strong>PDF Generation</strong></td>
  <td>FPDF</td>
</tr>
</table>

---

## 📁 Project Structure

```
Code_IT/
│
├── MODELS/
│   ├── APP.py               # Main application source code
│   └── best.pt              # Custom trained helmet detection model
│
├── yolov8n.pt               # Base YOLOv8 nano model
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

> ⚠️ **Important:** Both `best.pt` and `yolov8n.pt` must be present for the application to run correctly.

---

## 🚀 Getting Started

Choose the method that best fits your setup:

---

### Method 1 — Run from Source

> Best for developers who want full control and customization.

**Step 1 — Clone the Repository**
```bash
git clone https://github.com/Krishna-2415070/Code_IT.git
cd Code_IT
```

**Step 2 — Create a Virtual Environment**
```bash
python -m venv .venv
```

**Step 3 — Activate the Environment**

| Platform | Command |
|---|---|
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Windows CMD | `.venv\Scripts\activate.bat` |
| Linux / macOS | `source .venv/bin/activate` |

**Step 4 — Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 5 — Run the Application**
```bash
python APP.py
```

---

### Method 2 — Standalone Executable

> Best for non-technical users. No Python or setup required.

**📥 Download**

[![Download ZIP](https://img.shields.io/badge/Download-Windows%20Executable-0078D6?style=for-the-badge&logo=windows)](https://drive.google.com/file/d/1yLVL0KgYn60xEaEIDXLlegMlzgledl50/view?usp=sharing)

**Installation Steps**

1. Download the ZIP file from the link above
2. Extract it to any folder on your system
3. Verify the extracted folder structure:

```
Traffic-APP/
│
├── README.pdf
├── Test-Files/          ← Sample media for testing
│
└── TrafficIntelligence/
    ├── TrafficIntelligence.exe
    └── _internal/       ← Do NOT modify this folder
```

4. Navigate to `Traffic-APP → TrafficIntelligence`
5. Double-click `TrafficIntelligence.exe` to launch

> 🧪 **Test Files:** Sample images and videos are included in the `Test-Files/` folder for trying out detection, video processing, and report generation.

> 🔒 **Note:** Do not delete, move, or rename any files inside the `_internal/` folder — the app will fail to start.

---

### Method 3 — Web Version

> Best for quick demos and browser-based access. No installation needed.

[![Open Web App](https://img.shields.io/badge/Open-Web%20Application-FFD21E?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/Krishna-2415070/traffic-intelligence-system)

**Available in the Web Version:**
- ✅ Image-based helmet violation detection
- ✅ Basic video analysis
- ✅ AI-powered detection via browser interface
- ✅ Works on any device without installation

> ⚡ For the **complete feature set**, best FPS, and most reliable performance — use **Method 2 (Standalone Executable)**.

---

## ⚙️ Hardware Support

The application **automatically detects your hardware** at startup — no manual configuration needed.

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   NVIDIA GPU detected?                              │
│        YES  →  CUDA-accelerated inference ⚡        │
│        NO   →  Automatic CPU fallback mode 🖥️       │
│                                                     │
└─────────────────────────────────────────────────────┘
```

> 💡 **Recommendation:** For best real-time performance (high FPS in live webcam and video modes), use a system with an **NVIDIA GPU**.

---

## 👥 Team

<div align="center">

**Team Code_IT**
*National Institute of Technology Silchar (NIT Silchar)*
*Department of Electronics and Instrumentation Engineering — EIE-28*

<br/>

| Role | Name |
|---|---|
| 👑 Team Lead | Krishna Mahato |
| 👨‍💻 Member | Akash De |
| 👩‍💻 Member | Himanshi Agarwal |
| 👨‍💻 Member | Tinkal Das |

</div>

---

<div align="center">

**Built with ❤️ at NIT Silchar**

</div>
