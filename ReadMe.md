

# Traffic Intelligence System 🚦

An AI-powered helmet violation detection and traffic monitoring system built using YOLO, OpenCV, PyTorch, and Tkinter.

## Features

* Helmet Violation Detection
* Image-Based Analysis
* Video-Based Analysis
* Real-Time Webcam Monitoring
* Automatic Violation Evidence Capture
* Analytics Dashboard
* PDF Report Generation
* HTML Report Generation
* CPU and GPU Support

## Technologies Used

* Python
* YOLO (Ultralytics)
* OpenCV
* PyTorch
* Tkinter
* FPDF

## Project Files

* `APP.py` – Main application source code (inside folder named, "MODELS")
* `best.pt` – Custom trained helmet detection model (Inside folder named, "MODELS")
* `yolov8n.pt` – Base YOLO model
* `README.md` – Project documentation


## Running the Project from Source Code (METHOD - 1)

### 1. Clone the Repository

```bash
git clone https://github.com/Krishna-2415070/Code_IT.git
cd Code_IT
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

### 4. Install Required Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python APP.py
```

### Notes

* Ensure `best.pt` and `yolov8n.pt` are present in the project directory.
* The application automatically detects available hardware.
* NVIDIA GPU users will benefit from CUDA acceleration.
* Systems without NVIDIA GPUs will automatically run in CPU mode.



## Hardware Support

The application automatically detects available hardware.

* NVIDIA GPU → CUDA accelerated inference
* CPU → Automatic fallback mode

For the best detection speed and highest FPS during real-time video and webcam processing, it is recommended to run the application on a system equipped with an NVIDIA GPU.


## Standalone Executable Version (Method - 2)

For users who do not want to set up Python, virtual environments, or dependencies, a pre-built Windows executable version of the application is available.

### Download

Traffic Intelligence System (Windows Executable)

https://drive.google.com/file/d/1yLVL0KgYn60xEaEIDXLlegMlzgledl50/view?usp=sharing

### Installation Steps

1. Download the ZIP file from the link above.

2. Extract the ZIP file to any desired location.

3. After extraction, the folder structure should appear as:

```
Traffic-APP/
│
├── README.pdf
├── Test-Files/
│
└── TrafficIntelligence/
    ├── TrafficIntelligence.exe
    └── _internal/
```

4. Open:

```
Traffic-APP → TrafficIntelligence
```

5. Double-click:

```
TrafficIntelligence.exe
```

6. The application will start automatically.

### Important Notes

* Do not delete, move, or modify any files inside the `_internal` folder.
* No Python installation is required.
* No dependency installation is required.
* No virtual environment setup is required.
* The application automatically detects available hardware.
* Systems with NVIDIA GPUs will utilize CUDA acceleration for improved performance.
* Systems without NVIDIA GPUs will automatically run in CPU mode.

### Testing

Sample media files are included in the `Test-Files` folder and can be used to evaluate:

* Image Detection
* Video Processing
* Helmet Violation Detection
* Report Generation



## Web Version (Method - 3)

A browser-based version of the Traffic Intelligence System is also available for quick access and demonstration purposes.

### Web Application Link

https://huggingface.co/spaces/Krishna-2415070/traffic-intelligence-system

### Features Available in the Web Version

* Image-Based Helmet Violation Detection
* Basic Video-Based Analysis
* Browser-Based Access Without Installation
* Easy Demo Access From Any Device
* AI-Powered Detection Through a Web Interface

### Notes

* The web version is designed for accessibility and demonstration.
* Some advanced desktop features may not be fully available in the web version.
* For the complete feature set, best performance, and most reliable experience, the Standalone Executable Version (Method - 2) is highly recommended.


## Developed By

## Team - Code_IT 
-->Krishna Mahato (lead)
-->Akash De
-->Himanshi Agarwal
-->Tinkal Das


National Institute of Technology Silchar (NIT Silchar)
Electronics and Instrumentation, EIE-28

