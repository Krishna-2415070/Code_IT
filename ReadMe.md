

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


## Running the Project from Source Code

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

## Developed By

Code_IT

National Institute of Technology Silchar (NIT Silchar)
