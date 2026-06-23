

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

## Running the Project

1. Clone the repository


git clone https://github.com/Krishna-2415070/Code_IT.git


2. Install dependencies


pip install -r requirements.txt


3. Run the application


python APP.py


## Hardware Support

The application automatically detects available hardware.

* NVIDIA GPU → CUDA accelerated inference
* CPU → Automatic fallback mode

For the best detection speed and highest FPS during real-time video and webcam processing, it is recommended to run the application on a system equipped with an NVIDIA GPU.

## Developed By

Code_IT

National Institute of Technology Silchar (NIT Silchar)
