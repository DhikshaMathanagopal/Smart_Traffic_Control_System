# Smart-Traffic-Control-System-using-YOLOv3-and-OpenCV
Real-Time Vehicle & Ambulance Detection with Dynamic Traffic Signal Control

This project implements an AI-powered traffic management system that detects vehicles and ambulances in live or recorded footage using YOLOv3 and automatically adjusts traffic light signals to prioritize emergency response vehicles.

Built with Python, OpenCV, and Ultralytics YOLOv3u, this simulation demonstrates how computer vision can optimize real-world traffic control.

YOLOv3u-based vehicle detection – detects cars, buses, trucks, and ambulances
Ambulance priority logic – turns traffic signal 🟢 green only when detection confidence ≥ 90%
Dynamic signal overlay – red/green lights generated directly with OpenCV (no images required)
Color-based heuristic – detects white vans or trucks as possible ambulances
Automatic frame saving – captures every ambulance detection to /output/

System Architecture
A[Traffic Video] --> B[YOLOv3u Model]
B --> C[Vehicle Detection + Confidence Filtering]
C --> D[Ambulance Logic (≥0.9 Confidence)]
D -->|Ambulance Detected| E[Traffic Signal -> GREEN]
D -->|No Ambulance| F[Traffic Signal -> RED]
E --> G[Overlay Output Video + Save Frames]
F --> G

Install Dependencies
pip install --upgrade ultralytics torch torchvision torchaudio opencv-python numpy

Add Video File

Place your input traffic video inside the data/ folder:

Smart_Traffic_Control/data/traffic.mp4

Running the Project
cd Smart_Traffic_Control
source venv/bin/activate
python src/YOLO_ultra.py

Outputs
screenshots/redsignal.png
screenshots/greensignal.png

Future Improvements:
ntegrate multiple camera feeds for multi-lane control
Deploy real-time version on Jetson Nano / Raspberry Pi
Retrain YOLO with a custom “Ambulance” dataset for higher accuracy
Integrate GPS + IoT sensors for live emergency vehicle routing
