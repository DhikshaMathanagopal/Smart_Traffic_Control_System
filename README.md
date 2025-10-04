
This project simulates an **AI-powered traffic control system** that detects vehicles and ambulances using **YOLOv3**.  
When an ambulance is detected (≥ 90 % confidence), the system turns the signal 🟢 green to prioritize emergency vehicles.  

Built with **Python, OpenCV, and Ultralytics YOLOv3u**, this project demonstrates how computer vision can optimize real-world traffic management.  

<<<<<<< HEAD
System Architecture
A[Traffic Video] --> B[YOLOv3u Model]
B --> C[Vehicle Detection + Confidence FilterinG
C --> D[Ambulance Logic (≥0.9 Confidence)]
D -->|Ambulance Detected| E[Traffic Signal -> GREEN
E --> G[Overlay Output Video + Save Frames]
F --> G
=======
## Features
>>>>>>> edf4ac9 (Updated professional README)

Real-time object detection using YOLOv3  
Detects cars, trucks, buses, and ambulances  
Signal turns **green** only when an ambulance is confidently detected  
Automatically overlays traffic light and detection boxes  
Saves ambulance detection frames in `/output/`  
Simple, clean code — no extra image files required  

## Installation & Setup

### Clone this repository
```bash
git clone https://github.com/DhikshaMathanagopal/Smart_Traffic_Control_System.git
cd Smart_Traffic_Control_System
Create and activate a virtual environment

python3 -m venv venv
source venv/bin/activate    
Install dependencies
bash
Copy code
pip install -r requirements.txt
Add your traffic video
Place your input video inside the data/ folder:


Smart_Traffic_Control_System/data/traffic.mp4
Run the Simulation
python src/YOLO_ultra.py


Model: YOLOv3u (Ultralytics PyTorch version)

Frameworks: PyTorch, OpenCV, NumPy

Threshold: Ambulance detection ≥ 0.90 confidence
