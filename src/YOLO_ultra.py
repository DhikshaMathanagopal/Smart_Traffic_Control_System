import cv2 as cv
import numpy as np
from ultralytics import YOLO
import torch
import os

# ✅ Allow YOLO model + Sequential + Conv for PyTorch 2.6+
torch.serialization.add_safe_globals([
    __import__("ultralytics").nn.tasks.DetectionModel,
    torch.nn.modules.container.Sequential,
    __import__("ultralytics").nn.modules.conv.Conv,
    torch.nn.modules.activation.ReLU
])

print("🚦 Starting YOLO simulation...")

# ✅ Load YOLOv3u model (modern version)
try:
    model = YOLO("yolov3u.pt")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading YOLO model:", e)
    exit()

# ✅ Create all traffic signals using OpenCV (no image files needed)
def make_signal(color):
    signal = np.zeros((400, 300, 3), dtype=np.uint8)
    if color == "red":
        cv.circle(signal, (150, 100), 80, (0, 0, 255), -1)
    elif color == "yellow":
        cv.circle(signal, (150, 200), 80, (0, 255, 255), -1)
    elif color == "green":
        cv.circle(signal, (150, 300), 80, (0, 255, 0), -1)
    return signal

red_signal = make_signal("red")
yellow_signal = make_signal("yellow")
green_signal = make_signal("green")

# ✅ Create output folder
os.makedirs("output", exist_ok=True)

# ✅ Set video path
video_path = "data/traffic.mp4"

# ✅ Open video
cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Error: Could not open video at {video_path}")
    exit()
else:
    print("🎥 Video loaded successfully!")

ambulance_detected = False
frame_count = 0

# ✅ Resize window for visibility
cv.namedWindow("Traffic Camera", cv.WINDOW_NORMAL)
cv.resizeWindow("Traffic Camera", 960, 540)

print("🚘 Running Smart Traffic Control simulation...")
print("👉 Press 'q' or 'ESC' anytime to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⏹️ Video ended or cannot read frame.")
        break

    frame_count += 1
    ambulance_detected = False
    highest_conf = 0.0

    # YOLOv3 inference
    results = model(frame, verbose=False)
    detections = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else []

    for det in detections:
        x1, y1, x2, y2, conf, cls = det[:6]
        label = model.names[int(cls)]
        conf = float(conf)

        # --- Smarter heuristic: white vehicles may be ambulances ---
        region = frame[int(y1):int(y2), int(x1):int(x2)]
        region_hsv = cv.cvtColor(region, cv.COLOR_BGR2HSV)
        mask_white = cv.inRange(region_hsv, (0, 0, 160), (180, 60, 255))
        white_ratio = np.sum(mask_white > 0) / mask_white.size

        if label.lower() in ["truck", "bus", "van"] and white_ratio > 0.35:
            label = "ambulance"

        # Draw detection box and label
        cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 🚑 Detect ambulance only if confidence ≥ 0.90
        if "ambulance" in label.lower() and conf >= 0.90:
            ambulance_detected = True
            highest_conf = max(highest_conf, conf)
            cv.putText(frame, "🚑 Ambulance Detected - Signal Turning Green",
                       (30, 50), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            # Save snapshot
            filename = f"output/ambulance_detected_{frame_count}.jpg"
            cv.imwrite(filename, frame)

    # ✅ Choose signal: Green only for ≥90% confidence ambulance
    signal_img = green_signal if ambulance_detected else red_signal

    # ✅ Overlay larger signal in top-left corner
    small_signal = cv.resize(signal_img, (200, 280))
    frame[20:300, 20:220] = small_signal

    # ✅ Label below signal
    if ambulance_detected:
        cv.putText(frame, f"Signal: GREEN (Conf {highest_conf:.2f})", (25, 320),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv.putText(frame, "Signal: RED", (25, 320),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ✅ Display camera feed
    cv.imshow("Traffic Camera", frame)

    # Exit on 'q' or ESC
    key = cv.waitKey(1)
    if key == ord('q') or key == 27:
        print("🛑 Simulation manually stopped.")
        break

cap.release()
cv.destroyAllWindows()
print("✅ Simulation ended successfully.")
cv.waitKey(0)
