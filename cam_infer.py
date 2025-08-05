import cv2
import os
from ultralytics import YOLO

# Load YOLO model with error handling
model_path = 'dataset/runs/detect/toy_model_v2_trained/weights/best.pt'

if not os.path.exists(model_path):
    print(f"❌ Model not found at: {model_path}")
    print("Please run train_yolo.py first to train the model.")
    exit()

model = YOLO(model_path)
print(f"✅ Model loaded from: {model_path}")

# Try primary camera first, then secondary
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Primary camera (index 0) not available, trying index 1...")
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ Failed to open any webcam")
    exit()

print("✅ Webcam opened successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run inference
    results = model(frame, verbose=False)

    # Assume no toys detected initially
    toy_detected = False
    detection_count = 0

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = box.conf[0].item()
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            # Only show detections above confidence threshold
            if cls_id == 0 and conf > 0.5:  # Class 0 = toy, confidence > 50%
                toy_detected = True
                detection_count += 1
                label = f"Toy {conf:.2f}"
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Status display
    if not toy_detected:
        cv2.putText(frame, "No Toy Detected", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f"Toys Detected: {detection_count}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("Toy Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("✅ Camera released and windows closed.")