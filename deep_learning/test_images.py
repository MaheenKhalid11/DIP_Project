import cv2
import os
from ultralytics import YOLO

# Load pretrained YOLOv8n (downloads automatically first time)
model = YOLO("yolov8n.pt")

# --- Step 1: Extract a few frames from your video ---
cap = cv2.VideoCapture("D:/NUST/6th sem/DIP/Project/deep_learning/2.mp4")  # change to your video path
os.makedirs("sample_frames", exist_ok=True)

for i in range(10):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i * 30)  # every 30th frame
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f"sample_frames/frame_{i}.jpg", frame)
cap.release()

# --- Step 2: Run YOLO on each saved frame ---
import glob
for img_path in glob.glob("sample_frames/*.jpg"):
    results = model(img_path, conf=0.4)
    annotated = results[0].plot()  # draws boxes on the frame
    cv2.imshow("Detection", annotated)
    cv2.waitKey(0)

cv2.destroyAllWindows()