import cv2, time
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("D:/NUST/6th sem/DIP/Project/deep_learning/2.mp4")

configs = [
    ("640x360", (640, 360)),
    ("320x180", (320, 180)),
]

for name, (w, h) in configs:
    times = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(100):
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (w, h))
        t0 = time.time()
        model(frame_resized, conf=0.4, verbose=False)
        times.append(time.time() - t0)
    avg_fps = 1 / (sum(times) / len(times))
    print(f"{name}: avg FPS = {avg_fps:.1f}, avg latency = {sum(times)/len(times)*1000:.1f}ms")

cap.release()