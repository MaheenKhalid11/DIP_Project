import cv2
import time
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("D:/NUST/6th sem/DIP/Project/deep_learning/2.mp4")

fps_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to 640x360 for speed
    frame_small = cv2.resize(frame, (640, 360))

    t0 = time.time()
    results = model(frame_small, conf=0.4, verbose=False)
    inference_time = (time.time() - t0) * 1000  # ms

    annotated = results[0].plot()
    fps = 1000 / inference_time
    fps_list.append(fps)

    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("YOLO Test", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Average FPS: {sum(fps_list)/len(fps_list):.1f}")