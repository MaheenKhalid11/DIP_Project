# test_traffic.py
import cv2
from traffic_analyzer import analyze_traffic

cap = cv2.VideoCapture("D:/NUST/6th sem/DIP/Project/deep_learning/0000f77c-6257be58.mov")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = analyze_traffic(frame)

    # Print decision to terminal
    print(f"Action: {result['final_action']}")
    for det in result["detections"]:
        print(f"  → {det['class']} | state: {det['state']} | {det['detail'] or ''}")

    cv2.imshow("Traffic Analysis", result["annotated_frame"])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()