import cv2
import time
import numpy as np
from ultralytics import YOLO
from utils import filter_detections, DetectionSmoother, RELEVANT_CLASSES
from risk_estimator import get_path_zone, estimate_risk

model = YOLO("yolov8n.pt")
smoother = DetectionSmoother(window=5)

def process_frame_dl(frame):
    """
    Main entry point for the deep learning module.
    Takes a BGR frame (numpy array), returns a result dict.
    """
    h, w = frame.shape[:2]
    frame_small = cv2.resize(frame, (640, 360))
    sh, sw = frame_small.shape[:2]

    t0 = time.time()
    results = model(frame_small, conf=0.4, verbose=False)
    inference_ms = round((time.time() - t0) * 1000, 1)

    # Filter to relevant classes
    detections = filter_detections(results, frame_width=sw, frame_height=sh)

    # Define path zone
    zone = get_path_zone(sw, sh)

    # Estimate risk for each detection
    for det in detections:
        estimate_risk(det, zone, sw, sh)

    # Find the most dangerous obstacle
    in_path_objects = [d for d in detections if d["in_path"]]

    if in_path_objects:
        closest = max(in_path_objects, key=lambda d: d["risk_score"])
    else:
        closest = None

    # Smooth results
    obstacle_in_path = len(in_path_objects) > 0
    top_risk = closest["risk_score"] if closest else 0.0
    smoother.update(obstacle_in_path, top_risk)
    stable_in_path, stable_risk = smoother.get_stable()

    # Decision
    if stable_in_path and stable_risk > 0.65:
        action = "STOP"
    elif stable_in_path and stable_risk > 0.35:
        action = "SLOW"
    else:
        action = "CLEAR"

    # Annotate frame
    annotated = results[0].plot()
    cv2.polylines(annotated, [zone], isClosed=True, color=(0, 255, 255), thickness=2)
    cv2.putText(annotated, f"Action: {action}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if action == "STOP" else (0, 165, 255) if action == "SLOW" else (0, 255, 0), 2)
    cv2.putText(annotated, f"FPS: {1000/inference_ms:.1f}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return {
        "objects": detections,
        "obstacle_in_path": stable_in_path,
        "closest_obstacle_class": closest["class"] if closest else None,
        "closest_obstacle_risk": closest["risk_score"] if closest else 0.0,
        "closest_obstacle_bbox": closest["bbox"] if closest else None,
        "recommended_action": action,
        "annotated_frame": annotated,
        "inference_time_ms": inference_ms,
    }


# Run as standalone test
if __name__ == "__main__":
    cap = cv2.VideoCapture("D:/NUST/6th sem/DIP/Project/deep_learning/o1.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = process_frame_dl(frame)
        print(f"Action: {result['recommended_action']} | "
              f"In-path: {result['obstacle_in_path']} | "
              f"Risk: {result['closest_obstacle_risk']} | "
              f"FPS: {1000/result['inference_time_ms']:.1f}")
        cv2.imshow("DL Module Output", result["annotated_frame"])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()