"""
YOLO detection and risk scoring.

This module wraps the general COCO model and the custom barrier model, then
turns their boxes into the obstacle and traffic summaries used by decision.py.
"""

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO


# COCO class IDs that matter for this project.
OBSTACLE_CLASSES = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    58: "potted plant",   # common campus obstacle
}

# Traffic-control classes available in COCO.
TRAFFIC_CLASSES = {
    9:  "traffic_light",
    11: "stop_sign",
}

# Higher number wins when multiple traffic detections disagree.
ACTION_PRIORITY = {
    "STOP":    4,
    "SLOW":    3,
    "CAUTION": 2,
    "GO":      1,
    "CLEAR":   0,
}


class DetectionSmoother:
    """
    Smooth obstacle decisions over a short frame window.

    YOLO boxes can flicker under blur or lighting changes. A short majority
    vote keeps the decision from changing on every single noisy frame.
    """
    def __init__(self, window: int = 5):
        self.history: deque = deque(maxlen=window)

    def update(self, obstacle_in_path: bool, risk_score: float):
        self.history.append((obstacle_in_path, risk_score))

    def get_stable(self):
        if not self.history:
            return False, 0.0
        in_path_votes = sum(1 for flag, _ in self.history if flag)
        avg_risk      = sum(r for _, r in self.history) / len(self.history)
        stable_flag   = in_path_votes >= (len(self.history) / 2)
        return stable_flag, round(avg_risk, 2)


def _bbox_zone_overlap(bbox, zone_mask, frame_width, frame_height):
    """
    Return how much of a bounding box overlaps a binary mask.

    The masks are already built for the frame, so this is just a crop and a
    white-pixel count.
    """
    x1, y1, x2, y2 = bbox
    # Clamp to valid frame coordinates.
    x1 = max(0, min(frame_width  - 1, x1))
    x2 = max(0, min(frame_width,      x2))
    y1 = max(0, min(frame_height - 1, y1))
    y2 = max(0, min(frame_height,     y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    region    = zone_mask[y1:y2, x1:x2]
    inter     = float(np.sum(region > 0))
    bbox_area = float(max((x2 - x1) * (y2 - y1), 1))
    return inter / bbox_area


def estimate_risk(detection, car_zone_mask, danger_zone_mask, frame_width, frame_height):
    """
    Assign a risk score from 0.0 to 1.0 using image-only cues.

    Direct overlap with the car area matters most, followed by overlap with
    the wider safety zone. Box size and vertical position are used as rough
    distance hints because there is no depth sensor.
    """
    x1, y1, x2, y2 = detection["bbox"]

    car_overlap    = _bbox_zone_overlap(detection["bbox"], car_zone_mask,    frame_width, frame_height)
    danger_overlap = _bbox_zone_overlap(detection["bbox"], danger_zone_mask, frame_width, frame_height)
    bottom_y_ratio = y2 / frame_height
    box_area_ratio = ((x2 - x1) * (y2 - y1)) / (frame_width * frame_height)

    bbox_center_x = ((x1 + x2) * 0.5) / max(frame_width, 1)
    lane_center_bonus = 0.12 if abs(bbox_center_x - 0.5) <= 0.22 else 0.0

    risk_score = (
        danger_overlap * 0.40
        + car_overlap  * 0.35
        + bottom_y_ratio * 0.15
        + min(box_area_ratio * 10, 1.0) * 0.10
        + lane_center_bonus
    )

    # Slightly more sensitive in-path trigger to react a bit earlier.
    detection["in_path"] = (danger_overlap > 0.10) or (car_overlap > 0.03)
    if detection["in_path"]:
        risk_score = max(risk_score, 0.35)
    risk_score = round(min(risk_score, 1.0), 2)
    proximity = "NEAR" if risk_score > 0.65 else "MEDIUM" if risk_score > 0.35 else "FAR"

    detection["path_overlap"] = round(danger_overlap, 2)
    detection["car_overlap"]  = round(car_overlap, 2)
    detection["risk_score"]   = risk_score
    detection["proximity"]    = proximity

    return detection


def _detect_school_sign(frame, bbox):
    """
    Check whether a sign crop looks like a yellow school/warning sign.

    This is only a colour heuristic because the COCO model has no separate
    school-sign class.
    """
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    hsv         = cv2.cvtColor(cv2.resize(crop, (80, 80)), cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, np.array([20, 120, 120]), np.array([40, 255, 255]))
    return float(np.sum(yellow_mask > 0) / yellow_mask.size) > 0.30


class YOLODetector:
    """
    Wrap the general COCO model plus the custom boom-barrier model.

    COCO handles common road objects and traffic signals. The second model is
    kept separate because the campus barrier is not detected reliably by COCO.
    """

    def __init__(self, coco_weights: str = "models/yolov8n.pt",
                 barrier_weights: str = "models/boom_barrier_best.pt"):
        coco_path    = Path(coco_weights).expanduser().resolve()
        barrier_path = Path(barrier_weights).expanduser().resolve()

        if not coco_path.exists():
            raise FileNotFoundError(f"COCO weights not found: {coco_path}")
        if not barrier_path.exists():
            raise FileNotFoundError(f"Barrier weights not found: {barrier_path}")

        self.coco_model    = YOLO(str(coco_path))
        self.barrier_model = YOLO(str(barrier_path))
        self.smoother      = DetectionSmoother(window=5)

        # Some trained barrier models are single-class, others keep class names.
        barrier_names = getattr(self.barrier_model, "names", {}) or {}
        self._barrier_single_class = len(barrier_names) == 1

        print(f"[YOLODetector] COCO model:    {coco_path.name}")
        print(f"[YOLODetector] Barrier model: {barrier_path.name}"
              f"  (single-class={self._barrier_single_class})")

    def _extract_barriers(self, frame_small, barrier_results,
                          car_zone_mask, danger_zone_mask):
        """Convert barrier detections into the standard obstacle format."""
        sh, sw     = frame_small.shape[:2]
        detections = []
        names      = (getattr(barrier_results[0], "names", {})
                      or getattr(self.barrier_model, "names", {}))

        for box in barrier_results[0].boxes:
            cls_id     = int(box.cls[0])
            class_name = str(names.get(cls_id, "")).lower()

            # Multi-class barrier models may include non-barrier labels.
            if not self._barrier_single_class and "barrier" not in class_name:
                continue

            conf = float(box.conf[0])
            if conf < 0.4:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if (x2 - x1) * (y2 - y1) / (sw * sh) < 0.001:
                continue    # tiny noise

            det = {
                "class":      "boom_barrier",
                "confidence": round(conf, 2),
                "bbox":       [x1, y1, x2, y2],
            }
            det = estimate_risk(det, car_zone_mask, danger_zone_mask, sw, sh)
            detections.append(det)

        return detections

    def _process_obstacles(self, frame_small, coco_results,
                           car_zone_mask, danger_zone_mask,
                           extra_detections):
        """
        Filter COCO detections to road-relevant classes and score each one.

        Barrier detections are passed in separately and merged here so the
        rest of the pipeline sees one obstacle list.
        """
        sh, sw     = frame_small.shape[:2]
        detections = list(extra_detections)

        for box in coco_results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id not in OBSTACLE_CLASSES:
                continue

            conf = float(box.conf[0])
            if conf < 0.4:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if (x2 - x1) * (y2 - y1) / (sw * sh) < 0.005:
                continue    # tiny or very distant noise

            det = {
                "class":      OBSTACLE_CLASSES[cls_id],
                "confidence": round(conf, 2),
                "bbox":       [x1, y1, x2, y2],
            }
            det = estimate_risk(det, car_zone_mask, danger_zone_mask, sw, sh)
            detections.append(det)

        in_path = [d for d in detections if d["in_path"]]
        closest = max(in_path, key=lambda d: d["risk_score"]) if in_path else None

        # Smooth the in-path decision before choosing an obstacle action.
        self.smoother.update(bool(in_path), closest["risk_score"] if closest else 0.0)
        stable_flag, stable_risk = self.smoother.get_stable()

        if stable_flag and stable_risk > 0.65:
            action = "STOP"
        elif stable_flag and stable_risk > 0.35:
            action = "SLOW"
        else:
            action = "CLEAR"

        return {
            "obstacles":        detections,
            "obstacle_in_path": stable_flag,
            "closest_obstacle": closest,
            "stable_risk":      stable_risk,
            "obstacle_action":  action,
        }

    def _process_traffic(self, frame_small, coco_results):
        """
        Look for traffic-control detections and return the strongest action.

        Stop signs are direct STOPs. Traffic lights are treated conservatively
        because colour-state parsing is disabled.
        """
        detections = []

        for box in coco_results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id not in TRAFFIC_CLASSES:
                continue

            conf            = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label           = TRAFFIC_CLASSES[cls_id]

            det = {
                "class":      label,
                "confidence": round(conf, 2),
                "bbox":       [x1, y1, x2, y2],
                "state":      None,
                "action":     None,
                "detail":     None,
            }

            if label == "traffic_light":
                det["state"]  = "traffic_light"
                det["action"] = "CAUTION"

            elif label == "stop_sign":
                det["state"]  = "stop"
                det["action"] = "STOP"

                if _detect_school_sign(frame_small, [x1, y1, x2, y2]):
                    det["state"]  = "school_ahead"
                    det["action"] = "SLOW"
                    det["detail"] = "School zone"

            detections.append(det)

        # Pick the most urgent traffic action.
        traffic_action = "GO"
        for d in detections:
            if ACTION_PRIORITY.get(d["action"], 0) > ACTION_PRIORITY.get(traffic_action, 0):
                traffic_action = d["action"]

        return {
            "traffic_detections": detections,
            "traffic_action":     traffic_action,
        }

    def process_frame(self, frame_small, car_zone_mask, danger_zone_mask):
        """
        Run both YOLO models on the resized frame and combine their results.

        COCO inference is shared by obstacle and traffic processing, so the
        general model runs only once per frame.
        """
        t0 = time.time()

        # COCO is reused by both obstacle and traffic processors.
        coco_results    = self.coco_model(frame_small,    conf=0.4, verbose=False)
        # The fine-tuned barrier model runs separately.
        barrier_results = self.barrier_model(frame_small, conf=0.4, verbose=False)

        inference_ms = round((time.time() - t0) * 1000, 1)

        # Convert raw detections into the dictionaries used downstream.
        barrier_dets  = self._extract_barriers(
            frame_small, barrier_results, car_zone_mask, danger_zone_mask
        )
        obstacle_data = self._process_obstacles(
            frame_small, coco_results, car_zone_mask, danger_zone_mask, barrier_dets
        )
        traffic_data  = self._process_traffic(frame_small, coco_results)

        return {
            **obstacle_data,
            **traffic_data,
            "coco_results":    coco_results,     # passed to annotator
            "barrier_results": barrier_results,  # passed to annotator
            "barrier_dets":    barrier_dets,     # passed to annotator
            "inference_time_ms": inference_ms,
        }
