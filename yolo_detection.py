"""
yolo_detection.py
─────────────────
Responsibility: run YOLO inference and turn raw bounding boxes into
structured, risk-scored detections.

This file answers two questions:
    1. What objects are in the frame, and how dangerous are they?
    2. Are there any traffic control signals we need to obey?

It does NOT make final driving decisions — that is decision.py's job.

Public API
──────────
    detector = YOLODetector(coco_weights, barrier_weights)

    result = detector.process_frame(frame_small, car_zone_mask, danger_zone_mask)

    result keys
        "obstacles"         – list of obstacle dicts with risk scores
        "obstacle_in_path"  – smoothed bool
        "closest_obstacle"  – highest-risk in-path obstacle (or None)
        "stable_risk"       – smoothed float 0-1
        "obstacle_action"   – "STOP" / "SLOW" / "CLEAR"
        "traffic_detections"– list of traffic control dicts
        "traffic_action"    – "STOP" / "SLOW" / "GO" / "CAUTION"
        "inference_time_ms" – wall-clock ms for both YOLO passes
"""

import re
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

try:
    import pytesseract
    # Windows users: uncomment and set this path
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    _TESSERACT_OK = True
except ImportError:
    pytesseract = None
    _TESSERACT_OK = False

from ultralytics import YOLO


# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────

# COCO class IDs relevant to road driving
OBSTACLE_CLASSES = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    58: "potted plant",   # large flower-pot obstacles on campus
}

# COCO classes for traffic control (lights + signs)
TRAFFIC_CLASSES = {
    9:  "traffic_light",
    11: "stop_sign",
}

# Priority used when combining actions from multiple detections
# Higher number = more urgent
ACTION_PRIORITY = {
    "STOP":    4,
    "SLOW":    3,
    "CAUTION": 2,
    "GO":      1,
    "CLEAR":   0,
}


# ──────────────────────────────────────────────────────────────
# TEMPORAL SMOOTHER
# ──────────────────────────────────────────────────────────────

class DetectionSmoother:
    """
    Reduces frame-to-frame flickering in obstacle decisions.

    Raw YOLO output can vary slightly each frame due to lighting changes,
    motion blur, and NMS randomness.  Instead of reacting instantly, we
    keep a short history and require a majority vote before declaring an
    obstacle "in path".

    For 30 fps video, window=5 → ~167 ms of lag — acceptable for
    low-speed campus driving.
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


# ──────────────────────────────────────────────────────────────
# RISK ESTIMATION
# ──────────────────────────────────────────────────────────────

def _bbox_zone_overlap(bbox, zone_mask, frame_width, frame_height):
    """
    Fraction of a bounding box that overlaps a binary zone mask.

    We crop the pre-built mask to the bbox region and count white pixels.
    This is fast (no polygon math per-call) because the masks are built
    once per frame in main.py.
    """
    x1, y1, x2, y2 = bbox
    # Clamp to valid frame coordinates
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
    Assigns a risk score (0.0 → 1.0) using monocular visual cues.

    We have no depth sensor, so we approximate distance from the image:

    car_overlap   – object overlaps the car's own footprint (immediate collision)
    danger_overlap – object in the expanded safety zone around the car
    bottom_y      – lower in frame = physically closer (perspective geometry)
    box_area      – larger box = closer object (perspective scaling)

    Weights chosen so that spatial overlap dominates over size heuristics.
    """
    x1, y1, x2, y2 = detection["bbox"]

    car_overlap    = _bbox_zone_overlap(detection["bbox"], car_zone_mask,    frame_width, frame_height)
    danger_overlap = _bbox_zone_overlap(detection["bbox"], danger_zone_mask, frame_width, frame_height)
    bottom_y_ratio = y2 / frame_height
    box_area_ratio = ((x2 - x1) * (y2 - y1)) / (frame_width * frame_height)

    risk_score = (
        danger_overlap * 0.50
        + car_overlap  * 0.25
        + bottom_y_ratio * 0.15
        + min(box_area_ratio * 10, 1.0) * 0.10
    )
    risk_score = round(min(risk_score, 1.0), 2)

    proximity = "NEAR" if risk_score > 0.65 else "MEDIUM" if risk_score > 0.35 else "FAR"

    detection["path_overlap"] = round(danger_overlap, 2)
    detection["car_overlap"]  = round(car_overlap, 2)
    detection["risk_score"]   = risk_score
    detection["proximity"]    = proximity
    # An obstacle is "in path" if it meaningfully overlaps the danger or car zone
    detection["in_path"]      = (danger_overlap > 0.15) or (car_overlap > 0.05)

    return detection


# ──────────────────────────────────────────────────────────────
# TRAFFIC LIGHT + SIGN HELPERS
# ──────────────────────────────────────────────────────────────

def _read_traffic_light_state(frame, bbox):
    """
    Determines the lit colour of a traffic light bounding box crop.

    Why HSV instead of BGR:
    BGR mixes colour and brightness so the same red looks different in
    shadow vs sunlight.  HSV separates Hue (colour) from Value (brightness),
    making thresholds robust to outdoor lighting variation.

    Why split into thirds:
    Traffic lights always follow RED-top / YELLOW-middle / GREEN-bottom
    internationally.  Checking each third independently is more reliable
    than asking "what colour is brightest overall?"

    Why two red masks:
    In OpenCV HSV the hue wheel runs 0–179.  Red sits at both ends (0–10
    and 160–179) so we need two ranges merged with bitwise OR.
    """
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "unknown"

    crop = cv2.resize(crop, (30, 90))   # tall narrow — matches light shape
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h    = crop.shape[0]

    top    = hsv[0     : h//3,   :]   # red bulb region
    middle = hsv[h//3  : 2*h//3, :]   # yellow bulb region
    bottom = hsv[2*h//3: h,      :]   # green bulb region

    r1         = cv2.inRange(top, np.array([0,   100, 100]), np.array([10,  255, 255]))
    r2         = cv2.inRange(top, np.array([160, 100, 100]), np.array([179, 255, 255]))
    red_score  = np.sum(cv2.bitwise_or(r1, r2) > 0) / top.size

    ym         = cv2.inRange(middle, np.array([20, 100, 100]), np.array([35, 255, 255]))
    yel_score  = np.sum(ym > 0) / middle.size

    gm         = cv2.inRange(bottom, np.array([40, 50, 50]),  np.array([90, 255, 255]))
    grn_score  = np.sum(gm > 0) / bottom.size

    scores = {"red": red_score, "yellow": yel_score, "green": grn_score}
    best   = max(scores, key=scores.get)

    # Require a minimum score to avoid false readings on dark/blurry crops
    return best if scores[best] >= 0.05 else "unknown"


def _read_speed_limit(frame, bbox):
    """
    Uses OCR (Optical Character Recognition) to read a number from a
    speed-limit sign crop.

    Preprocessing steps:
      Enlarge → grayscale → binary threshold
    Each step makes the text cleaner for Tesseract to parse.

    Returns an integer speed limit, or None if unreadable.
    """
    if not _TESSERACT_OK:
        return None

    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop  = cv2.resize(crop, (120, 120))
    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # psm 7 = single line of text; whitelist = digits only
    cfg  = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(th, config=cfg).strip()
    nums = re.findall(r'\d+', text)
    if nums:
        limit = int(nums[0])
        if 5 <= limit <= 130:
            return limit
    return None


def _detect_school_sign(frame, bbox):
    """
    Checks whether a detected sign region is fluorescent yellow —
    the distinctive colour of school/warning signs.

    YOLO has no "school sign" class, so we use a classical HSV colour
    threshold on the crop.  This is a heuristic, not a classifier, but
    it is fast and interpretable.
    """
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    hsv         = cv2.cvtColor(cv2.resize(crop, (80, 80)), cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, np.array([20, 120, 120]), np.array([40, 255, 255]))
    return float(np.sum(yellow_mask > 0) / yellow_mask.size) > 0.30


# ──────────────────────────────────────────────────────────────
# MAIN DETECTOR CLASS
# ──────────────────────────────────────────────────────────────

class YOLODetector:
    """
    Wraps two YOLO models (general COCO + fine-tuned boom barrier) and
    exposes a single process_frame() call.

    Two separate models because:
    - The COCO model covers people, cars, bikes, flower pots, etc.
    - The barrier model is fine-tuned for our specific campus barrier,
      which the general model does not reliably detect.
    Both run on the same resized frame so total overhead is ~2× single.
    """

    def __init__(self, coco_weights: str = "yolov8n.pt",
                 barrier_weights: str = "boom_barrier_best.pt"):
        coco_path    = Path(coco_weights).expanduser().resolve()
        barrier_path = Path(barrier_weights).expanduser().resolve()

        if not coco_path.exists():
            raise FileNotFoundError(f"COCO weights not found: {coco_path}")
        if not barrier_path.exists():
            raise FileNotFoundError(f"Barrier weights not found: {barrier_path}")

        self.coco_model    = YOLO(str(coco_path))
        self.barrier_model = YOLO(str(barrier_path))
        self.smoother      = DetectionSmoother(window=5)

        # Detect whether the barrier model is single-class (always barrier)
        barrier_names = getattr(self.barrier_model, "names", {}) or {}
        self._barrier_single_class = len(barrier_names) == 1

        print(f"[YOLODetector] COCO model:    {coco_path.name}")
        print(f"[YOLODetector] Barrier model: {barrier_path.name}"
              f"  (single-class={self._barrier_single_class})")

    # ── internal helpers ──────────────────────────────────────

    def _extract_barriers(self, frame_small, barrier_results,
                          car_zone_mask, danger_zone_mask):
        """Converts barrier model detections into the standard obstacle dict format."""
        sh, sw     = frame_small.shape[:2]
        detections = []
        names      = (getattr(barrier_results[0], "names", {})
                      or getattr(self.barrier_model, "names", {}))

        for box in barrier_results[0].boxes:
            cls_id     = int(box.cls[0])
            class_name = str(names.get(cls_id, "")).lower()

            # For multi-class barrier models keep only barrier detections
            if not self._barrier_single_class and "barrier" not in class_name:
                continue

            conf = float(box.conf[0])
            if conf < 0.4:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if (x2 - x1) * (y2 - y1) / (sw * sh) < 0.001:
                continue    # skip tiny noise

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
        Filters COCO detections to road-relevant classes, scores each,
        merges barrier detections, and returns the obstacle summary.
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
                continue    # discard tiny/distant noise

            det = {
                "class":      OBSTACLE_CLASSES[cls_id],
                "confidence": round(conf, 2),
                "bbox":       [x1, y1, x2, y2],
            }
            det = estimate_risk(det, car_zone_mask, danger_zone_mask, sw, sh)
            detections.append(det)

        in_path = [d for d in detections if d["in_path"]]
        closest = max(in_path, key=lambda d: d["risk_score"]) if in_path else None

        # Update smoother and get stable (flicker-free) result
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
        Scans COCO detections for traffic lights and signs.
        Runs sub-analysis (colour read / OCR / school-sign heuristic)
        for each and returns the most urgent traffic action.
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
                state       = _read_traffic_light_state(frame_small, [x1, y1, x2, y2])
                det["state"]  = state
                det["action"] = {"red": "STOP", "yellow": "SLOW",
                                 "green": "GO"}.get(state, "CAUTION")

            elif label == "stop_sign":
                det["state"]  = "stop"
                det["action"] = "STOP"

                if _detect_school_sign(frame_small, [x1, y1, x2, y2]):
                    det["state"]  = "school_ahead"
                    det["action"] = "SLOW"
                    det["detail"] = "School zone"

                limit = _read_speed_limit(frame_small, [x1, y1, x2, y2])
                if limit:
                    det["state"]  = "speed_limit"
                    det["action"] = "SLOW"
                    det["detail"] = f"{limit} km/h"

            detections.append(det)

        # Pick the single most urgent traffic action
        traffic_action = "GO"
        for d in detections:
            if ACTION_PRIORITY.get(d["action"], 0) > ACTION_PRIORITY.get(traffic_action, 0):
                traffic_action = d["action"]

        return {
            "traffic_detections": detections,
            "traffic_action":     traffic_action,
        }

    # ── public API ────────────────────────────────────────────

    def process_frame(self, frame_small, car_zone_mask, danger_zone_mask):
        """
        Runs both YOLO models on frame_small and returns a combined result.

        frame_small     – already-resized BGR frame (e.g. 640×360)
        car_zone_mask   – binary mask matching the car icon footprint
        danger_zone_mask – dilated version of car_zone_mask

        Important: both YOLO calls happen here (not in two places).
        COCO results feed both obstacle and traffic processors,
        so inference runs only once for COCO.
        """
        t0 = time.time()

        # Run COCO model once — reused by both obstacle and traffic processors
        coco_results    = self.coco_model(frame_small,    conf=0.4, verbose=False)
        # Run fine-tuned barrier model separately
        barrier_results = self.barrier_model(frame_small, conf=0.4, verbose=False)

        inference_ms = round((time.time() - t0) * 1000, 1)

        # Extract barrier detections first so they can be merged into obstacles
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
