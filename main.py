import cv2
import time
import re
import numpy as np
import argparse
import sys
from pathlib import Path
from ultralytics import YOLO
from collections import deque
import pytesseract



from segmentation import color_segmentation


DEFAULT_COCO_MODEL_PATH = "yolov8n.pt"
DEFAULT_BARRIER_MODEL_PATH = "boom_barrier_best.pt"


def load_detection_model(weights_path):
    weights_path = Path(weights_path).expanduser().resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    return YOLO(str(weights_path))


model = load_detection_model(DEFAULT_COCO_MODEL_PATH)
barrier_model = load_detection_model(DEFAULT_BARRIER_MODEL_PATH)
CAR_ICON_PATH = "car.png"
CAR_ICON_IMAGE = cv2.imread(str(CAR_ICON_PATH), cv2.IMREAD_UNCHANGED)


# Classes we care about for obstacle detection
# These are COCO dataset class IDs
OBSTACLE_CLASSES = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    58: "potted plant",   # catches large flower pots on campus
}

# Classes we care about for traffic control
TRAFFIC_CLASSES = {
    9:  "traffic_light",
    11: "stop_sign",
}

# Action priority — higher number = more urgent
# Used when multiple detections give conflicting actions
ACTION_PRIORITY = {
    "STOP":    4,
    "SLOW":    3,
    "CAUTION": 2,
    "GO":      1,
    "CLEAR":   0,
}

class DetectionSmoother:
    """
    Keeps a rolling window of recent results.
    Instead of reacting to every single frame (which causes flickering),
    we take a majority vote over the last N frames.
    For 30fps video, window=5 means ~167ms of smoothing.
    """
    def __init__(self, window=5):
        self.history = deque(maxlen=window)

    def update(self, obstacle_in_path, risk_score):
        self.history.append((obstacle_in_path, risk_score))

    def get_stable(self):
        if not self.history:
            return False, 0.0
        # Majority vote: more than half the recent frames must agree
        in_path_votes = sum(1 for x, _ in self.history if x)
        avg_risk = sum(r for _, r in self.history) / len(self.history)
        stable_in_path = in_path_votes >= (len(self.history) / 2)
        return stable_in_path, round(avg_risk, 2)


# One global smoother instance — persists across frames
smoother = DetectionSmoother(window=5)


# Temporal smoother for road masks to reduce frame-to-frame jitter.
class RoadMaskTemporalSmoother:
    def __init__(self, fast_alpha=0.45, slow_alpha=0.18, threshold=0.50, iou_gate=0.35):
        self.fast_alpha = float(fast_alpha)
        self.slow_alpha = float(slow_alpha)
        self.threshold = float(threshold)
        self.iou_gate = float(iou_gate)
        self.ema_mask = None

    def update(self, mask):
        current = (mask > 0).astype(np.float32)
        if self.ema_mask is None:
            self.ema_mask = current.copy()
        else:
            prev_bin = self.ema_mask >= self.threshold
            curr_bin = current > 0.5
            inter = float(np.logical_and(prev_bin, curr_bin).sum())
            union = float(np.logical_or(prev_bin, curr_bin).sum()) + 1e-6
            iou = inter / union
            alpha = self.fast_alpha if iou >= self.iou_gate else self.slow_alpha
            self.ema_mask = (1.0 - alpha) * self.ema_mask + alpha * current

        stable = (self.ema_mask >= self.threshold).astype(np.uint8) * 255
        stable = cv2.morphologyEx(stable, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        stable = cv2.medianBlur(stable, 5)
        return stable


road_mask_smoother = RoadMaskTemporalSmoother()
path_offset_history = deque(maxlen=7)


def overlay_car_icon(frame, icon_image):
    if icon_image is None:
        return frame

    h, w = frame.shape[:2]
    target_w = max(138, int(0.40 * w))
    target_h = max(84, int(0.28 * h))

    icon = cv2.resize(icon_image, (target_w, target_h), interpolation=cv2.INTER_AREA)

    x1 = (w - target_w) // 2
    y1 = h - target_h - int(0.03 * h)
    x2 = x1 + target_w
    y2 = y1 + target_h

    if icon.shape[2] == 4:
        alpha = (icon[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
        rgb = icon[:, :, :3].astype(np.float32)
        bg = frame[y1:y2, x1:x2].astype(np.float32)
        blended = (alpha * rgb) + ((1.0 - alpha) * bg)
        frame[y1:y2, x1:x2] = blended.astype(np.uint8)
    else:
        frame[y1:y2, x1:x2] = icon[:, :, :3]

    return frame


def get_car_zone_bbox(frame_width, frame_height):
    # Must match overlay_car_icon placement for consistent risk logic.
    target_w = max(138, int(0.40 * frame_width))
    target_h = max(84, int(0.28 * frame_height))
    x1 = (frame_width - target_w) // 2
    y1 = frame_height - target_h - int(0.03 * frame_height)
    x2 = x1 + target_w
    y2 = y1 + target_h
    return [x1, y1, x2, y2]


def get_car_zone_masks(frame_width, frame_height):
    # Build car-shaped occupancy masks (exact icon shape + slight expansion).
    x1, y1, x2, y2 = get_car_zone_bbox(frame_width, frame_height)
    w = x2 - x1
    h = y2 - y1

    car_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    if CAR_ICON_IMAGE is None:
        cv2.rectangle(car_mask, (x1, y1), (x2, y2), 255, -1)
    else:
        icon = cv2.resize(CAR_ICON_IMAGE, (w, h), interpolation=cv2.INTER_AREA)
        if icon.shape[2] == 4:
            local = (icon[:, :, 3] > 20).astype(np.uint8) * 255
        else:
            gray = cv2.cvtColor(icon[:, :, :3], cv2.COLOR_BGR2GRAY)
            local = (gray > 20).astype(np.uint8) * 255
        car_mask[y1:y2, x1:x2] = local

    # Safety zone keeps same shape, expanded for conservative obstacle clearance.
    danger_mask = cv2.dilate(
        car_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        iterations=2,
    )
    return car_mask, danger_mask


def mask_to_polygon(mask, fallback_w, fallback_h):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return get_path_zone(fallback_w, fallback_h, lane_offset=0)
    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)
    epsilon = 0.015 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    return approx.reshape(-1, 2)



def get_path_zone(frame_width, frame_height, lane_offset=0):
    """
    Returns a trapezoid that represents the danger zone directly ahead.

    Why a trapezoid and not a rectangle:
    Due to perspective projection, the road appears narrower in the distance
    and wider up close. The trapezoid mimics this — wide at the bottom
    (close to the car) and narrow at the top (far away).

    lane_offset: shifts the zone left or right if lane detection tells
    us the road is curving. Positive = shift right, negative = shift left.
    """
    w, h = frame_width, frame_height
    cx = w // 2 + lane_offset  # center of the danger zone

    zone = np.array([
        [cx - int(w * 0.15), int(h * 0.5)],   # top-left  (far, narrow)
        [cx + int(w * 0.15), int(h * 0.5)],   # top-right (far, narrow)
        [cx + int(w * 0.35), h],               # bottom-right (close, wide)
        [cx - int(w * 0.35), h],               # bottom-left  (close, wide)
    ], dtype=np.int32)

    return zone




def detect_road_portion(frame_small):
    """
    Estimates drivable road region using adaptive color segmentation,
    then applies temporal smoothing for stable video overlays.
    """
    h, w = frame_small.shape[:2]
    seg = color_segmentation(frame_small, clusters=4, spatial_weight=0.25)
    raw_mask = seg["road_mask"]
    road_mask = road_mask_smoother.update(raw_mask)
    roi_poly = mask_to_polygon(road_mask, w, h)
    road_coverage = round(float(np.sum(road_mask > 0)) / float(h * w), 3)

    return {
        "mask": road_mask,
        "raw_mask": raw_mask,
        "roi_polygon": roi_poly,
        "coverage": road_coverage,
    }



def get_bbox_zone_overlap(bbox, zone_mask, frame_width, frame_height):
    """
    Calculates what fraction of the bounding box overlaps a zone mask.
    Returns a value from 0.0 (no overlap) to 1.0 (completely inside zone).

    We draw both the zone and bbox as masks and measure the intersection.
    """
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(frame_width - 1, x1))
    x2 = max(0, min(frame_width, x2))
    y1 = max(0, min(frame_height - 1, y1))
    y2 = max(0, min(frame_height, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    zone_region = zone_mask[y1:y2, x1:x2]
    inter_area = float(np.sum(zone_region > 0))
    bbox_area = float(max((x2 - x1) * (y2 - y1), 1))
    # Fraction of obstacle bbox covered by zone footprint.
    overlap = inter_area / bbox_area
    return float(overlap)


def estimate_risk(detection, car_zone_mask, danger_zone_mask, frame_width, frame_height):
    """
    Assigns a risk score (0.0 to 1.0) to each detected obstacle.

    Since we have no depth sensor, we approximate distance using
    visual cues from the image — this is called monocular depth estimation:

    - Path overlap: is the object in our lane?
    - Bottom y position: objects lower in frame = physically closer to camera
    - Bounding box size: bigger box = closer object (perspective scaling)

    These three signals are weighted and combined into one risk score.
    """
    x1, y1, x2, y2 = detection["bbox"]

    # Direct overlap with car icon footprint.
    car_overlap = get_bbox_zone_overlap(
        detection["bbox"], car_zone_mask, frame_width, frame_height
    )
    # Overlap with expanded danger area around the car.
    danger_overlap = get_bbox_zone_overlap(
        detection["bbox"], danger_zone_mask, frame_width, frame_height
    )

    # How far down the frame the bottom of the box is (0=top, 1=bottom)
    # Lower = physically closer to the vehicle
    bottom_y_ratio = y2 / frame_height

    # How large the box is relative to the whole frame
    box_area_ratio = ((x2 - x1) * (y2 - y1)) / (frame_width * frame_height)

    # Weighted combination:
    # Path overlap matters most (50%) — irrelevant if not in our lane
    # Position matters second (30%) — how close it is vertically
    # Size matters least (20%) — supporting signal
    risk_score = (
        danger_overlap * 0.50 +
        car_overlap    * 0.25 +
        bottom_y_ratio * 0.15 +
        min(box_area_ratio * 10, 1.0) * 0.10
    )
    risk_score = round(min(risk_score, 1.0), 2)

    # Human-readable proximity label based on risk score
    if risk_score > 0.65:
        proximity = "NEAR"
    elif risk_score > 0.35:
        proximity = "MEDIUM"
    else:
        proximity = "FAR"

    # Add these results directly into the detection dict
    detection["path_overlap"] = round(danger_overlap, 2)
    detection["car_overlap"] = round(car_overlap, 2)
    detection["risk_score"]   = risk_score
    detection["proximity"]    = proximity
    detection["in_path"]      = (danger_overlap > 0.15) or (car_overlap > 0.05)

    return detection


def extract_barrier_detections(frame_small, results, car_zone_mask, danger_zone_mask):
    """
    Converts detections from the dedicated barrier model into the same
    obstacle dict format used by COCO detections.
    """
    sh, sw = frame_small.shape[:2]
    detections = []
    names = getattr(results[0], "names", {}) or getattr(barrier_model, "names", {})
    single_class_barrier_model = len(names) == 1

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = str(names.get(cls_id, "")).lower()

        if not single_class_barrier_model and "barrier" not in class_name:
            continue

        conf = float(box.conf[0])
        if conf < 0.4:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_area = (x2 - x1) * (y2 - y1)
        if box_area / (sw * sh) < 0.001:
            continue

        det = {
            "class": "boom_barrier",
            "confidence": round(conf, 2),
            "bbox": [x1, y1, x2, y2],
        }
        detections.append(estimate_risk(det, car_zone_mask, danger_zone_mask, sw, sh))

    return detections


def process_obstacles(frame_small, results, car_zone_mask, danger_zone_mask, extra_detections=None):
    """
    Filters YOLO results down to obstacles only,
    scores each one for risk, and returns the findings.
    """
    sh, sw = frame_small.shape[:2]
    boxes = results[0].boxes
    detections = list(extra_detections or [])

    for box in boxes:
        cls_id = int(box.cls[0])

        # Skip anything that isn't an obstacle class
        if cls_id not in OBSTACLE_CLASSES:
            continue

        conf = float(box.conf[0])

        # Skip low confidence detections
        if conf < 0.4:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Skip tiny detections — likely noise or far-away irrelevant objects
        box_area = (x2 - x1) * (y2 - y1)
        if box_area / (sw * sh) < 0.005:
            continue

        det = {
            "class":      OBSTACLE_CLASSES[cls_id],
            "confidence": round(conf, 2),
            "bbox":       [x1, y1, x2, y2],
        }

        # Score the risk for this obstacle
        det = estimate_risk(det, car_zone_mask, danger_zone_mask, sw, sh)
        detections.append(det)

    # Find the most dangerous obstacle currently in the path
    in_path = [d for d in detections if d["in_path"]]
    closest = max(in_path, key=lambda d: d["risk_score"]) if in_path else None

    # Update the smoother with this frame's findings
    obstacle_in_path = len(in_path) > 0
    top_risk = closest["risk_score"] if closest else 0.0
    smoother.update(obstacle_in_path, top_risk)

    # Get the smoothed (stable) result across recent frames
    stable_in_path, stable_risk = smoother.get_stable()

    # Decide obstacle action based on smoothed risk
    if stable_in_path and stable_risk > 0.65:
        obstacle_action = "STOP"
    elif stable_in_path and stable_risk > 0.35:
        obstacle_action = "SLOW"
    else:
        obstacle_action = "CLEAR"

    return {
        "obstacles":        detections,
        "obstacle_in_path": stable_in_path,
        "closest_obstacle": closest,
        "stable_risk":      stable_risk,
        "obstacle_action":  obstacle_action,
    }


def get_path_guidance(road_mask, car_zone_bbox):
    h, w = road_mask.shape[:2]
    x1, y1, x2, _ = car_zone_bbox
    car_center_x = 0.5 * (x1 + x2)

    # Multi-band lookahead: near/mid/far centers improve curve handling.
    bands = [
        (max(0, y1 - int(0.16 * h)), max(1, y1 - int(0.02 * h))),  # near
        (max(0, y1 - int(0.30 * h)), max(1, y1 - int(0.16 * h))),  # mid
        (max(0, y1 - int(0.46 * h)), max(1, y1 - int(0.30 * h))),  # far
    ]
    centers = []
    ratios = []
    for top, bottom in bands:
        band = road_mask[top:bottom, :]
        if band.size == 0:
            continue
        ratios.append(float(np.mean(band > 0)))
        road_pixels = np.where(band > 0)
        if road_pixels[1].size >= 20:
            centers.append(float(np.median(road_pixels[1])))

    road_ratio = float(np.mean(ratios)) if ratios else 0.0
    if not centers:
        path_offset_history.append(0.0)
        return {"path_action": "SLOW_DOWN", "path_offset": 0.0, "road_ahead_ratio": round(road_ratio, 3), "curvature_hint": 0.0}

    near_center = centers[0]
    far_center = centers[-1]
    curvature_hint = (far_center - near_center) / max(w, 1)
    desired_center = (0.75 * near_center) + (0.25 * far_center)
    offset_norm = (desired_center - car_center_x) / max(w, 1)
    offset_norm += 0.35 * curvature_hint

    path_offset_history.append(float(offset_norm))
    smooth_offset = float(np.mean(path_offset_history))

    if smooth_offset > 0.05:
        action = "MOVE_RIGHT"
    elif smooth_offset < -0.05:
        action = "MOVE_LEFT"
    else:
        action = "FORWARD"

    if road_ratio < 0.12:
        action = "SLOW_DOWN"

    return {
        "path_action": action,
        "path_offset": round(smooth_offset, 3),
        "road_ahead_ratio": round(road_ratio, 3),
        "curvature_hint": round(float(curvature_hint), 3),
    }


def choose_avoid_action(obstacles, car_zone_bbox, road_mask):
    risky = [o for o in obstacles if o["in_path"] and o["risk_score"] >= 0.35]
    if not risky:
        return "CLEAR", None

    primary = max(risky, key=lambda o: o["risk_score"])
    px1, _, px2, _ = primary["bbox"]
    obs_center = 0.5 * (px1 + px2)
    cx1, _, cx2, _ = car_zone_bbox
    car_center = 0.5 * (cx1 + cx2)

    h, w = road_mask.shape[:2]
    y_top = max(0, int(0.55 * h))
    y_bottom = min(h, int(0.90 * h))
    slice_mask = road_mask[y_top:y_bottom, :]
    left_free = float(np.mean(slice_mask[:, : int(car_center)] > 0)) if int(car_center) > 1 else 0.0
    right_free = float(np.mean(slice_mask[:, int(car_center) :] > 0)) if int(car_center) < w - 1 else 0.0

    if primary["risk_score"] >= 0.75 or primary["car_overlap"] > 0.08:
        return "STOP", primary
    if primary["risk_score"] >= 0.55:
        if obs_center >= car_center and left_free > 0.08:
            return "MOVE_LEFT", primary
        if obs_center < car_center and right_free > 0.08:
            return "MOVE_RIGHT", primary
        return "SLOW_DOWN", primary
    if primary["risk_score"] >= 0.35:
        return "SLOW_DOWN", primary
    return "CLEAR", primary


def decide_driving_action(path_data, avoid_action, traffic_action):
    # Traffic STOP always overrides.
    if traffic_action == "STOP":
        return "STOP"
    if avoid_action == "STOP":
        return "STOP"
    if avoid_action in {"MOVE_LEFT", "MOVE_RIGHT"}:
        return avoid_action
    if traffic_action in {"SLOW", "CAUTION"}:
        return "SLOW_DOWN"
    if avoid_action == "SLOW_DOWN" or path_data["path_action"] == "SLOW_DOWN":
        return "SLOW_DOWN"
    if path_data["path_action"] in {"MOVE_LEFT", "MOVE_RIGHT"}:
        return path_data["path_action"]
    return "FORWARD"




def read_traffic_light_state(frame, bbox):
    """
    Crops the traffic light and checks which bulb is lit
    by analyzing the color in each third of the crop.

    Traffic lights are always: RED top, YELLOW middle, GREEN bottom.
    This physical standard lets us know where to look for each color.

    We use HSV color space because it separates hue (color) from
    brightness, making detection robust to different lighting conditions.
    """
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return "unknown"

    # Tall narrow resize — matches the physical shape of a traffic light
    crop = cv2.resize(crop, (30, 90))
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    h = crop.shape[0]
    top    = hsv[0      : h//3,   :]   # red bulb region
    middle = hsv[h//3   : 2*h//3, :]   # yellow bulb region
    bottom = hsv[2*h//3 : h,      :]   # green bulb region

    # Red wraps around in HSV (hue 0-10 AND 160-179)
    # so we need two separate masks merged with OR
    r1 = cv2.inRange(top, np.array([0,   100, 100]), np.array([10,  255, 255]))
    r2 = cv2.inRange(top, np.array([160, 100, 100]), np.array([179, 255, 255]))
    red_score    = np.sum(cv2.bitwise_or(r1, r2) > 0) / top.size

    yellow_mask  = cv2.inRange(middle, np.array([20, 100, 100]), np.array([35, 255, 255]))
    yellow_score = np.sum(yellow_mask > 0) / middle.size

    green_mask   = cv2.inRange(bottom, np.array([40, 50, 50]),  np.array([90, 255, 255]))
    green_score  = np.sum(green_mask > 0) / bottom.size

    scores    = {"red": red_score, "yellow": yellow_score, "green": green_score}
    best      = max(scores, key=scores.get)

    # Minimum score threshold — avoids false readings on dark/blurry crops
    return best if scores[best] >= 0.05 else "unknown"




def read_speed_limit(frame, bbox):
    """
    Crops the sign region and uses OCR to read the number.
    OCR = Optical Character Recognition — reads text from images.

    We preprocess the crop (grayscale + threshold) to give OCR
    clean high-contrast input, which improves accuracy significantly.
    """
    if pytesseract is None:
        return None

    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    # Enlarge — OCR accuracy improves greatly on larger images
    crop = cv2.resize(crop, (120, 120))

    # Grayscale — OCR doesn't need color, just contrast
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Binary threshold — makes text sharply black on white
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # psm 7 = treat image as a single line of text
    # whitelist = only look for digits, ignore everything else
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text   = pytesseract.image_to_string(thresh, config=config).strip()

    numbers = re.findall(r'\d+', text)
    if numbers:
        limit = int(numbers[0])
        # Sanity check — valid speed limits are between 5 and 130 km/h
        if 5 <= limit <= 130:
            return limit

    return None




def detect_school_sign(frame, bbox):
    """
    School ahead signs are typically bright yellow/fluorescent.
    Since YOLO doesn't have a school sign class, we use a
    classical color-based heuristic on the cropped region.

    This is a good example of where classical methods fill the
    gap where deep learning has no coverage.
    """
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return False

    crop_resized = cv2.resize(crop, (80, 80))
    hsv          = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)

    # Fluorescent warning yellow — distinctive to school/warning signs
    yellow_mask  = cv2.inRange(hsv,
                               np.array([20, 120, 120]),
                               np.array([40, 255, 255]))

    yellow_ratio = np.sum(yellow_mask > 0) / yellow_mask.size

    # If more than 30% of the region is warning yellow, treat as school sign
    return yellow_ratio > 0.30




def process_traffic(frame_small, results):
    """
    Goes through YOLO results looking for traffic lights and stop signs.
    For each one found, runs the appropriate sub-analysis
    (color reading, OCR, or school sign check).
    """
    boxes      = results[0].boxes
    detections = []

    for box in boxes:
        cls_id = int(box.cls[0])

        if cls_id not in TRAFFIC_CLASSES:
            continue

        conf         = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label        = TRAFFIC_CLASSES[cls_id]

        det = {
            "class":      label,
            "confidence": round(conf, 2),
            "bbox":       [x1, y1, x2, y2],
            "state":      None,
            "action":     None,
            "detail":     None,
        }

        if label == "traffic_light":
            state       = read_traffic_light_state(frame_small, [x1, y1, x2, y2])
            det["state"]  = state
            det["action"] = {
                "red":     "STOP",
                "yellow":  "SLOW",
                "green":   "GO",
                "unknown": "CAUTION",
            }.get(state, "CAUTION")

        elif label == "stop_sign":
            # Default assumption: it's a stop sign
            det["state"]  = "stop"
            det["action"] = "STOP"

            # Check if it might actually be a school sign
            # (both can appear as rectangular/octagonal regions)
            if detect_school_sign(frame_small, [x1, y1, x2, y2]):
                det["state"]  = "school_ahead"
                det["action"] = "SLOW"
                det["detail"] = "School zone"

            # Also try OCR in case it's a speed limit sign
            # that YOLO misclassified as a stop sign
            limit = read_speed_limit(frame_small, [x1, y1, x2, y2])
            if limit:
                det["state"]  = "speed_limit"
                det["action"] = "SLOW"
                det["detail"] = f"{limit} km/h"

        detections.append(det)

    # Find the most urgent traffic action across all detections
    traffic_action = "GO"
    for det in detections:
        if ACTION_PRIORITY.get(det["action"], 0) > ACTION_PRIORITY.get(traffic_action, 0):
            traffic_action = det["action"]

    return {
        "traffic_detections": detections,
        "traffic_action":     traffic_action,
    }


def annotate_frame(
    frame_small,
    results,
    barrier_results,
    zone,
    obstacle_data,
    traffic_data,
    final_action,
    road_data,
    car_zone_bbox,
    danger_zone_mask,
    path_data,
):
    """
    Draws all visual output onto the frame:
    - YOLO bounding boxes
    - Danger zone trapezoid
    - Risk labels on obstacles
    - Traffic state labels
    - Final action banner at the top
    """
    # Start with YOLO's own box drawing as the base
    annotated = results[0].plot()

    # Draw boom barrier detections from the dedicated barrier model.
    for barrier in extract_barrier_detections(
        frame_small,
        barrier_results,
        np.zeros(frame_small.shape[:2], dtype=np.uint8),
        np.zeros(frame_small.shape[:2], dtype=np.uint8),
    ):
        x1, y1, x2, y2 = barrier["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 80, 255), 2)
        cv2.putText(
            annotated,
            f"boom_barrier {barrier['confidence']:.2f}",
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 80, 255),
            2,
        )

    # Draw detected road portion as a semi-transparent green overlay.
    road_overlay = np.zeros_like(annotated)
    road_overlay[:, :, 1] = road_data["mask"]
    annotated = cv2.addWeighted(annotated, 1.0, road_overlay, 0.30, 0)
    cv2.polylines(annotated, [road_data["roi_polygon"]], isClosed=True,
                  color=(120, 255, 120), thickness=1)
    cv2.putText(annotated, f"Road coverage: {road_data['coverage']:.2f}",
                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 120), 2)

    # Replace visible danger zone trapezoid with a car icon.
    annotated = overlay_car_icon(annotated, CAR_ICON_IMAGE)
    contours, _ = cv2.findContours(danger_zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(annotated, [c], -1, (0, 180, 255), 1)
        x, y, _, _ = cv2.boundingRect(c)
        cv2.putText(
            annotated,
            "safety zone",
            (x, max(12, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 180, 255),
            1,
        )
    cv2.putText(
        annotated,
        f"path: {path_data['path_action']} ({path_data['path_offset']:+.2f})",
        (10, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (180, 255, 180),
        2,
    )

    # Label each obstacle with its risk info
    for obs in obstacle_data["obstacles"]:
        x1, y1 = obs["bbox"][0], obs["bbox"][1]
        label  = f"{obs['class']} | {obs['proximity']} | risk:{obs['risk_score']}"
        color  = (0, 0, 255) if obs["in_path"] else (0, 200, 200)
        cv2.putText(annotated, label, (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Label each traffic detection with its state and action
    for tdet in traffic_data["traffic_detections"]:
        x1, y1     = tdet["bbox"][0], tdet["bbox"][1]
        label_text = f"{tdet['state']} → {tdet['action']}"
        if tdet["detail"]:
            label_text += f" ({tdet['detail']})"
        text_color = {
            "STOP":    (0, 0, 255),
            "SLOW":    (0, 165, 255),
            "GO":      (0, 255, 0),
            "CAUTION": (0, 255, 255),
        }.get(tdet["action"], (255, 255, 255))
        cv2.putText(annotated, label_text, (x1, max(y1 - 25, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    # Final action banner at the top of the frame
    banner_color = {
        "STOP": (0, 0, 255),
        "SLOW": (0, 165, 255),
        "GO":   (0, 255, 0),
        "FORWARD": (0, 255, 0),
        "MOVE_LEFT": (255, 220, 0),
        "MOVE_RIGHT": (255, 220, 0),
        "SLOW_DOWN": (0, 165, 255),
    }.get(final_action, (255, 255, 255))

    cv2.rectangle(annotated, (0, 0), (300, 50), (0, 0, 0), -1)  # black background
    cv2.putText(annotated, f"ACTION: {final_action}", (8, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, banner_color, 2)

    return annotated



def process_frame_dl(frame, lane_offset=0):
    """
    Master function. Takes one BGR frame, runs the full
    deep learning pipeline, returns everything as a clean dict.

    lane_offset: optional — pass in the lane curve offset from
    the classical pipeline to shift the danger zone accordingly.

    Returns a dictionary your integration team can use directly.
    """
    # Resize for faster inference — 640x360 is our speed/accuracy sweet spot
    frame_small = cv2.resize(frame, (640, 360))
    sh, sw      = frame_small.shape[:2]

    # Time the inference
    t0 = time.time()

    # Run COCO YOLO for normal road objects/signs and the fine-tuned model for barriers.
    results = model(frame_small, conf=0.4, verbose=False)
    barrier_results = barrier_model(frame_small, conf=0.4, verbose=False)

    inference_ms = round((time.time() - t0) * 1000, 1)

    # Build the danger zone for this frame
    zone = get_path_zone(sw, sh, lane_offset=lane_offset)
    car_zone_bbox = get_car_zone_bbox(sw, sh)
    car_zone_mask, danger_zone_mask = get_car_zone_masks(sw, sh)
    road_data = detect_road_portion(frame_small)
    path_data = get_path_guidance(road_data["mask"], car_zone_bbox)

    barrier_detections = extract_barrier_detections(
        frame_small, barrier_results, car_zone_mask, danger_zone_mask
    )
    obstacle_data = process_obstacles(
        frame_small,
        results,
        car_zone_mask,
        danger_zone_mask,
        extra_detections=barrier_detections,
    )
    traffic_data  = process_traffic(frame_small, results)

    # ── Combine into one final action ──
    obs_action = obstacle_data["obstacle_action"]
    traffic_action = traffic_data["traffic_action"]
    avoid_action, primary_risky = choose_avoid_action(
        obstacle_data["obstacles"], car_zone_bbox, road_data["mask"]
    )
    final_action = decide_driving_action(path_data, avoid_action, traffic_action)

    # Draw everything on the frame
    annotated = annotate_frame(
        frame_small, results, barrier_results, zone,
        obstacle_data, traffic_data, final_action, road_data,
        car_zone_bbox, danger_zone_mask, path_data
    )

    return {
        # Obstacle outputs
        "obstacles":          obstacle_data["obstacles"],
        "obstacle_in_path":   obstacle_data["obstacle_in_path"],
        "closest_obstacle":   obstacle_data["closest_obstacle"],
        "obstacle_risk":      obstacle_data["stable_risk"],
        "obstacle_action":    obs_action,
        "avoid_action":       avoid_action,
        "path_action":        path_data["path_action"],
        "path_offset":        path_data["path_offset"],
        "primary_risky_object": primary_risky,

        # Traffic outputs
        "traffic_detections": traffic_data["traffic_detections"],
        "traffic_action":     traffic_action,

        # Master decision
        "final_action":       final_action,

        # Road outputs
        "road_coverage":      road_data["coverage"],

        # Display + timing
        "annotated_frame":    annotated,
        "inference_time_ms":  inference_ms,
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run detector with adaptive stable road overlay.")
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "segmentation" / "detector2_overlay.mp4",
        help="Output annotated video path",
    )
    parser.add_argument(
        "--coco-weights",
        type=Path,
        default=DEFAULT_COCO_MODEL_PATH,
        help="YOLO weights for normal COCO classes.",
    )
    parser.add_argument(
        "--barrier-weights",
        "--weights",
        dest="barrier_weights",
        type=Path,
        default=DEFAULT_BARRIER_MODEL_PATH,
        help="Fine-tuned YOLO weights for boom barrier detection.",
    )
    parser.add_argument("--no-display", action="store_true", help="Disable live preview window")
    args = parser.parse_args()

    model = load_detection_model(args.coco_weights)
    barrier_model = load_detection_model(args.barrier_weights)
    print(f"Loaded COCO YOLO weights: {Path(args.coco_weights).expanduser().resolve()}")
    print(f"Loaded barrier YOLO weights: {Path(args.barrier_weights).expanduser().resolve()}")

    cap = cv2.VideoCapture(str(args.video))

    if not cap.isOpened():
        print(f"Error: could not open video at '{args.video}'")
        print("Make sure the path is correct and the file exists.")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1e-3:
            fps = 25.0
        out_size = (640, 360)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(args.output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            out_size,
        )
        print("Running — press Q to quit\n")

        while True:
            ret, frame = cap.read()

            # ret is False when the video ends
            if not ret:
                print("\nVideo finished.")
                break

            result = process_frame_dl(frame)

            # Print status to terminal on one line (overwrites itself)
            print(
                f"ACTION: {result['final_action']:<6} | "
                f"Obstacle: {result['obstacle_action']:<5} "
                f"(risk {result['obstacle_risk']}) | "
                f"Traffic: {result['traffic_action']:<8} | "
                f"FPS: {1000/result['inference_time_ms']:.1f}",
                end="\r"
            )

            writer.write(result["annotated_frame"])
            if not args.no_display:
                cv2.imshow("DL Module — Full Pipeline", result["annotated_frame"])

            # Press Q to quit early
            if (not args.no_display) and (cv2.waitKey(1) & 0xFF == ord('q')):
                print("\nStopped by user.")
                break

        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"\nSaved annotated video: {args.output}")
