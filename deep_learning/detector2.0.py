import cv2
import time
import re
import numpy as np
import pytesseract
from ultralytics import YOLO
from collections import deque

# Windows users — uncomment and fix this path:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ─────────────────────────────────────────────────────────────
# MODEL — loaded once at startup, not inside any function
# Loading inside a function would reload it every frame = very slow
# ─────────────────────────────────────────────────────────────

model = YOLO("yolov8n.pt")


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# SMOOTHING — reduces flickering in detections across frames
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# DANGER ZONE — the trapezoid representing the road ahead
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# OBSTACLE DETECTION + RISK SCORING
# ─────────────────────────────────────────────────────────────

def get_bbox_zone_overlap(bbox, zone, frame_width, frame_height):
    """
    Calculates what fraction of the bounding box is inside the danger zone.
    Returns a value from 0.0 (no overlap) to 1.0 (completely inside zone).

    We draw both the zone and bbox as masks and measure the intersection.
    """
    x1, y1, x2, y2 = bbox

    # Create a blank black image the size of the frame
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

    # Fill the danger zone trapezoid with white on the mask
    cv2.fillPoly(mask, [zone], 255)

    # Crop the mask to just the bounding box area
    bbox_region = mask[y1:y2, x1:x2]

    if bbox_region.size == 0:
        return 0.0

    # Fraction of the bbox area that falls inside the zone
    overlap = np.sum(bbox_region > 0) / bbox_region.size
    return float(overlap)


def estimate_risk(detection, zone, frame_width, frame_height):
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

    # How much of the object overlaps our danger zone (0 to 1)
    path_overlap = get_bbox_zone_overlap(
        detection["bbox"], zone, frame_width, frame_height
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
        path_overlap   * 0.5 +
        bottom_y_ratio * 0.3 +
        min(box_area_ratio * 10, 1.0) * 0.2
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
    detection["path_overlap"] = round(path_overlap, 2)
    detection["risk_score"]   = risk_score
    detection["proximity"]    = proximity
    detection["in_path"]      = path_overlap > 0.25

    return detection


def process_obstacles(frame_small, results, zone):
    """
    Filters YOLO results down to obstacles only,
    scores each one for risk, and returns the findings.
    """
    sh, sw = frame_small.shape[:2]
    boxes = results[0].boxes
    detections = []

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
        det = estimate_risk(det, zone, sw, sh)
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


# ─────────────────────────────────────────────────────────────
# TRAFFIC LIGHT COLOR READER
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# SPEED LIMIT SIGN READER
# ─────────────────────────────────────────────────────────────

def read_speed_limit(frame, bbox):
    """
    Crops the sign region and uses OCR to read the number.
    OCR = Optical Character Recognition — reads text from images.

    We preprocess the crop (grayscale + threshold) to give OCR
    clean high-contrast input, which improves accuracy significantly.
    """
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


# ─────────────────────────────────────────────────────────────
# SCHOOL SIGN DETECTOR
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# TRAFFIC SIGN + LIGHT PROCESSOR
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# ANNOTATION — draws everything on the frame for display
# ─────────────────────────────────────────────────────────────

def annotate_frame(frame_small, results, zone,
                   obstacle_data, traffic_data, final_action):
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

    # Draw the danger zone trapezoid in cyan
    cv2.polylines(annotated, [zone], isClosed=True,
                  color=(255, 255, 0), thickness=2)
    cv2.putText(annotated, "danger zone", (zone[0][0], zone[0][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

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
    }.get(final_action, (255, 255, 255))

    cv2.rectangle(annotated, (0, 0), (300, 50), (0, 0, 0), -1)  # black background
    cv2.putText(annotated, f"ACTION: {final_action}", (8, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, banner_color, 2)

    return annotated


# ─────────────────────────────────────────────────────────────
# MAIN FUNCTION — this is what your teammates call
# ─────────────────────────────────────────────────────────────

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

    # Run YOLO once — results are used by both obstacle and traffic processors
    # Running it once instead of twice saves significant time
    results = model(frame_small, conf=0.4, verbose=False)

    inference_ms = round((time.time() - t0) * 1000, 1)

    # Build the danger zone for this frame
    zone = get_path_zone(sw, sh, lane_offset=lane_offset)

    # Process obstacles and traffic from the same YOLO results
    obstacle_data = process_obstacles(frame_small, results, zone)
    traffic_data  = process_traffic(frame_small, results)

    # ── Combine into one final action ──
    # Both pipelines produce an action — we take the most urgent one
    # STOP from either source overrides everything
    obs_action     = obstacle_data["obstacle_action"]
    traffic_action = traffic_data["traffic_action"]

    if ACTION_PRIORITY.get(obs_action, 0) >= ACTION_PRIORITY.get(traffic_action, 0):
        final_action = obs_action
    else:
        final_action = traffic_action

    # If obstacle says CLEAR but traffic says GO, output GO
    if final_action == "CLEAR":
        final_action = "GO"

    # Draw everything on the frame
    annotated = annotate_frame(
        frame_small, results, zone,
        obstacle_data, traffic_data, final_action
    )

    return {
        # Obstacle outputs
        "obstacles":          obstacle_data["obstacles"],
        "obstacle_in_path":   obstacle_data["obstacle_in_path"],
        "closest_obstacle":   obstacle_data["closest_obstacle"],
        "obstacle_risk":      obstacle_data["stable_risk"],
        "obstacle_action":    obs_action,

        # Traffic outputs
        "traffic_detections": traffic_data["traffic_detections"],
        "traffic_action":     traffic_action,

        # Master decision
        "final_action":       final_action,

        # Display + timing
        "annotated_frame":    annotated,
        "inference_time_ms":  inference_ms,
    }


# ─────────────────────────────────────────────────────────────
# RUN AS STANDALONE — for testing this file directly
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Just change this to your video path ──
    VIDEO_PATH = "D:/NUST/6th sem/DIP/Project/deep_learning/5.mp4"
    # ─────────────────────────────────────────

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: could not open video at '{VIDEO_PATH}'")
        print("Make sure the path is correct and the file exists.")
    else:
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

            cv2.imshow("DL Module — Full Pipeline", result["annotated_frame"])

            # Press Q to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopped by user.")
                break

        cap.release()
        cv2.destroyAllWindows()