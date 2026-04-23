import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO

# If you're on Windows, uncomment and fix this path:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YOLOv8n once when this file is imported — not inside the function
# so we don't reload the model on every single frame (that would be very slow)
model = YOLO("yolov8n.pt")

# These are the only COCO classes we care about for traffic control
TRAFFIC_CLASSES = {
    9:  "traffic_light",
    11: "stop_sign",
}

# Priority order for actions — higher number = more urgent
# If two detections conflict, the higher priority one wins
ACTION_PRIORITY = {
    "STOP":    4,
    "SLOW":    3,
    "CAUTION": 2,
    "GO":      1,
    "NONE":    0,
}


# ─────────────────────────────────────────────
# TRAFFIC LIGHT COLOR READER
# ─────────────────────────────────────────────

def read_traffic_light_state(frame, bbox):
    """
    Takes the cropped traffic light region and figures out
    if it's showing red, yellow, or green.

    Why we use HSV instead of RGB:
    RGB mixes color and brightness together, so the same red
    looks very different in shadow vs sunlight.
    HSV separates Hue (actual color) from brightness,
    making color detection much more reliable outdoors.
    """
    x1, y1, x2, y2 = bbox

    # Crop just the traffic light from the frame
    crop = frame[y1:y2, x1:x2]

    # If the crop is somehow empty, we can't read it
    if crop.size == 0:
        return "unknown"

    # Resize to a fixed size so our third-splitting math is consistent
    # 30 wide x 90 tall — tall because traffic lights are taller than wide
    crop = cv2.resize(crop, (30, 90))

    # Convert from BGR (OpenCV default) to HSV
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    h = crop.shape[0]  # height = 90

    # Traffic lights always follow the same physical layout:
    # RED on top, YELLOW in middle, GREEN at bottom
    # This is standardized internationally so we can use it as prior knowledge
    top_third    = hsv[0      : h//3,   :]   # where red bulb is
    middle_third = hsv[h//3   : 2*h//3, :]   # where yellow bulb is
    bottom_third = hsv[2*h//3 : h,      :]   # where green bulb is

    # ── Red detection ──
    # Red is special in HSV: the hue wheel goes 0-179 in OpenCV
    # Red sits at BOTH ends of the wheel (0-10 AND 160-179)
    # so we need TWO masks and combine them
    red_mask1 = cv2.inRange(top_third,
                            np.array([0,   100, 100]),   # lower bound
                            np.array([10,  255, 255]))   # upper bound
    red_mask2 = cv2.inRange(top_third,
                            np.array([160, 100, 100]),
                            np.array([179, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # ── Yellow detection ──
    yellow_mask = cv2.inRange(middle_third,
                              np.array([20, 100, 100]),
                              np.array([35, 255, 255]))

    # ── Green detection ──
    green_mask = cv2.inRange(bottom_third,
                             np.array([40, 50, 50]),
                             np.array([90, 255, 255]))

    # Score = what percentage of that region matches the expected color
    # A lit bulb should light up a decent chunk of its third
    red_score    = np.sum(red_mask    > 0) / red_mask.size
    yellow_score = np.sum(yellow_mask > 0) / yellow_mask.size
    green_score  = np.sum(green_mask  > 0) / green_mask.size

    scores = {"red": red_score, "yellow": yellow_score, "green": green_score}
    best_color = max(scores, key=scores.get)

    # If the winning score is too low, no bulb is clearly lit
    # This avoids false readings on dark or unclear crops
    if scores[best_color] < 0.05:
        return "unknown"

    return best_color


# ─────────────────────────────────────────────
# SPEED LIMIT SIGN READER
# ─────────────────────────────────────────────

def read_speed_limit(frame, bbox):
    """
    Crops the speed limit sign region and uses OCR
    (Optical Character Recognition) to read the number on it.

    OCR = software that reads text from images.
    We use pytesseract which wraps Google's Tesseract OCR engine.
    """
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    # Make it bigger — OCR works much better on larger images
    # Small text in a small crop is hard to read
    crop = cv2.resize(crop, (120, 120))

    # Convert to grayscale — OCR doesn't need color, just contrast
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Apply binary threshold: pixels above 100 become white, below become black
    # This gives OCR clean black text on white background to work with
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Tell tesseract we expect only digits (config: digits only, single line)
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(thresh, config=config).strip()

    # Extract just the number from whatever tesseract returned
    import re
    numbers = re.findall(r'\d+', text)

    if numbers:
        limit = int(numbers[0])
        # Sanity check — real speed limits are between 5 and 130 km/h
        if 5 <= limit <= 130:
            return limit

    return None  # couldn't read a valid number


# ─────────────────────────────────────────────
# SCHOOL AHEAD SIGN DETECTOR
# ─────────────────────────────────────────────

def detect_school_sign(frame, bbox):
    """
    School ahead signs are typically yellow/fluorescent with
    a pentagonal (house) shape in many countries.

    Since YOLO doesn't detect school signs (not in COCO),
    we use a classical approach: check if the cropped region
    contains the distinctive yellow/fluorescent color.

    This is a simple heuristic — not perfect, but reasonable
    for a campus environment and shows good hybrid thinking.
    """
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return False

    crop_resized = cv2.resize(crop, (80, 80))
    hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)

    # School signs are bright yellow/yellow-green
    # This HSV range captures that fluorescent warning yellow
    yellow_mask = cv2.inRange(hsv,
                              np.array([20, 120, 120]),
                              np.array([40, 255, 255]))

    # What fraction of the sign is that yellow?
    yellow_ratio = np.sum(yellow_mask > 0) / yellow_mask.size

    # If more than 30% of the region is warning yellow, call it a school sign
    return yellow_ratio > 0.30


# ─────────────────────────────────────────────
# MAIN FUNCTION — runs everything together
# ─────────────────────────────────────────────

def analyze_traffic(frame):
    """
    Main entry point. Takes one frame, returns all traffic
    detections and a single final action decision.
    """
    # Resize for faster inference — same reason as in detector.py
    small = cv2.resize(frame, (640, 360))
    sh, sw = small.shape[:2]

    # Run YOLO on the resized frame
    results = model(small, conf=0.4, verbose=False)
    boxes = results[0].boxes

    detections = []

    for box in boxes:
        cls_id = int(box.cls[0])

        # Skip anything that isn't a traffic light or stop sign
        if cls_id not in TRAFFIC_CLASSES:
            continue

        conf  = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = TRAFFIC_CLASSES[cls_id]

        # Start building this detection's result
        det = {
            "class":      label,
            "confidence": round(conf, 2),
            "bbox":       [x1, y1, x2, y2],
            "state":      None,
            "action":     None,
            "detail":     None,   # extra info e.g. "30 km/h"
        }

        # ── Handle each sign type ──

        if label == "traffic_light":
            state = read_traffic_light_state(small, [x1, y1, x2, y2])
            det["state"] = state
            det["action"] = {
                "red":     "STOP",
                "yellow":  "SLOW",
                "green":   "GO",
                "unknown": "CAUTION",
            }.get(state, "CAUTION")

        elif label == "stop_sign":
            # Stop sign — no further analysis needed, always STOP
            det["state"]  = "stop"
            det["action"] = "STOP"

            # Also check if this might be a school sign
            # (school signs are sometimes octagonal like stop signs)
            if detect_school_sign(small, [x1, y1, x2, y2]):
                det["state"]  = "school_ahead"
                det["action"] = "SLOW"
                det["detail"] = "School zone"

            # Also try reading a speed number from this region
            # in case it's a speed limit sign YOLO mis-classified
            limit = read_speed_limit(small, [x1, y1, x2, y2])
            if limit:
                det["state"]  = "speed_limit"
                det["action"] = "SLOW"
                det["detail"] = f"{limit} km/h"

        detections.append(det)

    # ── Decide the single final action ──
    # Go through all detections and pick the most urgent action
    final_action = "NONE"
    for det in detections:
        if ACTION_PRIORITY.get(det["action"], 0) > ACTION_PRIORITY.get(final_action, 0):
            final_action = det["action"]

    # If nothing was detected, default to GO — road is clear
    if final_action == "NONE":
        final_action = "GO"

    # ── Annotate the frame for display ──
    annotated = results[0].plot()  # YOLO draws its boxes

    # Draw our own labels on top showing state and action
    for det in detections:
        x1, y1 = det["bbox"][0], det["bbox"][1]

        # Build display text e.g. "red → STOP" or "speed_limit → SLOW (30 km/h)"
        label_text = f"{det['state']} → {det['action']}"
        if det["detail"]:
            label_text += f" ({det['detail']})"

        # Color of the text matches urgency
        text_color = {
            "STOP":    (0, 0, 255),    # red
            "SLOW":    (0, 165, 255),  # orange
            "GO":      (0, 255, 0),    # green
            "CAUTION": (0, 255, 255),  # yellow
        }.get(det["action"], (255, 255, 255))

        cv2.putText(annotated, label_text, (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)

    # Big status text at top of frame
    status_color = {
        "STOP": (0, 0, 255),
        "SLOW": (0, 165, 255),
        "GO":   (0, 255, 0),
    }.get(final_action, (255, 255, 255))

    cv2.putText(annotated, f"Traffic: {final_action}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

    return {
        "detections":    detections,    # full list of what was found
        "final_action":  final_action,  # single master decision
        "annotated_frame": annotated,   # frame with everything drawn on it
    }