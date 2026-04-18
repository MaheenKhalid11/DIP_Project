# COCO class IDs relevant to driving
RELEVANT_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    58: "potted plant",   # catches your flower pots
    9: "traffic light",
    11: "stop sign",
}

def filter_detections(results, conf_threshold=0.4, frame_width=640, frame_height=360):
    """Returns list of dicts for relevant detections only."""
    detections = []
    boxes = results[0].boxes

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id not in RELEVANT_CLASSES:
            continue
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Ignore tiny detections (noise)
        box_area = (x2 - x1) * (y2 - y1)
        total_area = frame_width * frame_height
        if box_area / total_area < 0.005:  # less than 0.5% of frame
            continue

        detections.append({
            "class": RELEVANT_CLASSES[cls_id],
            "class_id": cls_id,
            "confidence": round(conf, 2),
            "bbox": [x1, y1, x2, y2],
        })

    return detections

from collections import deque

class DetectionSmoother:
    def __init__(self, window=5):
        self.history = deque(maxlen=window)

    def update(self, obstacle_in_path, risk_score):
        self.history.append((obstacle_in_path, risk_score))

    def get_stable(self):
        if not self.history:
            return False, 0.0
        in_path_votes = sum(1 for x, _ in self.history if x)
        avg_risk = sum(r for _, r in self.history) / len(self.history)
        # Need majority vote to say obstacle is in path
        stable_in_path = in_path_votes >= (len(self.history) / 2)
        return stable_in_path, round(avg_risk, 2)