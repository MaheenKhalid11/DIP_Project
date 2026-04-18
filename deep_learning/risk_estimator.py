import numpy as np

def get_path_zone(frame_width, frame_height):
    """
    Returns a trapezoid representing the danger zone ahead.
    Wide at the bottom (close to car), narrow at top (far away).
    """
    w, h = frame_width, frame_height
    zone = np.array([
        [int(w * 0.35), int(h * 0.5)],   # top-left
        [int(w * 0.65), int(h * 0.5)],   # top-right
        [int(w * 0.85), h],               # bottom-right
        [int(w * 0.15), h],               # bottom-left
    ], dtype=np.int32)
    return zone

def bbox_zone_overlap(bbox, zone, frame_width, frame_height):
    """
    Returns what fraction of the bounding box overlaps the path zone.
    Uses a simple approximation with a mask.
    """
    import cv2
    x1, y1, x2, y2 = bbox
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.fillPoly(mask, [zone], 255)
    
    # Region of bbox in mask
    bbox_region = mask[y1:y2, x1:x2]
    if bbox_region.size == 0:
        return 0.0
    
    overlap = np.sum(bbox_region > 0) / bbox_region.size
    return overlap

def estimate_risk(detection, zone, frame_width, frame_height):
    """
    Returns risk_score (0.0 to 1.0) and proximity label.
    """
    x1, y1, x2, y2 = detection["bbox"]
    
    # How much of the box is in the path zone
    path_overlap = bbox_zone_overlap(
        detection["bbox"], zone, frame_width, frame_height
    )
    
    # Box bottom position (higher y = closer to camera = closer obstacle)
    bottom_y_ratio = y2 / frame_height  # 0=top, 1=bottom
    
    # Box size relative to frame
    box_area_ratio = ((x2 - x1) * (y2 - y1)) / (frame_width * frame_height)
    
    # Combine into a risk score
    risk_score = (
        path_overlap * 0.5 +
        bottom_y_ratio * 0.3 +
        min(box_area_ratio * 10, 1.0) * 0.2
    )
    risk_score = round(min(risk_score, 1.0), 2)
    
    # Proximity label
    if risk_score > 0.65:
        proximity = "NEAR"
    elif risk_score > 0.35:
        proximity = "MEDIUM"
    else:
        proximity = "FAR"
    
    detection["path_overlap"] = round(path_overlap, 2)
    detection["risk_score"] = risk_score
    detection["proximity"] = proximity
    detection["in_path"] = path_overlap > 0.25  # at least 25% overlap
    
    return detection