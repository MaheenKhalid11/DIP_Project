"""
Decision helpers for the driving-assist demo.

This module does not run any models. It takes the road mask and detection
summaries, then chooses one final action for the current frame.
"""

import numpy as np
from collections import deque

# from the segmentation mask without adding much delay.
_path_offset_history: deque = deque(maxlen=7)


def get_path_guidance(road_mask: np.ndarray, car_zone_bbox: list) -> dict:
    """
    Analyse the road mask ahead of the car and return a steering hint.

    The mask is sampled in near, mid, and far bands. Comparing the near and
    far road centers gives a simple curve hint, which is blended into the
    steering offset.
    """
    h, w          = road_mask.shape[:2]
    x1, y1, x2, _ = car_zone_bbox
    car_center_x   = 0.5 * (x1 + x2)

    # Three horizontal lookahead bands above the car icon.
    bands = [
        (max(0, y1 - int(0.16 * h)),  max(1, y1 - int(0.02 * h))),  # near
        (max(0, y1 - int(0.30 * h)),  max(1, y1 - int(0.16 * h))),  # mid
        (max(0, y1 - int(0.46 * h)),  max(1, y1 - int(0.30 * h))),  # far
    ]

    centers = []
    ratios  = []

    for top, bottom in bands:
        band = road_mask[top:bottom, :]
        if band.size == 0:
            continue
        ratios.append(float(np.mean(band > 0)))
        road_pixels = np.where(band > 0)
        if road_pixels[1].size >= 20:
            # Median road x-position in this band.
            centers.append(float(np.median(road_pixels[1])))

    road_ratio = float(np.mean(ratios)) if ratios else 0.0

    if not centers:
        # No road visible ahead.
        _path_offset_history.append(0.0)
        return {
            "path_action":      "SLOW_DOWN",
            "path_offset":      0.0,
            "road_ahead_ratio": round(road_ratio, 3),
            "curvature_hint":   0.0,
        }

    near_center = centers[0]
    far_center  = centers[-1]

    # Positive means the road drifts right; negative means it drifts left.
    curvature_hint = (far_center - near_center) / max(w, 1)

    # Near center dominates, far center gives a bit of anticipation.
    desired_center = 0.75 * near_center + 0.25 * far_center

    # Normalise offset relative to frame width.
    offset_norm  = (desired_center - car_center_x) / max(w, 1)
    offset_norm += 0.35 * curvature_hint   # curve anticipation

    _path_offset_history.append(float(offset_norm))
    smooth_offset = float(np.mean(_path_offset_history))

    # Small dead-band avoids steering jitter on a straight road.
    if smooth_offset > 0.05:
        action = "MOVE_RIGHT"
    elif smooth_offset < -0.05:
        action = "MOVE_LEFT"
    else:
        action = "FORWARD"

    # Slow down if there is not enough road visible ahead.
    if road_ratio < 0.12:
        action = "SLOW_DOWN"

    return {
        "path_action":      action,
        "path_offset":      round(smooth_offset, 3),
        "road_ahead_ratio": round(road_ratio, 3),
        "curvature_hint":   round(float(curvature_hint), 3),
    }


def choose_avoid_action(obstacles: list, car_zone_bbox: list,
                        road_mask: np.ndarray) -> tuple:
    """
    Decide how to react to the highest-risk obstacle in the path.

    Returns the avoidance action and the obstacle that caused it. Side checks
    use the road mask so the car does not steer into non-road space.
    """
    h, w = road_mask.shape[:2]
    cx1, cy1, cx2, _ = car_zone_bbox
    car_center = 0.5 * (cx1 + cx2)

    # 1) Hard blockers from mask overlap.
    in_path = [o for o in obstacles if o["in_path"]]
    if in_path:
        primary = max(in_path, key=lambda o: o["risk_score"])
        return "STOP", primary

    # 2) Fallback: visually front-center blockers (even if mask overlap misses).
    front_candidates = []
    for o in obstacles:
        x1, y1, x2, y2 = o["bbox"]
        obs_center = 0.5 * (x1 + x2)
        center_dist = abs(obs_center - car_center) / max(w, 1)
        in_front_band = (y2 >= int(0.40 * h)) and (y1 <= cy1 + int(0.08 * h))
        near_lane_center = center_dist <= 0.24
        if in_front_band and near_lane_center and o["risk_score"] >= 0.20:
            front_candidates.append(o)

    if not front_candidates:
        return "CLEAR", None

    primary = max(front_candidates, key=lambda o: (o["risk_score"], o["bbox"][3]))
    # If the object is very close in image space, stop; otherwise slow down.
    return ("STOP", primary) if primary["bbox"][3] >= int(0.58 * h) else ("SLOW_DOWN", primary)


def decide_final_action(path_data: dict, obstacle_data: dict,
                        traffic_data: dict, road_mask: np.ndarray,
                        car_zone_bbox: list) -> tuple:
    """
    Combine path guidance, obstacle avoidance, and traffic signals.

    Traffic stops have the highest priority, then immediate obstacle risk,
    then avoidance, slowdown, steering, and finally forward motion.
    """
    avoid_action, primary_risky = choose_avoid_action(
        obstacle_data["obstacles"], car_zone_bbox, road_mask
    )

    traffic_action = traffic_data["traffic_action"]

    # Traffic STOP.
    if traffic_action == "STOP":
        return "STOP", avoid_action, primary_risky

    # Obstacle STOP.
    if avoid_action == "STOP":
        return "STOP", avoid_action, primary_risky

    # Lateral avoidance.
    if avoid_action in {"MOVE_LEFT", "MOVE_RIGHT"}:
        return avoid_action, avoid_action, primary_risky

    # Traffic slowdown.
    if traffic_action in {"SLOW", "CAUTION"}:
        return "SLOW_DOWN", avoid_action, primary_risky

    # Path or obstacle slowdown.
    if avoid_action == "SLOW_DOWN" or path_data["path_action"] == "SLOW_DOWN":
        return "SLOW_DOWN", avoid_action, primary_risky

    # Path steering.
    if path_data["path_action"] in {"MOVE_LEFT", "MOVE_RIGHT"}:
        return path_data["path_action"], avoid_action, primary_risky

    # Default.
    return "FORWARD", avoid_action, primary_risky
