"""
decision.py
───────────
Responsibility: take outputs from the segmentation and YOLO detection
modules and produce a single final driving action.

This file answers: "Given what we can see, what should the car do?"

It does NOT run any model or process pixels — it only interprets the
structured data passed in from the other two modules.

Public API
──────────
    from decision import get_path_guidance, decide_final_action

    path_data    = get_path_guidance(road_mask, car_zone_bbox)
    final_action = decide_final_action(path_data, obstacle_data, traffic_data)

    path_data keys
        "path_action"       – "FORWARD" / "MOVE_LEFT" / "MOVE_RIGHT" / "SLOW_DOWN"
        "path_offset"       – smoothed normalised offset from road centre (-1 to +1)
        "road_ahead_ratio"  – fraction of lookahead region that is road
        "curvature_hint"    – sign indicates curve direction

    decide_final_action returns one string:
        "FORWARD" / "MOVE_LEFT" / "MOVE_RIGHT" / "SLOW_DOWN" / "STOP"
"""

import numpy as np
from collections import deque

# ──────────────────────────────────────────────────────────────
# SMOOTHING FOR PATH OFFSET
# ──────────────────────────────────────────────────────────────

# Rolling window of recent path offsets — smooths out jitter from the
# road mask varying frame to frame.
# 7 frames at 30 fps ≈ 233 ms of lag, which is acceptable for steering.
_path_offset_history: deque = deque(maxlen=7)


# ──────────────────────────────────────────────────────────────
# PATH GUIDANCE  (uses road mask from segmentation.py)
# ──────────────────────────────────────────────────────────────

def get_path_guidance(road_mask: np.ndarray, car_zone_bbox: list) -> dict:
    """
    Analyses the road mask ahead of the car and returns a steering hint.

    Why three lookahead bands?
    ──────────────────────────
    A single horizontal slice only tells us about one distance.
    Using near / mid / far bands lets us detect a curve early:
    if the far-band road centre drifts left while the near-band stays
    centred, the road is curving left ahead.

    The curvature_hint (far_center - near_center) captures this drift and
    is blended into the desired centre so the system steers into the curve
    slightly before the car reaches it.

    Offset convention
    ─────────────────
    positive offset → road centre is to the RIGHT of the car → steer right
    negative offset → road centre is to the LEFT  of the car → steer left
    """
    h, w          = road_mask.shape[:2]
    x1, y1, x2, _ = car_zone_bbox
    car_center_x   = 0.5 * (x1 + x2)

    # Three horizontal bands above the car icon
    # Each tuple is (top_y, bottom_y) as fractions of frame height
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
            # Median x-coordinate of road pixels in this band
            centers.append(float(np.median(road_pixels[1])))

    road_ratio = float(np.mean(ratios)) if ratios else 0.0

    if not centers:
        # No road visible ahead — play it safe
        _path_offset_history.append(0.0)
        return {
            "path_action":      "SLOW_DOWN",
            "path_offset":      0.0,
            "road_ahead_ratio": round(road_ratio, 3),
            "curvature_hint":   0.0,
        }

    near_center = centers[0]
    far_center  = centers[-1]

    # Curvature: positive = road curves right, negative = curves left
    curvature_hint = (far_center - near_center) / max(w, 1)

    # Blend near centre (dominant) with far centre (anticipation)
    desired_center = 0.75 * near_center + 0.25 * far_center

    # Normalise offset to [-1, +1] relative to frame width
    offset_norm  = (desired_center - car_center_x) / max(w, 1)
    offset_norm += 0.35 * curvature_hint   # add curve anticipation

    _path_offset_history.append(float(offset_norm))
    smooth_offset = float(np.mean(_path_offset_history))

    # Convert offset to a discrete steering action
    # ±0.05 dead-band avoids jittering on a straight road
    if smooth_offset > 0.05:
        action = "MOVE_RIGHT"
    elif smooth_offset < -0.05:
        action = "MOVE_LEFT"
    else:
        action = "FORWARD"

    # Override if not enough road is visible ahead
    if road_ratio < 0.12:
        action = "SLOW_DOWN"

    return {
        "path_action":      action,
        "path_offset":      round(smooth_offset, 3),
        "road_ahead_ratio": round(road_ratio, 3),
        "curvature_hint":   round(float(curvature_hint), 3),
    }


# ──────────────────────────────────────────────────────────────
# AVOIDANCE LOGIC  (uses obstacle data from yolo_detection.py)
# ──────────────────────────────────────────────────────────────

def choose_avoid_action(obstacles: list, car_zone_bbox: list,
                        road_mask: np.ndarray) -> tuple:
    """
    Decides how to react to the most dangerous obstacle.

    Returns (action_string, primary_obstacle_or_None).

    Logic hierarchy
    ───────────────
    1. No risky obstacle in path → CLEAR (keep going, path_data steers).
    2. Very high risk (car_overlap > 0.08 or risk ≥ 0.75) → STOP.
       The obstacle is essentially on top of us.
    3. Medium-high risk → try to go around:
         obstacle to the right → steer LEFT if left side has free road
         obstacle to the left  → steer RIGHT if right side has free road
         no free side           → SLOW_DOWN
    4. Low-medium risk → SLOW_DOWN as a precaution.

    The left_free / right_free check uses the road mask to confirm there
    is actually drivable space to move into — avoiding steering into a wall.
    """
    # Only consider obstacles that are in the path AND have meaningful risk
    risky = [o for o in obstacles if o["in_path"] and o["risk_score"] >= 0.35]
    if not risky:
        return "CLEAR", None

    # Focus on the single most dangerous obstacle
    primary        = max(risky, key=lambda o: o["risk_score"])
    px1, _, px2, _ = primary["bbox"]
    obs_center     = 0.5 * (px1 + px2)
    cx1, _, cx2, _ = car_zone_bbox
    car_center     = 0.5 * (cx1 + cx2)

    # Check how much free road is available on each side of the car
    # (using a horizontal slice between 55 % and 90 % down the frame)
    h, w       = road_mask.shape[:2]
    y_top      = max(0, int(0.55 * h))
    y_bottom   = min(h, int(0.90 * h))
    slice_mask = road_mask[y_top:y_bottom, :]

    left_free  = (float(np.mean(slice_mask[:, :int(car_center)] > 0))
                  if int(car_center) > 1 else 0.0)
    right_free = (float(np.mean(slice_mask[:, int(car_center):] > 0))
                  if int(car_center) < w - 1 else 0.0)

    # Immediate collision risk → STOP regardless of surroundings
    if primary["risk_score"] >= 0.75 or primary["car_overlap"] > 0.08:
        return "STOP", primary

    # Medium-high risk → try to steer around
    if primary["risk_score"] >= 0.55:
        if obs_center >= car_center and left_free > 0.08:
            return "MOVE_LEFT", primary
        if obs_center < car_center and right_free > 0.08:
            return "MOVE_RIGHT", primary
        return "SLOW_DOWN", primary

    # Lower risk → slow down as a precaution
    if primary["risk_score"] >= 0.35:
        return "SLOW_DOWN", primary

    return "CLEAR", primary


# ──────────────────────────────────────────────────────────────
# FINAL DECISION  (fuses all three sources)
# ──────────────────────────────────────────────────────────────

def decide_final_action(path_data: dict, obstacle_data: dict,
                        traffic_data: dict, road_mask: np.ndarray,
                        car_zone_bbox: list) -> tuple:
    """
    Combines path guidance, obstacle avoidance, and traffic signals
    into one final driving action string.

    Priority order (highest → lowest)
    ──────────────────────────────────
    1. Traffic STOP  — a red light or stop sign overrides everything.
    2. Obstacle STOP — an obstacle on the car footprint means stop now.
    3. Lateral avoid — MOVE_LEFT / MOVE_RIGHT to go around an obstacle.
    4. Traffic SLOW  — slow down for yellow light or caution.
    5. Path/obstacle SLOW_DOWN — road ending or medium risk obstacle.
    6. Path steering — MOVE_LEFT / MOVE_RIGHT from lane geometry.
    7. FORWARD       — default when nothing else applies.

    Why this order?
    Traffic signals are legal/safety obligations — ignore them and we cause
    an accident regardless of what the road looks like.
    Obstacle avoidance comes next because a physical collision is immediate.
    Path steering is last because it is the "nominal" state — only relevant
    when everything else is clear.
    """
    avoid_action, primary_risky = choose_avoid_action(
        obstacle_data["obstacles"], car_zone_bbox, road_mask
    )

    traffic_action = traffic_data["traffic_action"]

    # 1. Traffic STOP
    if traffic_action == "STOP":
        return "STOP", avoid_action, primary_risky

    # 2. Obstacle STOP
    if avoid_action == "STOP":
        return "STOP", avoid_action, primary_risky

    # 3. Lateral avoidance
    if avoid_action in {"MOVE_LEFT", "MOVE_RIGHT"}:
        return avoid_action, avoid_action, primary_risky

    # 4. Traffic slow
    if traffic_action in {"SLOW", "CAUTION"}:
        return "SLOW_DOWN", avoid_action, primary_risky

    # 5. Path or obstacle slow
    if avoid_action == "SLOW_DOWN" or path_data["path_action"] == "SLOW_DOWN":
        return "SLOW_DOWN", avoid_action, primary_risky

    # 6. Path steering
    if path_data["path_action"] in {"MOVE_LEFT", "MOVE_RIGHT"}:
        return path_data["path_action"], avoid_action, primary_risky

    # 7. Default
    return "FORWARD", avoid_action, primary_risky
