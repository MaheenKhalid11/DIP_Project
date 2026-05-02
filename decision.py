"""
Decision helpers for the driving-assist demo.

This module does not run any models. It takes the road mask and detection
summaries, then chooses one final action for the current frame.
"""

import numpy as np
from collections import deque

_path_offset_history: deque = deque(maxlen=7)


def get_path_guidance(road_mask: np.ndarray, car_zone_bbox: list) -> dict:
    """
    Analyse the road mask ahead of the car and return a steering hint.

    The mask is sampled in near, mid, and far bands. Comparing the near and
    far road centers gives a simple curve hint, which is blended into the
    steering offset.
    """
    h, w           = road_mask.shape[:2]
    x1, y1, x2, _ = car_zone_bbox
    car_center_x   = 0.5 * (x1 + x2)

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
            centers.append(float(np.median(road_pixels[1])))

    road_ratio = float(np.mean(ratios)) if ratios else 0.0

    if not centers:
        _path_offset_history.append(0.0)
        return {
            "path_action":      "SLOW_DOWN",
            "path_offset":      0.0,
            "road_ahead_ratio": round(road_ratio, 3),
            "curvature_hint":   0.0,
        }

    near_center    = centers[0]
    far_center     = centers[-1]
    curvature_hint = (far_center - near_center) / max(w, 1)
    desired_center = 0.75 * near_center + 0.25 * far_center
    offset_norm    = (desired_center - car_center_x) / max(w, 1)
    offset_norm   += 0.35 * curvature_hint

    _path_offset_history.append(float(offset_norm))
    smooth_offset = float(np.mean(_path_offset_history))

    if smooth_offset > 0.05:
        action = "MOVE_RIGHT"
    elif smooth_offset < -0.05:
        action = "MOVE_LEFT"
    else:
        action = "FORWARD"

    if road_ratio < 0.12:
        action = "SLOW_DOWN"

    return {
        "path_action":      action,
        "path_offset":      round(smooth_offset, 3),
        "road_ahead_ratio": round(road_ratio, 3),
        "curvature_hint":   round(float(curvature_hint), 3),
    }


def _check_road_behind(road_mask: np.ndarray, car_zone_bbox: list) -> float:
    """
    Check how much road is visible BEHIND the car.

    Samples one band just below the car zone. Returns a ratio 0.0-1.0.
    Used to decide whether reversing is safe.
    """
    h, w            = road_mask.shape[:2]
    x1, y1, x2, y2 = car_zone_bbox

    # One band just below the car icon bottom edge.
    band_top    = min(y2 + int(0.01 * h), h - 1)
    band_bottom = min(y2 + int(0.10 * h), h)

    if band_bottom <= band_top:
        return 0.0

    band = road_mask[band_top:band_bottom, x1:x2]
    if band.size == 0:
        return 0.0

    return float(np.mean(band > 0))


def choose_avoid_action(obstacles: list, car_zone_bbox: list,
                        road_mask: np.ndarray) -> tuple:
    """
    Decide how to react to the highest-risk obstacle in the path.

    Priority:
      1. Hard mask-overlap blockers  -> try left/right, else STOP
      2. Visual front-center blockers -> try left/right based on free space,
                                         STOP if very close, SLOW_DOWN otherwise
      3. Nothing blocking             -> CLEAR
    """
    h, w             = road_mask.shape[:2]
    cx1, cy1, cx2, _ = car_zone_bbox
    car_center        = 0.5 * (cx1 + cx2)

    # ------------------------------------------------------------------ #
    # Helper: given an obstacle bbox, check which side has more free road #
    # ------------------------------------------------------------------ #
    def _side_action(obs):
        ox1, oy1, ox2, oy2 = obs["bbox"]
        check_row  = min(int(oy2), road_mask.shape[0] - 1)
        road_row   = road_mask[check_row, :]
        left_space  = float(np.sum(road_row[:max(0, int(ox1))] > 0))
        right_space = float(np.sum(road_row[min(int(ox2), w):] > 0))
        min_space   = 0.12 * w          # at least 12% of width must be free

        if left_space > right_space and left_space >= min_space:
            return "MOVE_LEFT"
        elif right_space > left_space and right_space >= min_space:
            return "MOVE_RIGHT"
        else:
            return "STOP"

    # ------------------------------------------------------------------ #
    # 1. Hard blockers from mask overlap                                  #
    # ------------------------------------------------------------------ #
    in_path = [o for o in obstacles if o["in_path"]]
    if in_path:
        primary = max(in_path, key=lambda o: o["risk_score"])
        side    = _side_action(primary)
        return side, primary          # MOVE_LEFT / MOVE_RIGHT / STOP

    # ------------------------------------------------------------------ #
    # 2. Fallback: visually front-center blockers                        #
    # ------------------------------------------------------------------ #
    front_candidates = []
    for o in obstacles:
        x1, y1, x2, y2 = o["bbox"]
        obs_center      = 0.5 * (x1 + x2)
        center_dist     = abs(obs_center - car_center) / max(w, 1)
        in_front_band   = (y2 >= int(0.40 * h)) and (y1 <= cy1 + int(0.08 * h))
        near_lane_center = center_dist <= 0.24
        if in_front_band and near_lane_center and o["risk_score"] >= 0.20:
            front_candidates.append(o)

    if not front_candidates:
        return "CLEAR", None

    primary = max(front_candidates, key=lambda o: (o["risk_score"], o["bbox"][3]))

    # Very close (bottom edge below 58% of frame) -> try to go around or stop
    if primary["bbox"][3] >= int(0.58 * h):
        return _side_action(primary), primary

    return "SLOW_DOWN", primary


def decide_final_action(path_data: dict, obstacle_data: dict,
                        traffic_data: dict, road_mask: np.ndarray,
                        car_zone_bbox: list) -> tuple:
    """
    Combine path guidance, obstacle avoidance, and traffic signals into
    one final action.

    Priority order (highest first):
      1. Traffic STOP  (stop sign)
      2. Obstacle STOP (no room to go around)
      3. Lateral avoidance  (MOVE_LEFT / MOVE_RIGHT around obstacle)
      4. Traffic SLOW / CAUTION
      5. Obstacle or path SLOW_DOWN
      6. BACKWARD  (dead-end: no road ahead, nothing blocking behind)
      7. Path steering (road curves left/right)
      8. FORWARD   (default)
    """
    avoid_action, primary_risky = choose_avoid_action(
        obstacle_data["obstacles"], car_zone_bbox, road_mask
    )
    traffic_action = traffic_data["traffic_action"]
    road_ahead     = path_data["road_ahead_ratio"]

    # 1. Traffic STOP
    if traffic_action == "STOP":
        return "STOP", avoid_action, primary_risky

    # 2. Obstacle STOP (no room either side)
    if avoid_action == "STOP":
        return "STOP", avoid_action, primary_risky

    # 3. Lateral avoidance around obstacle
    if avoid_action in {"MOVE_LEFT", "MOVE_RIGHT"}:
        return avoid_action, avoid_action, primary_risky

    # 4. Traffic slowdown
    if traffic_action in {"SLOW", "CAUTION"}:
        return "SLOW_DOWN", avoid_action, primary_risky

    # 5. Obstacle or path slowdown
    if avoid_action == "SLOW_DOWN" or path_data["path_action"] == "SLOW_DOWN":
        return "SLOW_DOWN", avoid_action, primary_risky

    # 6. BACKWARD — road ahead is gone and nothing is blocking behind
    if road_ahead < 0.06:
        road_behind = _check_road_behind(road_mask, car_zone_bbox)
        if road_behind >= 0.20 and avoid_action == "CLEAR":
            return "BACKWARD", avoid_action, primary_risky
        # Road ahead gone but can't reverse either -> stop
        return "STOP", avoid_action, primary_risky

    # 7. Path steering
    if path_data["path_action"] in {"MOVE_LEFT", "MOVE_RIGHT"}:
        return path_data["path_action"], avoid_action, primary_risky

    # 8. Default
    return "FORWARD", avoid_action, primary_risky