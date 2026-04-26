"""
Road segmentation helpers.

This module takes a BGR frame and returns a road mask plus a few debugging
outputs. It stays separate from YOLO and decision logic so the mask can be
tuned independently.
"""

import cv2
import numpy as np


def normalize_feature(x):
    """Scale a 2-D feature array to the [0, 1] range."""
    x = x.astype(np.float32)
    x_min, x_max = float(x.min()), float(x.max())
    if x_max - x_min < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def road_prior_map(height, width, center_x=0.5, sigma_x=0.28):
    """
    Build a soft prior that favours the lower middle of the frame.

    It nudges cluster selection toward the expected road area, but it does
    not force pixels into the road mask by itself.
    """
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width,  dtype=np.float32)[None, :]

    # Start weighting pixels more heavily below the rough horizon line.
    vertical   = np.clip((y - 0.35) / 0.65, 0.0, 1.0)

    sigma_x  = float(np.clip(sigma_x,   0.16, 0.50))
    center_x = float(np.clip(center_x,  0.0,  1.0))
    horizontal = np.exp(-((x - center_x) ** 2) / (2.0 * sigma_x ** 2))

    return normalize_feature(vertical * horizontal)


def seed_region_mask(height, width):
    """
    Return the bottom-center patch used as the road seed.

    In these videos, the area directly in front of the car is the most
    reliable place to anchor connected-component cleanup.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    y1   = int(0.78 * height)
    x1   = int(0.20 * width)
    x2   = int(0.80 * width)
    mask[y1:height, x1:x2] = 255
    return mask


def road_geometry_mask(height, width):
    """Static trapezoid used when the adaptive road shape is unreliable."""
    polygon = np.array(
        [
            (int(0.04 * width),  height - 1),
            (int(0.96 * width),  height - 1),
            (int(0.62 * width),  int(0.50 * height)),
            (int(0.38 * width),  int(0.50 * height)),
        ],
        dtype=np.int32,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def adaptive_geometry_and_prior(hsv, s_n, v_n):
    """
    Estimate a road-shaped trapezoid from simple colour cues.

    The fitted shape gives the later K-means result a more realistic road
    boundary. If the frame does not provide enough usable pixels, the static
    trapezoid is used instead.
    """
    h, w       = s_n.shape
    seed_mask  = seed_region_mask(h, w)
    green_mask = cv2.inRange(hsv, (35, 30, 30), (95, 255, 255)) > 0

    # Coarse road guess: not too saturated, not too dark, and not vegetation.
    coarse = ((s_n < 0.42) & (v_n > 0.20) & (~green_mask)).astype(np.uint8) * 255
    coarse[:int(0.42 * h), :] = 0                                        # ignore the sky/upper frame
    coarse = cv2.morphologyEx(coarse, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    coarse = cv2.morphologyEx(coarse, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)
    coarse = keep_seed_connected(coarse, seed_mask, max(120, int(0.0015 * h * w)))

    # Scan horizontal slices to find rough road edges.
    y_top    = int(0.50 * h)
    rows     = range(y_top, h, max(1, int(0.03 * h)))
    lefts, rights, ys = [], [], []
    for y in rows:
        xs = np.where(coarse[y] > 0)[0]
        if xs.size >= 6:
            lefts.append(xs[0]  / max(w - 1, 1))
            rights.append(xs[-1] / max(w - 1, 1))
            ys.append(y)

    if len(ys) < 3:
        # Not enough data to fit a shape confidently.
        return road_geometry_mask(h, w), 0.5, 0.28, seed_mask, 0.6

    left_med  = float(np.median(lefts))
    right_med = float(np.median(rights))

    # Estimate road width close to the vehicle from the lowest usable rows.
    bottom_rows = [i for i, y in enumerate(ys) if y >= int(0.82 * h)]
    if bottom_rows:
        left_bottom  = float(np.median([lefts[i]  for i in bottom_rows]))
        right_bottom = float(np.median([rights[i] for i in bottom_rows]))
    else:
        left_bottom, right_bottom = left_med, right_med

    width_ratio = float(np.clip(right_bottom - left_bottom, 0.18, 0.98))
    center_x    = float(np.clip((left_bottom + right_bottom) * 0.5, 0.05, 0.95))
    sigma_x     = float(np.clip(0.50 * width_ratio, 0.18, 0.48))

    # Build the fitted trapezoid.
    top_width = float(np.clip(0.42 * width_ratio, 0.14, 0.70))
    x1b = int(np.clip((center_x - 0.5 * width_ratio) * w, 0, w - 1))
    x2b = int(np.clip((center_x + 0.5 * width_ratio) * w, 0, w - 1))
    x1t = int(np.clip((center_x - 0.5 * top_width)   * w, 0, w - 1))
    x2t = int(np.clip((center_x + 0.5 * top_width)   * w, 0, w - 1))

    polygon  = np.array([(x1b, h - 1), (x2b, h - 1), (x2t, y_top), (x1t, y_top)], dtype=np.int32)
    geom_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(geom_mask, [polygon], 255)

    return geom_mask, center_x, sigma_x, seed_mask, width_ratio


def keep_seed_connected(mask, seed_mask, min_component_area):
    """
    Keep only components that connect back to the bottom-center seed.

    This removes distant same-colour regions, such as walls or sidewalks,
    that K-means can otherwise group with the road.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    seed_ids = set(np.unique(labels[seed_mask > 0]).tolist())
    seed_ids.discard(0)   # background label

    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for lid in seed_ids:
        if int(stats[lid, cv2.CC_STAT_AREA]) >= min_component_area:
            cleaned[labels == lid] = 255

    if cleaned.max() == 0:
        # If the seed misses, keep the largest component rather than returning blank.
        best_id, best_area = None, 0
        for lid in range(1, num_labels):
            area = int(stats[lid, cv2.CC_STAT_AREA])
            if area > best_area:
                best_area, best_id = area, lid
        if best_id is not None and best_area >= min_component_area:
            cleaned[labels == best_id] = 255

    return cleaned


def fill_holes(mask):
    """
    Fill small enclosed gaps inside the road mask.

    Shadows and texture can leave tiny holes in the mask; border-touching
    background areas are left alone so non-road regions are not flooded.
    """
    inv        = (mask == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    filled     = mask.copy()
    h, w       = mask.shape
    max_hole   = int(0.02 * h * w)

    for lid in range(1, num_labels):
        x    = int(stats[lid, cv2.CC_STAT_LEFT])
        y    = int(stats[lid, cv2.CC_STAT_TOP])
        bw   = int(stats[lid, cv2.CC_STAT_WIDTH])
        bh   = int(stats[lid, cv2.CC_STAT_HEIGHT])
        area = int(stats[lid, cv2.CC_STAT_AREA])
        touches_border = x == 0 or y == 0 or (x + bw) >= w or (y + bh) >= h
        if (not touches_border) and area <= max_hole:
            filled[labels == lid] = 255

    return filled


def color_segmentation(
    image,
    clusters=4,
    spatial_weight=0.25,
    min_cluster_ratio=0.015,
    min_component_ratio=0.003,
):
    """
    Segment the drivable road area with K-means colour clustering.

    The feature vector mixes HSV, LAB, and pixel position. Colour separates
    road-like surfaces, while the spatial terms help keep clusters compact
    enough to reject distant patches with similar colour.
    """
    # Smooth while preserving stronger edges.
    smoothed = cv2.bilateralFilter(image, d=7, sigmaColor=60, sigmaSpace=60)

    # Convert to colour spaces that are more useful than raw BGR here.
    hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)

    h_chan, s_chan, v_chan = cv2.split(hsv)
    l_chan, a_chan, b_chan = cv2.split(lab)

    # Normalise channels so no single feature dominates K-means by scale.
    h_n = h_chan.astype(np.float32) / 179.0
    s_n = s_chan.astype(np.float32) / 255.0
    v_n = v_chan.astype(np.float32) / 255.0
    l_n = l_chan.astype(np.float32) / 255.0
    a_n = a_chan.astype(np.float32) / 255.0
    b_n = b_chan.astype(np.float32) / 255.0

    h, w = h_chan.shape
    y_coords, x_coords = np.indices((h, w), dtype=np.float32)
    x_n = x_coords / max(w - 1, 1)
    y_n = y_coords / max(h - 1, 1)

    # Feature matrix: colour channels plus lightly weighted coordinates.
    features    = np.dstack((h_n, s_n, v_n, l_n, a_n, b_n,
                              x_n * spatial_weight, y_n * spatial_weight))
    feature_vec = features.reshape((-1, features.shape[2])).astype(np.float32)

    # Cluster pixels.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.4)
    _, labels, _ = cv2.kmeans(
        feature_vec, clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.reshape((h, w))

    # Score clusters for how road-like they are.
    geom_mask, prior_center, prior_sigma, seed_mask, est_width_ratio = \
        adaptive_geometry_and_prior(hsv=hsv, s_n=s_n, v_n=v_n)

    prior      = road_prior_map(h, w, center_x=prior_center, sigma_x=prior_sigma)
    green_mask = cv2.inRange(hsv, (35, 30, 30), (95, 255, 255)) > 0
    total_px   = float(h * w)

    cluster_scores = []
    cluster_info   = []

    for i in range(clusters):
        mask = labels == i
        if not np.any(mask):
            cluster_scores.append(np.inf)
            cluster_info.append({"cluster": i, "area_ratio": 0.0,
                                  "seed_overlap": 0.0, "prior_overlap": 0.0,
                                  "green_ratio": 0.0, "sat_mean": 1.0, "score": np.inf})
            continue

        area_ratio    = float(mask.sum() / total_px)
        seed_overlap  = float(mask[seed_mask > 0].mean())
        prior_overlap = float(prior[mask].mean())
        green_ratio   = float(green_mask[mask].mean())
        sat_mean      = float(s_n[mask].mean())
        val_mean      = float(v_n[mask].mean())

        # Lower score means the cluster looks more like road.
        score = (
              0.55 * sat_mean
            + 0.35 * green_ratio
            + 0.10 * abs(val_mean - 0.55)
            - 0.60 * seed_overlap
            - 0.40 * prior_overlap
        )
        cluster_scores.append(score)
        cluster_info.append({
            "cluster": i, "area_ratio": area_ratio,
            "seed_overlap": seed_overlap, "prior_overlap": prior_overlap,
            "green_ratio": green_ratio, "sat_mean": sat_mean, "score": score,
        })

    valid_info = [c for c in cluster_info if c["area_ratio"] >= min_cluster_ratio] or cluster_info
    best_score = min(c["score"] for c in valid_info)
    dyn_prior_min = 0.12 if est_width_ratio > 0.70 else 0.25

    candidate_clusters = [
        c["cluster"] for c in valid_info
        if ((c["seed_overlap"] >= 0.10
             or (c["score"] <= best_score + 0.10 and c["prior_overlap"] >= dyn_prior_min))
            and c["green_ratio"] <= 0.22)
    ]

    if not candidate_clusters:
        fallback = sorted(valid_info, key=lambda c: (c["score"], -c["seed_overlap"], -c["area_ratio"]))[0]
        candidate_clusters = [fallback["cluster"]]

    road_cluster = int(sorted(valid_info, key=lambda c: c["score"])[0]["cluster"])

    # Build and clean the final road mask.
    road_mask = np.isin(labels, candidate_clusters).astype(np.uint8) * 255
    road_mask = cv2.bitwise_and(road_mask, geom_mask)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)

    min_comp  = max(150, int(min_component_ratio * h * w))
    road_mask = keep_seed_connected(road_mask, seed_mask, min_comp)
    road_mask = fill_holes(road_mask)
    road_mask = cv2.medianBlur(road_mask, 7)

    # If the mask floods too much of the frame, retry with only the best cluster.
    if float((road_mask > 0).mean()) > 0.75:
        road_mask = (labels == road_cluster).astype(np.uint8) * 255
        road_mask = cv2.bitwise_and(road_mask, geom_mask)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)
        road_mask = keep_seed_connected(road_mask, seed_mask, min_comp)
        road_mask = fill_holes(road_mask)
        road_mask = cv2.medianBlur(road_mask, 7)

    # Visualisation outputs for debugging.
    overlay = image.copy()
    overlay[road_mask == 255] = [0, 255, 0]

    palette   = np.array([[66,133,244],[219,68,55],[244,180,0],[15,157,88],
                           [171,71,188],[0,172,193],[255,112,67],[124,179,66]], dtype=np.uint8)
    color_seg = palette[labels % len(palette)]

    return {
        "labels":            labels,
        "color_seg":         color_seg,
        "road_mask":         road_mask,
        "overlay":           overlay,
        "cluster_scores":    cluster_scores,
        "cluster_info":      cluster_info,
        "candidate_clusters": candidate_clusters,
        "road_cluster":      road_cluster,
    }
