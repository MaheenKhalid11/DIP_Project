import argparse
from pathlib import Path

import cv2
import numpy as np


def normalize_feature(x):
    x = x.astype(np.float32)
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max - x_min < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def road_prior_map(height, width, center_x=0.5, sigma_x=0.28):
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    vertical = np.clip((y - 0.35) / 0.65, 0.0, 1.0)
    sigma_x = float(np.clip(sigma_x, 0.16, 0.50))
    center_x = float(np.clip(center_x, 0.0, 1.0))
    horizontal = np.exp(-((x - center_x) ** 2) / (2.0 * (sigma_x ** 2)))
    return normalize_feature(vertical * horizontal)


def seed_region_mask(height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    y1 = int(0.78 * height)
    x1 = int(0.20 * width)
    x2 = int(0.80 * width)
    mask[y1:height, x1:x2] = 255
    return mask


def road_geometry_mask(height, width):
    polygon = np.array(
        [
            (int(0.04 * width), height - 1),
            (int(0.96 * width), height - 1),
            (int(0.62 * width), int(0.50 * height)),
            (int(0.38 * width), int(0.50 * height)),
        ],
        dtype=np.int32,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def adaptive_geometry_and_prior(hsv, s_n, v_n):
    h, w = s_n.shape
    seed_mask = seed_region_mask(h, w)

    green_mask = cv2.inRange(hsv, (35, 30, 30), (95, 255, 255)) > 0
    # Coarse road-likelihood from color cues in lower image region.
    coarse = ((s_n < 0.42) & (v_n > 0.20) & (~green_mask)).astype(np.uint8) * 255
    horizon = int(0.42 * h)
    coarse[:horizon, :] = 0
    coarse = cv2.morphologyEx(coarse, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    coarse = cv2.morphologyEx(coarse, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    coarse = keep_seed_connected(coarse, seed_mask, max(120, int(0.0015 * h * w)))

    y_top = int(0.50 * h)
    y_bottom = h - 1
    rows = range(y_top, y_bottom + 1, max(1, int(0.03 * h)))
    lefts = []
    rights = []
    ys = []
    for y in rows:
        xs = np.where(coarse[y] > 0)[0]
        if xs.size >= 6:
            lefts.append(xs[0] / max(w - 1, 1))
            rights.append(xs[-1] / max(w - 1, 1))
            ys.append(y)

    if len(ys) < 3:
        return road_geometry_mask(h, w), 0.5, 0.28, seed_mask, 0.6

    left_med = float(np.median(lefts))
    right_med = float(np.median(rights))
    bottom_rows = [i for i, y in enumerate(ys) if y >= int(0.82 * h)]
    if bottom_rows:
        left_bottom = float(np.median([lefts[i] for i in bottom_rows]))
        right_bottom = float(np.median([rights[i] for i in bottom_rows]))
    else:
        left_bottom = left_med
        right_bottom = right_med

    width_ratio = float(np.clip(right_bottom - left_bottom, 0.18, 0.98))
    center_x = float(np.clip((left_bottom + right_bottom) * 0.5, 0.05, 0.95))
    sigma_x = float(np.clip(0.50 * width_ratio, 0.18, 0.48))

    top_width = float(np.clip(0.42 * width_ratio, 0.14, 0.70))
    x1b = int(np.clip((center_x - 0.5 * width_ratio) * w, 0, w - 1))
    x2b = int(np.clip((center_x + 0.5 * width_ratio) * w, 0, w - 1))
    x1t = int(np.clip((center_x - 0.5 * top_width) * w, 0, w - 1))
    x2t = int(np.clip((center_x + 0.5 * top_width) * w, 0, w - 1))
    polygon = np.array([(x1b, h - 1), (x2b, h - 1), (x2t, y_top), (x1t, y_top)], dtype=np.int32)

    geom_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(geom_mask, [polygon], 255)
    return geom_mask, center_x, sigma_x, seed_mask, width_ratio


def keep_seed_connected(mask, seed_mask, min_component_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    seed_ids = set(np.unique(labels[seed_mask > 0]).tolist())
    seed_ids.discard(0)

    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for label_id in seed_ids:
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= min_component_area:
            cleaned[labels == label_id] = 255

    if cleaned.max() == 0:
        # Fallback: keep largest non-background component to avoid empty masks.
        best_id = None
        best_area = 0
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area > best_area:
                best_area = area
                best_id = label_id
        if best_id is not None and best_area >= min_component_area:
            cleaned[labels == best_id] = 255
    return cleaned


def fill_holes(mask):
    # Fill only enclosed background components; never convert border background to road.
    inv = (mask == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    filled = mask.copy()
    h, w = mask.shape
    max_hole_area = int(0.02 * h * w)
    for label_id in range(1, num_labels):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
        bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        touches_border = x == 0 or y == 0 or (x + bw) >= w or (y + bh) >= h
        if (not touches_border) and area <= max_hole_area:
            filled[labels == label_id] = 255
    return filled


def color_segmentation(
    image,
    clusters=4,
    spatial_weight=0.25,
    min_cluster_ratio=0.015,
    min_component_ratio=0.003,
):
    smoothed = cv2.bilateralFilter(image, d=7, sigmaColor=60, sigmaSpace=60)
    hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)

    h_chan, s_chan, v_chan = cv2.split(hsv)
    l_chan, a_chan, b_chan = cv2.split(lab)

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

    features = np.dstack(
        (
            h_n,
            s_n,
            v_n,
            l_n,
            a_n,
            b_n,
            x_n * spatial_weight,
            y_n * spatial_weight,
        )
    )
    feature_vec = features.reshape((-1, features.shape[2])).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.4)
    _compactness, labels, _centers = cv2.kmeans(
        feature_vec,
        clusters,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.reshape((h, w))

    geom_mask, prior_center, prior_sigma, seed_mask, est_width_ratio = adaptive_geometry_and_prior(
        hsv=hsv, s_n=s_n, v_n=v_n
    )
    prior = road_prior_map(h, w, center_x=prior_center, sigma_x=prior_sigma)
    green_mask = cv2.inRange(hsv, (35, 30, 30), (95, 255, 255)) > 0
    total_pixels = float(h * w)

    cluster_scores = []
    cluster_info = []
    for i in range(clusters):
        mask = labels == i
        if not np.any(mask):
            cluster_scores.append(np.inf)
            cluster_info.append(
                {
                    "cluster": i,
                    "area_ratio": 0.0,
                    "seed_overlap": 0.0,
                    "prior_overlap": 0.0,
                    "green_ratio": 0.0,
                    "sat_mean": 1.0,
                    "score": np.inf,
                }
            )
            continue

        area_ratio = float(mask.sum() / total_pixels)
        seed_overlap = float(mask[seed_mask > 0].mean())
        prior_overlap = float(prior[mask].mean())
        green_ratio = float(green_mask[mask].mean())
        sat_mean = float(s_n[mask].mean())
        val_mean = float(v_n[mask].mean())

        # Road is typically low-saturation, low-green, aligned with lower-center prior.
        score = (
            (0.55 * sat_mean)
            + (0.35 * green_ratio)
            + (0.10 * abs(val_mean - 0.55))
            - (0.60 * seed_overlap)
            - (0.40 * prior_overlap)
        )
        cluster_scores.append(score)
        cluster_info.append(
            {
                "cluster": i,
                "area_ratio": area_ratio,
                "seed_overlap": seed_overlap,
                "prior_overlap": prior_overlap,
                "green_ratio": green_ratio,
                "sat_mean": sat_mean,
                "score": score,
            }
        )

    valid_info = [c for c in cluster_info if c["area_ratio"] >= min_cluster_ratio]
    if not valid_info:
        valid_info = cluster_info

    best_score = min(c["score"] for c in valid_info)
    dynamic_prior_min = 0.12 if est_width_ratio > 0.70 else 0.25
    candidate_clusters = []
    for c in valid_info:
        seed_condition = c["seed_overlap"] >= 0.10
        score_condition = c["score"] <= (best_score + 0.10)
        prior_condition = c["prior_overlap"] >= dynamic_prior_min
        low_green = c["green_ratio"] <= 0.22
        if (seed_condition or (score_condition and prior_condition)) and low_green:
            candidate_clusters.append(c["cluster"])

    if not candidate_clusters:
        fallback = sorted(
            valid_info, key=lambda c: (c["score"], -c["seed_overlap"], -c["area_ratio"])
        )[0]
        candidate_clusters = [fallback["cluster"]]

    road_cluster = int(sorted(valid_info, key=lambda c: c["score"])[0]["cluster"])
    road_mask = np.isin(labels, candidate_clusters).astype(np.uint8) * 255

    road_mask = cv2.bitwise_and(road_mask, geom_mask)
    road_mask = cv2.morphologyEx(
        road_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1
    )
    road_mask = cv2.morphologyEx(
        road_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
    )

    min_component_area = max(150, int(min_component_ratio * h * w))
    road_mask = keep_seed_connected(road_mask, seed_mask, min_component_area)
    road_mask = fill_holes(road_mask)
    road_mask = cv2.medianBlur(road_mask, 7)

    # Safety fallback: if mask still too large, use only primary cluster.
    coverage = float((road_mask > 0).mean())
    if coverage > 0.75:
        road_mask = (labels == road_cluster).astype(np.uint8) * 255
        road_mask = cv2.bitwise_and(road_mask, geom_mask)
        road_mask = cv2.morphologyEx(
            road_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1
        )
        road_mask = cv2.morphologyEx(
            road_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
        )
        road_mask = keep_seed_connected(road_mask, seed_mask, min_component_area)
        road_mask = fill_holes(road_mask)
        road_mask = cv2.medianBlur(road_mask, 7)

    overlay = image.copy()
    overlay[road_mask == 255] = [0, 255, 0]

    palette = np.array(
        [
            [66, 133, 244],
            [219, 68, 55],
            [244, 180, 0],
            [15, 157, 88],
            [171, 71, 188],
            [0, 172, 193],
            [255, 112, 67],
            [124, 179, 66],
        ],
        dtype=np.uint8,
    )
    color_seg = palette[labels % len(palette)]

    return {
        "labels": labels,
        "color_seg": color_seg,
        "road_mask": road_mask,
        "overlay": overlay,
        "cluster_scores": cluster_scores,
        "cluster_info": cluster_info,
        "candidate_clusters": candidate_clusters,
        "road_cluster": road_cluster,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Road segmentation from a single image using color + connectivity."
    )
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "segmentation",
        help="Directory for output files",
    )
    parser.add_argument("--clusters", type=int, default=4, help="K-means cluster count")
    parser.add_argument(
        "--spatial-weight",
        type=float,
        default=0.25,
        help="Weight for (x,y) location features in K-means (default: 0.25).",
    )
    parser.add_argument(
        "--min-cluster-ratio",
        type=float,
        default=0.015,
        help="Ignore clusters smaller than this area ratio as artifacts (default: 0.015).",
    )
    parser.add_argument(
        "--min-component-ratio",
        type=float,
        default=0.003,
        help="Ignore connected components smaller than this area ratio (default: 0.003).",
    )
    parser.add_argument("--show", action="store_true", help="Display preview windows")
    args = parser.parse_args()

    if args.clusters < 2:
        raise ValueError("--clusters must be >= 2")
    if args.spatial_weight < 0:
        raise ValueError("--spatial-weight must be >= 0")
    if not (0.0 <= args.min_cluster_ratio <= 1.0):
        raise ValueError("--min-cluster-ratio must be in [0, 1]")
    if not (0.0 <= args.min_component_ratio <= 1.0):
        raise ValueError("--min-component-ratio must be in [0, 1]")

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    image = cv2.resize(image, (800, 500))
    result = color_segmentation(
        image=image,
        clusters=args.clusters,
        spatial_weight=args.spatial_weight,
        min_cluster_ratio=args.min_cluster_ratio,
        min_component_ratio=args.min_component_ratio,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    labels_path = args.output_dir / f"{stem}_color_labels.png"
    color_seg_path = args.output_dir / f"{stem}_color_segments.png"
    road_mask_path = args.output_dir / f"{stem}_road_mask.png"
    overlay_path = args.output_dir / f"{stem}_overlay.png"

    labels_vis = (result["labels"] * (255.0 / max(args.clusters - 1, 1))).astype(np.uint8)
    # cv2.imwrite(str(labels_path), labels_vis)
    # cv2.imwrite(str(color_seg_path), result["color_seg"])
    # cv2.imwrite(str(road_mask_path), result["road_mask"])
    cv2.imwrite(str(overlay_path), result["overlay"])

    print(f"Input image: {args.image}")
    print(f"Spatial weight: {args.spatial_weight}")
    print(f"Cluster scores: {[round(v, 4) for v in result['cluster_scores']]}")
    print(f"Candidate road clusters: {result['candidate_clusters']}")
    print(f"Selected road-like cluster: {result['road_cluster']}")
    print(f"Saved labels: {labels_path}")
    print(f"Saved color segments: {color_seg_path}")
    print(f"Saved road mask: {road_mask_path}")
    print(f"Saved overlay: {overlay_path}")

    if args.show:
        cv2.imshow("Original", image)
        cv2.imshow("Color Segments", result["color_seg"])
        cv2.imshow("Road Mask", result["road_mask"])
        cv2.imshow("Overlay", result["overlay"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()