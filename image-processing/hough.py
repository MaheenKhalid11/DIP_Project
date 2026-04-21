import os

import cv2
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_VIDEO = os.path.join(BASE_DIR, "videos", "PXL_20250325_045117252.TS.mp4")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "hough")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "PXL_20250325_045117252.TS.mp4")
DEBUG_VIDEO = os.path.join(OUTPUT_DIR, "PXL_20250325_045117252.TS_debug.mp4")
MAX_FRAMES = None

PREV_BOUNDARY_LINE = None
PREV_PATH_LINE = None
SMOOTHING_ALPHA = 0.2
PATH_OFFSET_PIXELS = 220


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 70, 170)


def capture_region_of_interest(image):
    height, width = image.shape[:2]
    polygon = np.array(
        [
            (int(0.10 * width), height - 1),
            (int(0.95 * width), height - 1),
            (int(0.72 * width), int(0.62 * height)),
            (int(0.38 * width), int(0.62 * height)),
        ],
        dtype=np.int32,
    )
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [polygon], 255)
    return cv2.bitwise_and(image, mask), mask


def calculate_coordinates(image, params):
    slope, intercept = params
    if abs(slope) < 1e-6:
        return None

    height, width = image.shape[:2]
    y1 = height - 1
    y2 = int(height * 0.60)

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))

    return np.array([x1, y1, x2, y2], dtype=np.int32)


def average_slope_intercept(image, lines):
    if lines is None:
        return None

    height, width = image.shape[:2]
    center_x = width / 2.0

    boundary_lines = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        if x1 == x2:
            continue

        slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
        length = np.hypot(x2 - x1, y2 - y1)
        x_mid = (x1 + x2) / 2.0
        y_mid = (y1 + y2) / 2.0

        if length < 40:
            continue

        if abs(slope) < 0.35 or abs(slope) > 2.5:
            continue

        # Prefer right-side boundary candidates only
        if x_mid < center_x:
            continue

        # Prefer lines in lower half of the road area
        if y_mid < 0.55 * height:
            continue

        # Score candidates so stronger and more right-side lines dominate
        right_bias = x_mid / width
        weight = length * (1.0 + right_bias)

        boundary_lines.append((float(slope), float(intercept), float(weight)))

    if not boundary_lines:
        return None

    boundary_array = np.array(boundary_lines, dtype=np.float64)
    boundary_avg = np.average(boundary_array[:, :2], axis=0, weights=boundary_array[:, 2])

    boundary_coords = calculate_coordinates(image, boundary_avg)
    if boundary_coords is None:
        return None

    return np.array([boundary_coords], dtype=np.int32)


def cosine_distance(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 1.0
    similarity = float(np.dot(a, b) / denom)
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity


def smooth_line(current_line, previous_line, alpha):
    if current_line is None:
        return previous_line
    if previous_line is None:
        return current_line

    current_line = np.asarray(current_line, dtype=np.float64)
    previous_line = np.asarray(previous_line, dtype=np.float64)

    smoothed = alpha * current_line + (1.0 - alpha) * previous_line
    return smoothed.astype(np.int32)


def create_path_line(boundary_line, offset_pixels):
    if boundary_line is None:
        return None

    x1, y1, x2, y2 = boundary_line.astype(np.float64)

    dx = x2 - x1
    dy = y2 - y1
    length = np.hypot(dx, dy)

    if length < 1e-6:
        return None

    # Unit normal pointing inward from right boundary toward road center
    nx = -dy / length
    ny = dx / length

    shifted_x1 = int(x1 + offset_pixels * nx)
    shifted_y1 = int(y1 + offset_pixels * ny)
    shifted_x2 = int(x2 + offset_pixels * nx)
    shifted_y2 = int(y2 + offset_pixels * ny)

    return np.array([shifted_x1, shifted_y1, shifted_x2, shifted_y2], dtype=np.int32)


def display_lines(image, lines):
    line_image = np.zeros_like(image)

    if lines is not None:
        for item in lines:
            line, color, thickness = item
            x1, y1, x2, y2 = np.asarray(line).reshape(4)
            cv2.line(
                line_image,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                thickness,
            )

    return cv2.addWeighted(image, 0.85, line_image, 1.0, 1.0)


def _to_bgr(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _draw_raw_hough_lines(image, hough_lines):
    raw = image.copy()
    if hough_lines is None:
        return raw
    for line in hough_lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(raw, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    return raw


def _label_tile(image, label):
    tile = image.copy()
    cv2.putText(
        tile,
        label,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return tile


def create_debug_panel(original, canny_img, roi_mask, hough_preview, final_image):
    roi_view = cv2.bitwise_and(original, original, mask=roi_mask)

    top_left = _label_tile(_to_bgr(original), "Original")
    top_right = _label_tile(_to_bgr(canny_img), "Canny")
    bottom_left = _label_tile(_to_bgr(roi_view), "ROI")
    bottom_right = _label_tile(_to_bgr(hough_preview), "Raw Hough")
    final_tile = _label_tile(_to_bgr(final_image), "Final")

    top_row = np.hstack([top_left, top_right])
    bottom_row = np.hstack([bottom_left, bottom_right])
    panel = np.vstack([top_row, bottom_row])

    cv2.rectangle(panel, (0, 0), (panel.shape[1], 52), (0, 0, 0), -1)
    cv2.putText(
        panel,
        "Intermediate Pipeline View",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Append final frame as a PiP thumbnail in the lower-right corner.
    pip_h = max(120, final_tile.shape[0] // 4)
    pip_w = max(200, final_tile.shape[1] // 4)
    pip = cv2.resize(final_tile, (pip_w, pip_h))
    y1 = panel.shape[0] - pip_h - 20
    x1 = panel.shape[1] - pip_w - 20
    panel[y1 : y1 + pip_h, x1 : x1 + pip_w] = pip
    return panel


def detect_lane(image):
    global PREV_BOUNDARY_LINE, PREV_PATH_LINE

    canny_img = canny(image)
    roi_img, roi_mask = capture_region_of_interest(canny_img)

    hough_lines = cv2.HoughLinesP(
        roi_img,
        1,
        np.pi / 180,
        45,
        np.array([]),
        minLineLength=35,
        maxLineGap=90,
    )

    boundary_lines = average_slope_intercept(image, hough_lines)

    boundary_line = None
    if boundary_lines is not None and len(boundary_lines) > 0:
        boundary_line = boundary_lines[0]

    boundary_line = smooth_line(boundary_line, PREV_BOUNDARY_LINE, SMOOTHING_ALPHA)
    PREV_BOUNDARY_LINE = boundary_line

    path_line = create_path_line(boundary_line, PATH_OFFSET_PIXELS)
    path_line = smooth_line(path_line, PREV_PATH_LINE, SMOOTHING_ALPHA)
    PREV_PATH_LINE = path_line

    threshold = 1.0
    if boundary_line is not None and path_line is not None:
        boundary_vec = np.array(
            [boundary_line[2] - boundary_line[0], boundary_line[3] - boundary_line[1]],
            dtype=np.float64,
        )
        path_vec = np.array(
            [path_line[2] - path_line[0], path_line[3] - path_line[1]],
            dtype=np.float64,
        )
        threshold = cosine_distance(boundary_vec, path_vec)

    lines_to_draw = []
    if boundary_line is not None:
        lines_to_draw.append((boundary_line, (0, 255, 255), 5))   # boundary

    lane_marked = display_lines(image, lines_to_draw if lines_to_draw else None)
    hough_preview = _draw_raw_hough_lines(image, hough_lines)
    debug_panel = create_debug_panel(image, canny_img, roi_mask, hough_preview, lane_marked)

    return threshold, lane_marked, debug_panel


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    debug_writer = cv2.VideoWriter(
        DEBUG_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2, height * 2),
    )

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        threshold, final_image, debug_panel = detect_lane(frame)

        cv2.putText(
            final_image,
            f"path-score: {threshold:.3f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(final_image)
        debug_writer.write(debug_panel)

        frame_index += 1
        if MAX_FRAMES is not None and frame_index >= MAX_FRAMES:
            break

    cap.release()
    writer.release()
    debug_writer.release()

    print(f"Processed {frame_index} frames")
    print(f"Saved output video to: {OUTPUT_VIDEO}")
    print(f"Saved debug video to: {DEBUG_VIDEO}")


if __name__ == "__main__":
    main()
