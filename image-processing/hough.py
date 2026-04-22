import os

import cv2
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_VIDEO = os.path.join(BASE_DIR, "videos", "PXL_20250325_043754655.TS.mp4")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "hough")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "PXL_20250325_043754655.TS.mp4")
DEBUG_VIDEO = os.path.join(OUTPUT_DIR, "PXL_20250325_043754655.TS_debug.mp4")
MAX_FRAMES = None

PREV_LEFT_LINE = None
PREV_RIGHT_LINE = None
SMOOTHING_ALPHA = 0.2


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 70, 170)


def capture_region_of_interest(image):
    height, width = image.shape[:2]
    polygon = np.array(
        [
            (int(0.05 * width), height - 1),
            (int(0.95 * width), height - 1),
            (int(0.78 * width), int(0.60 * height)),
            (int(0.22 * width), int(0.60 * height)),
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
        return None, None

    height, width = image.shape[:2]
    center_x = width / 2.0

    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        if x1 == x2:
            continue

        slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
        length = np.hypot(x2 - x1, y2 - y1)
        x_mid = (x1 + x2) / 2.0
        y_mid = (y1 + y2) / 2.0

        if length < 35:
            continue

        if abs(slope) < 0.15 or abs(slope) > 3.0:
            continue

        if y_mid < 0.50 * height:
            continue

        weight = length * (1.0 + (y_mid / height))

        if x_mid < center_x and slope < 0:
            left_lines.append((float(slope), float(intercept), float(weight)))

        elif x_mid > center_x and slope > 0:
            right_lines.append((float(slope), float(intercept), float(weight)))

    left_line = None
    right_line = None

    if left_lines:
        left_array = np.array(left_lines, dtype=np.float64)
        left_avg = np.average(left_array[:, :2], axis=0, weights=left_array[:, 2])
        left_line = calculate_coordinates(image, left_avg)

    if right_lines:
        right_array = np.array(right_lines, dtype=np.float64)
        right_avg = np.average(right_array[:, :2], axis=0, weights=right_array[:, 2])
        right_line = calculate_coordinates(image, right_avg)

    return left_line, right_line


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
        return None
    if previous_line is None:
        return current_line

    current_line = np.asarray(current_line, dtype=np.float64)
    previous_line = np.asarray(previous_line, dtype=np.float64)

    smoothed = alpha * current_line + (1.0 - alpha) * previous_line
    return smoothed.astype(np.int32)


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

    pip_h = max(120, final_tile.shape[0] // 4)
    pip_w = max(200, final_tile.shape[1] // 4)
    pip = cv2.resize(final_tile, (pip_w, pip_h))
    y1 = panel.shape[0] - pip_h - 20
    x1 = panel.shape[1] - pip_w - 20
    panel[y1 : y1 + pip_h, x1 : x1 + pip_w] = pip
    return panel


def detect_lane(image):
    global PREV_LEFT_LINE, PREV_RIGHT_LINE

    canny_img = canny(image)
    roi_img, roi_mask = capture_region_of_interest(canny_img)

    hough_lines = cv2.HoughLinesP(
        roi_img,
        1,
        np.pi / 180,
        40,
        np.array([]),
        minLineLength=30,
        maxLineGap=60,
    )

    left_line, right_line = average_slope_intercept(image, hough_lines)

    left_line = smooth_line(left_line, PREV_LEFT_LINE, SMOOTHING_ALPHA)
    right_line = smooth_line(right_line, PREV_RIGHT_LINE, SMOOTHING_ALPHA)

    PREV_LEFT_LINE = left_line
    PREV_RIGHT_LINE = right_line

    threshold = 1.0
    if left_line is not None and right_line is not None:
        left_vec = np.array(
            [left_line[2] - left_line[0], left_line[3] - left_line[1]],
            dtype=np.float64,
        )
        right_vec = np.array(
            [right_line[2] - right_line[0], right_line[3] - right_line[1]],
            dtype=np.float64,
        )
        threshold = cosine_distance(left_vec, right_vec)

    lines_to_draw = []
    if left_line is not None:
        lines_to_draw.append((left_line, (0, 255, 255), 5))

    if right_line is not None:
        lines_to_draw.append((right_line, (0, 0, 255), 5))

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