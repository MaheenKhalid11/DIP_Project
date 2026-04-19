import os

import cv2
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FRAME = os.path.join(BASE_DIR, "outputs", "frame_extractor", "frame_for_hough.jpg")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "hough_transform")
OUTPUT_IMAGE = os.path.join(OUTPUT_DIR, "frame_hough_output.jpg")

CANNY_LOW = 30
CANNY_HIGH = 120
HOUGH_RHO = 2
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 60
HOUGH_MIN_LINE_LENGTH = 50
HOUGH_MAX_LINE_GAP = 20


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    if abs(slope) < 1e-6:
        return None

    h, w = image.shape[:2]
    y1 = h - 1
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    return np.array([x1, y1, x2, y2], dtype=np.int32)


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x1 == x2:
            continue
        slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
        if abs(slope) < 0.3 or abs(slope) > 3:
            continue
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    lines_out = []
    if left_fit:
        left_line = make_coordinates(image, np.average(left_fit, axis=0))
        if left_line is not None:
            lines_out.append(left_line)
    if right_fit:
        right_line = make_coordinates(image, np.average(right_fit, axis=0))
        if right_line is not None:
            lines_out.append(right_line)

    return np.array(lines_out, dtype=np.int32) if lines_out else None


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 10)
    return line_image


def region_of_interest(image):
    height, width = image.shape[:2]
    polygons = np.array(
        [[
            (int(0.08 * width), height),
            (int(0.92 * width), height),
            (int(0.50 * width), int(0.43 * height)),
        ]],
        dtype=np.int32,
    )
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)


def process(frame):
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(
        cropped_image,
        HOUGH_RHO,
        HOUGH_THETA,
        HOUGH_THRESHOLD,
        np.array([]),
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )
    averaged_lines = average_slope_intercept(frame, lines)
    line_img = display_lines(frame, averaged_lines)
    return cv2.addWeighted(frame, 0.8, line_img, 1, 1)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_FRAME):
        raise FileNotFoundError(f"Could not find input frame: {INPUT_FRAME}")

    frame = cv2.imread(INPUT_FRAME)
    if frame is None:
        raise ValueError(f"Could not read input frame: {INPUT_FRAME}")

    result = process(frame)
    cv2.imwrite(OUTPUT_IMAGE, result)
    print(f"Saved processed frame to: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()
