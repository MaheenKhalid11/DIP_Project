import os

import cv2

from hough_transform import process


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_VIDEO = os.path.join(BASE_DIR, "videos", "PXL_20250325_043754655.TS.mp4")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "hough_transform", "all_frames")
OUTPUT_FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "hough_all_frames.mp4")
MAX_FRAMES = None


def main():
    os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

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

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = process(frame)
        writer.write(result)

        frame_name = f"frame_{frame_index:06d}.jpg"
        frame_path = os.path.join(OUTPUT_FRAMES_DIR, frame_name)
        cv2.imwrite(frame_path, result)

        frame_index += 1
        if MAX_FRAMES is not None and frame_index >= MAX_FRAMES:
            break

    cap.release()
    writer.release()

    print(f"Processed {frame_index} frames")
    print(f"Saved processed frames to: {OUTPUT_FRAMES_DIR}")
    print(f"Saved processed video to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
