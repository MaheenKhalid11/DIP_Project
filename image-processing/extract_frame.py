import os

import cv2


INPUT_VIDEO = "image-processing/videos/PXL_20250325_043754655.TS.mp4"
OUTPUT_DIR = "image-processing/outputs/frame_extractor"
OUTPUT_FRAME = "image-processing/outputs/frame_extractor/frame_for_hough.jpg"
FRAMES_TO_PROCESS = 60


def extract_frame(video_path, output_path, frame_number):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise ValueError(f"Could not read frame {frame_number} from: {video_path}")

    cv2.imwrite(output_path, frame)
    print(f"Saved frame {frame_number} to: {output_path}")


def main():
    extract_frame(INPUT_VIDEO, OUTPUT_FRAME, FRAMES_TO_PROCESS)


if __name__ == "__main__":
    main()
