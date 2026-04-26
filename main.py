"""
Main runner for the driving-assist demo.

The loop reads a frame, runs road segmentation and object detection in
separate worker threads, combines their outputs into a decision, and draws
the result on the video.
"""

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

import cv2
import numpy as np

from segmentation import color_segmentation
from yolo_detection import YOLODetector, estimate_risk
from decision import get_path_guidance, decide_final_action


# Basic settings. Update the paths before running on a new machine/video.

VIDEO_PATH          = "/Users/musfiraaslam/Documents/GitHub/Dip-Muh/DIP_Project/deep_learning/videos/PXL_20250325_043922504.TS.mp4"        # set your video path here
COCO_WEIGHTS        = "models/yolov8n.pt"
BARRIER_WEIGHTS     = "models/boom_barrier_best.pt"
CAR_ICON_PATH       = "assets/images/car.png"  # set to None if you have no icon
OUTPUT_VIDEO_PATH   = "outputs/result.mp4"    # set to None to skip saving
SHOW_DISPLAY        = True                    # set False for headless runs
FRAME_WIDTH         = 640
FRAME_HEIGHT        = 360

def _load_car_icon(path):
    """Load the optional car overlay image."""
    if path is None:
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return img   # cv2 returns None if the file is missing


def get_car_zone_bbox(frame_w, frame_h):
    """Return the on-screen footprint used for the car overlay and risk checks."""
    target_w = max(138, int(0.40 * frame_w))
    target_h = max(84,  int(0.28 * frame_h))
    x1 = (frame_w - target_w) // 2
    y1 = frame_h - target_h - int(0.03 * frame_h)
    return [x1, y1, x1 + target_w, y1 + target_h]


def get_car_zone_masks(frame_w, frame_h, car_icon):
    """
    Build the car footprint mask plus a wider safety buffer around it.

    The risk code uses the tighter mask for direct overlap and the wider one
    to catch objects that are close enough to react to.
    """
    x1, y1, x2, y2 = get_car_zone_bbox(frame_w, frame_h)
    w, h = x2 - x1, y2 - y1

    car_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    if car_icon is not None:
        icon = cv2.resize(car_icon, (w, h), interpolation=cv2.INTER_AREA)
        if icon.shape[2] == 4:
            local = (icon[:, :, 3] > 20).astype(np.uint8) * 255
        else:
            gray  = cv2.cvtColor(icon[:, :, :3], cv2.COLOR_BGR2GRAY)
            local = (gray > 20).astype(np.uint8) * 255
        car_mask[y1:y2, x1:x2] = local
    else:
        cv2.rectangle(car_mask, (x1, y1), (x2, y2), 255, -1)

    danger_mask = cv2.dilate(
        car_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        iterations=2,
    )
    return car_mask, danger_mask


def overlay_car_icon(frame, car_icon, frame_w, frame_h):
    """Draw the car overlay, respecting alpha when the image has it."""
    if car_icon is None:
        return frame
    x1, y1, x2, y2 = get_car_zone_bbox(frame_w, frame_h)
    w, h = x2 - x1, y2 - y1
    icon = cv2.resize(car_icon, (w, h), interpolation=cv2.INTER_AREA)
    if icon.shape[2] == 4:
        alpha = (icon[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
        rgb   = icon[:, :, :3].astype(np.float32)
        bg    = frame[y1:y2, x1:x2].astype(np.float32)
        frame[y1:y2, x1:x2] = (alpha * rgb + (1.0 - alpha) * bg).astype(np.uint8)
    else:
        frame[y1:y2, x1:x2] = icon[:, :, :3]
    return frame


class RoadMaskSmoother:
    """
    Smooth road masks between frames.

    K-means can jump a little from frame to frame, so this keeps a lightweight
    moving average and updates more slowly when the new mask is very different.
    """
    def __init__(self, fast_alpha=0.45, slow_alpha=0.18,
                 threshold=0.50, iou_gate=0.35):
        self.fast_alpha = fast_alpha
        self.slow_alpha = slow_alpha
        self.threshold  = threshold
        self.iou_gate   = iou_gate
        self.ema_mask   = None

    def update(self, mask: np.ndarray) -> np.ndarray:
        current = (mask > 0).astype(np.float32)
        if self.ema_mask is None:
            self.ema_mask = current.copy()
        else:
            prev  = self.ema_mask >= self.threshold
            curr  = current > 0.5
            inter = float(np.logical_and(prev, curr).sum())
            union = float(np.logical_or(prev, curr).sum()) + 1e-6
            alpha = self.fast_alpha if (inter / union) >= self.iou_gate else self.slow_alpha
            self.ema_mask = (1.0 - alpha) * self.ema_mask + alpha * current

        stable = (self.ema_mask >= self.threshold).astype(np.uint8) * 255
        stable = cv2.morphologyEx(stable, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        stable = cv2.medianBlur(stable, 5)
        return stable


def annotate(frame_small, coco_results, barrier_results, barrier_dets,
             road_mask, roi_polygon, road_coverage,
             obstacle_data, traffic_data,
             path_data, final_action, avoid_action,
             car_zone_bbox, danger_mask,
             car_icon, inference_ms):
    """
    Draw the model outputs and final driving decision on a frame.

    This is intentionally display-only; it should not change any of the
    decision data passed in from the processing pipeline.
    """
    # Start with YOLO's own box drawing.
    annotated = coco_results[0].plot()

    def draw_label(img, text, x, y, text_color, bg_color=(0, 0, 0), alpha=0.65):
        """Draw text with a semi-transparent background for readability."""
        h, w = img.shape[:2]
        x = int(np.clip(x, 0, max(w - 1, 0)))
        y = int(np.clip(y, 16, max(h - 1, 16)))
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.52
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x2 = min(w - 1, x + tw + 8)
        y1 = max(0, y - th - 8)
        y2 = min(h - 1, y + 4)

        if x2 > x and y2 > y1:
            roi = img[y1:y2, x:x2]
            overlay = np.full_like(roi, bg_color, dtype=np.uint8)
            cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)

        cv2.putText(img, text, (x + 4, y - 4), font, scale, text_color, thickness, cv2.LINE_AA)

    # The separate barrier model gets an orange box so it is easy to spot.
    for b in barrier_dets:
        x1, y1, x2, y2 = b["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 80, 255), 2)
        cv2.putText(annotated, f"boom_barrier {b['confidence']:.2f}",
                    (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 80, 255), 2)

    # Road overlay.
    road_layer = np.zeros_like(annotated)
    road_layer[:, :, 1] = road_mask
    annotated = cv2.addWeighted(annotated, 1.0, road_layer, 0.30, 0)

    # Road boundary contour.
    cv2.polylines(annotated, [roi_polygon], isClosed=True,
                  color=(120, 255, 120), thickness=1)
    cv2.putText(annotated, f"Road: {road_coverage:.2f}",
                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 120), 2)

    # Car marker.
    h, w = annotated.shape[:2]
    annotated = overlay_car_icon(annotated, car_icon, w, h)

    # Safety buffer around the car marker.
    contours, _ = cv2.findContours(danger_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(annotated, [c], -1, (0, 180, 255), 1)
        x, y, _, _ = cv2.boundingRect(c)
        cv2.putText(annotated, "safety zone", (x, max(12, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 180, 255), 1)

    # Per-obstacle labels: keep class + risk only to avoid clutter.
    for obs in obstacle_data["obstacles"]:
        x1, y1 = obs["bbox"][0], obs["bbox"][1]
        label = f"{obs['class']}  r:{obs['risk_score']:.2f}"
        if obs["in_path"]:
            draw_label(annotated, label, x1, max(y1 - 8, 18), (255, 255, 255), (0, 0, 180))
        else:
            draw_label(annotated, label, x1, max(y1 - 8, 18), (255, 255, 255), (40, 120, 40))

    # Traffic signal labels.
    for tdet in traffic_data["traffic_detections"]:
        x1, y1     = tdet["bbox"][0], tdet["bbox"][1]
        label_text = f"{tdet['class']} -> {tdet['action']}"
        if tdet["detail"]:
            label_text += f" ({tdet['detail']})"
        text_color = {"STOP": (255, 255, 255), "SLOW": (0, 0, 0),
                      "GO": (0, 0, 0), "CAUTION": (0, 0, 0)}.get(
                      tdet["action"], (255, 255, 255))
        bg_color = {"STOP": (0, 0, 200), "SLOW": (0, 190, 255),
                    "GO": (60, 220, 60), "CAUTION": (0, 220, 220)}.get(
                    tdet["action"], (80, 80, 80))
        draw_label(annotated, label_text, x1, max(y1 - 24, 18), text_color, bg_color)

    # Path offset info.
    cv2.putText(
        annotated,
        f"path: {path_data['path_action']}  offset:{path_data['path_offset']:+.2f}",
        (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 255, 180), 2,
    )

    # Final action banner.
    banner_color = {
        "STOP":       (0,   0,   255),
        "SLOW_DOWN":  (0,   165, 255),
        "FORWARD":    (0,   255, 0),
        "MOVE_LEFT":  (255, 220, 0),
        "MOVE_RIGHT": (255, 220, 0),
    }.get(final_action, (255, 255, 255))

    cv2.rectangle(annotated, (0, 0), (320, 55), (0, 0, 0), -1)
    cv2.putText(annotated, f"ACTION: {final_action}",
                (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, banner_color, 2)

    # FPS counter.
    fps_text = f"FPS: {1000/max(inference_ms,1):.1f}"
    cv2.putText(annotated, fps_text,
                (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    return annotated


def mask_to_polygon(mask, fallback_w, fallback_h):
    """Convert the largest mask contour into a polygon for drawing."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Use a generic road shape when no contour is available.
        w, h = fallback_w, fallback_h
        return np.array([[int(0.04*w), h-1], [int(0.96*w), h-1],
                          [int(0.62*w), int(0.5*h)], [int(0.38*w), int(0.5*h)]])
    largest = max(contours, key=cv2.contourArea)
    hull    = cv2.convexHull(largest)
    epsilon = 0.015 * cv2.arcLength(hull, True)
    approx  = cv2.approxPolyDP(hull, epsilon, True)
    return approx.reshape(-1, 2)


class FramePipeline:
    """
    Small two-worker pipeline for one frame at a time.

    Segmentation and YOLO do not depend on each other, so running them in
    parallel keeps the main loop more responsive without copying frames
    between processes.
    """

    def __init__(self, detector: YOLODetector,
                 car_zone_mask, danger_zone_mask,
                 road_smoother: RoadMaskSmoother):
        self.detector         = detector
        self.car_zone_mask    = car_zone_mask
        self.danger_zone_mask = danger_zone_mask
        self.road_smoother    = road_smoother
        self._pool            = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pipeline")
        self._seg_future      = None
        self._det_future      = None

    def _run_segmentation(self, frame: np.ndarray):
        """Run road segmentation and return the processed mask bundle."""
        seg = color_segmentation(frame, clusters=4, spatial_weight=0.25)
        raw = seg["road_mask"]
        smooth = self.road_smoother.update(raw)
        polygon = mask_to_polygon(smooth, frame.shape[1], frame.shape[0])
        coverage = round(float(np.sum(smooth > 0)) / float(frame.shape[0] * frame.shape[1]), 3)
        return {
            "mask":        smooth,
            "roi_polygon": polygon,
            "coverage":    coverage,
        }

    def _run_detection(self, frame: np.ndarray):
        """Run YOLO detection for the given frame."""
        return self.detector.process_frame(
            frame, self.car_zone_mask, self.danger_zone_mask
        )

    def submit(self, frame: np.ndarray):
        """Send a fresh frame to both workers."""
        # Copy once so both workers read identical frame content safely.
        frame_copy = frame.copy()
        self._seg_future = self._pool.submit(self._run_segmentation, frame_copy)
        self._det_future = self._pool.submit(self._run_detection, frame_copy)

    def get_results(self, timeout: float = 5.0):
        """Return both worker results, or (None, None) if either times out."""
        if self._seg_future is None or self._det_future is None:
            return None, None
        try:
            seg_result = self._seg_future.result(timeout=timeout)
            det_result = self._det_future.result(timeout=timeout)
        except FutureTimeoutError:
            return None, None
        return seg_result, det_result

    def stop(self):
        """Ask worker threads to exit."""
        self._pool.shutdown(wait=True, cancel_futures=True)


def main():
    # Load assets.
    car_icon = _load_car_icon(CAR_ICON_PATH)
    detector = YOLODetector(COCO_WEIGHTS, BARRIER_WEIGHTS)

    # Open input video.
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: could not open video at '{VIDEO_PATH}'")
        print("Check that the path is correct and the file exists.")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 25.0

    # Set up output writer if requested.
    writer = None
    if OUTPUT_VIDEO_PATH:
        out_path = Path(OUTPUT_VIDEO_PATH)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            video_fps,
            (FRAME_WIDTH, FRAME_HEIGHT),
        )
        print(f"Saving output to: {out_path}")

    # Masks stay fixed because every frame is resized to the same size.
    car_zone_bbox  = get_car_zone_bbox(FRAME_WIDTH, FRAME_HEIGHT)
    car_mask, danger_mask = get_car_zone_masks(FRAME_WIDTH, FRAME_HEIGHT, car_icon)
    road_smoother  = RoadMaskSmoother()

    # Start the worker pipeline.
    pipeline = FramePipeline(detector, car_mask, danger_mask, road_smoother)

    print("Running — press Q to quit\n")
    frame_count  = 0
    total_ms     = 0.0

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("\nVideo finished.")
            break

        # Resize to the working resolution used by all masks.
        frame_small = cv2.resize(raw_frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Submit to both worker threads.
        t_frame_start = time.time()
        pipeline.submit(frame_small)

        # This demo waits here, but a robot loop could do other work meanwhile.
        seg_result, det_result = pipeline.get_results(timeout=5.0)

        if seg_result is None or det_result is None:
            # Skip this frame if either worker stalls.
            print("\nWarning: worker thread timed out, skipping frame.")
            continue

        frame_ms = (time.time() - t_frame_start) * 1000
        total_ms += frame_ms
        frame_count += 1

        # Decision stage.
        path_data = get_path_guidance(seg_result["mask"], car_zone_bbox)

        final_action, avoid_action, primary_risky = decide_final_action(
            path_data       = path_data,
            obstacle_data   = det_result,
            traffic_data    = det_result,
            road_mask       = seg_result["mask"],
            car_zone_bbox   = car_zone_bbox,
        )

        # Draw the debug/decision overlay.
        annotated = annotate(
            frame_small     = frame_small,
            coco_results    = det_result["coco_results"],
            barrier_results = det_result["barrier_results"],
            barrier_dets    = det_result["barrier_dets"],
            road_mask       = seg_result["mask"],
            roi_polygon     = seg_result["roi_polygon"],
            road_coverage   = seg_result["coverage"],
            obstacle_data   = det_result,
            traffic_data    = det_result,
            path_data       = path_data,
            final_action    = final_action,
            avoid_action    = avoid_action,
            car_zone_bbox   = car_zone_bbox,
            danger_mask     = danger_mask,
            car_icon        = car_icon,
            inference_ms    = det_result["inference_time_ms"],
        )

        # Keep terminal output to one updating line.
        avg_fps = 1000.0 / (total_ms / frame_count)
        print(
            f"Frame {frame_count:>5} | "
            f"ACTION: {final_action:<12} | "
            f"Obstacle: {det_result['obstacle_action']:<6} "
            f"(risk {det_result['stable_risk']:.2f}) | "
            f"Traffic: {det_result['traffic_action']:<8} | "
            f"Path: {path_data['path_action']:<12} | "
            f"FPS: {avg_fps:.1f}",
            end="\r",
        )

        # Save and display.
        if writer:
            writer.write(annotated)

        if SHOW_DISPLAY:
            cv2.imshow("Self-Driving Pipeline", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nStopped by user.")
                break

    # Cleanup.
    pipeline.stop()
    cap.release()
    if writer:
        writer.release()
        print(f"\nSaved annotated video: {OUTPUT_VIDEO_PATH}")
    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_count} frames. "
          f"Average FPS: {1000.0 / (total_ms / max(frame_count, 1)):.1f}")


if __name__ == "__main__":
    main()
