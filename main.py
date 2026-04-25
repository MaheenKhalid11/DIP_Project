"""
main.py
───────
Top-level entry point. Ties all four modules together and runs the
video loop with threaded parallel processing.

Architecture
────────────
Two background threads run concurrently on every frame:

    Thread A: segmentation  → road mask  (CPU-heavy, ~80-150 ms)
    Thread B: YOLO detection → obstacles + traffic  (GPU/CPU, ~20-60 ms)

The main thread reads frames and feeds both threads, then collects their
results, calls decision.py, annotates, and displays.

Why threading?
──────────────
Segmentation (K-means) and YOLO inference are independent — neither needs
the other's output to run.  Running them sequentially wastes time.
With threading they overlap: while YOLO is inferring, K-means is running
on the same frame.  Total latency approaches max(seg, yolo) instead of
seg + yolo.

Note: Python's GIL means CPU threads don't run truly in parallel for
pure-Python code, but OpenCV and NumPy operations release the GIL, so
real speedup is achieved for these workloads.

Usage
─────
Just set VIDEO_PATH below and run:
    python main.py
"""

import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from segmentation    import color_segmentation
from yolo_detection  import YOLODetector, estimate_risk
from decision        import get_path_guidance, decide_final_action


# ══════════════════════════════════════════════════════════════
# ▶  CONFIGURATION — change these to match your setup
# ══════════════════════════════════════════════════════════════

VIDEO_PATH          = "your_video.mp4"        # ← set your video path here
COCO_WEIGHTS        = "yolov8n.pt"
BARRIER_WEIGHTS     = "boom_barrier_best.pt"
CAR_ICON_PATH       = "car.png"               # set to None if you have no icon
OUTPUT_VIDEO_PATH   = "outputs/result.mp4"    # set to None to skip saving
SHOW_DISPLAY        = True                    # set False for headless runs
FRAME_WIDTH         = 640
FRAME_HEIGHT        = 360

# ══════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────
# CAR ICON + ZONE HELPERS
# ──────────────────────────────────────────────────────────────

def _load_car_icon(path):
    """Loads the car icon with alpha channel if available, else None."""
    if path is None:
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return img   # may be None if file missing — handled gracefully


def get_car_zone_bbox(frame_w, frame_h):
    """
    Returns the bounding box [x1,y1,x2,y2] of the car icon as drawn on screen.
    Kept consistent with overlay_car_icon so risk logic matches the visual.
    """
    target_w = max(138, int(0.40 * frame_w))
    target_h = max(84,  int(0.28 * frame_h))
    x1 = (frame_w - target_w) // 2
    y1 = frame_h - target_h - int(0.03 * frame_h)
    return [x1, y1, x1 + target_w, y1 + target_h]


def get_car_zone_masks(frame_w, frame_h, car_icon):
    """
    Builds two binary masks:
      car_mask    — exact footprint of the car icon
      danger_mask — dilated version representing the safety clearance zone

    Why two masks?
    car_mask catches objects that are literally on the car (risk score = max).
    danger_mask catches objects in the safety buffer around the car so we can
    react before physical contact.
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
    """Blends the car icon onto the frame using its alpha channel."""
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


# ──────────────────────────────────────────────────────────────
# ROAD MASK TEMPORAL SMOOTHER
# ──────────────────────────────────────────────────────────────

class RoadMaskSmoother:
    """
    Exponential moving average on the road mask across frames.

    Raw K-means output can flicker because cluster assignments shift
    slightly frame to frame.  EMA blends the current mask with recent
    history so the green overlay on screen is stable.

    Two alpha values:
      fast_alpha — used when the new mask largely agrees with history (high IoU)
      slow_alpha — used when the mask changes a lot (camera moved, new scene)
    This adaptive rate prevents the smoother from being too sluggish on
    genuine scene changes while still suppressing noise on straight roads.
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


# ──────────────────────────────────────────────────────────────
# ANNOTATION
# ──────────────────────────────────────────────────────────────

def annotate(frame_small, coco_results, barrier_results, barrier_dets,
             road_mask, roi_polygon, road_coverage,
             obstacle_data, traffic_data,
             path_data, final_action, avoid_action,
             car_zone_bbox, danger_mask,
             car_icon, inference_ms):
    """
    Draws all debug and decision information onto the frame.

    Layers (bottom → top):
      1. YOLO bounding boxes (from coco_results.plot())
      2. Boom barrier boxes (orange, from barrier model)
      3. Semi-transparent green road overlay
      4. Road boundary contour
      5. Car icon (replaces the raw danger-zone rectangle)
      6. Danger-zone safety contour
      7. Per-obstacle risk labels
      8. Traffic signal labels
      9. Path offset info
     10. Final action banner
    """
    # 1. Start with YOLO's own box drawing
    annotated = coco_results[0].plot()

    # 2. Boom barrier boxes in orange
    for b in barrier_dets:
        x1, y1, x2, y2 = b["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 80, 255), 2)
        cv2.putText(annotated, f"boom_barrier {b['confidence']:.2f}",
                    (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 80, 255), 2)

    # 3. Semi-transparent green road overlay
    road_layer = np.zeros_like(annotated)
    road_layer[:, :, 1] = road_mask
    annotated = cv2.addWeighted(annotated, 1.0, road_layer, 0.30, 0)

    # 4. Road boundary contour
    cv2.polylines(annotated, [roi_polygon], isClosed=True,
                  color=(120, 255, 120), thickness=1)
    cv2.putText(annotated, f"Road: {road_coverage:.2f}",
                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 120), 2)

    # 5. Car icon
    h, w = annotated.shape[:2]
    annotated = overlay_car_icon(annotated, car_icon, w, h)

    # 6. Danger-zone safety contour
    contours, _ = cv2.findContours(danger_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(annotated, [c], -1, (0, 180, 255), 1)
        x, y, _, _ = cv2.boundingRect(c)
        cv2.putText(annotated, "safety zone", (x, max(12, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 180, 255), 1)

    # 7. Per-obstacle risk labels
    for obs in obstacle_data["obstacles"]:
        x1, y1 = obs["bbox"][0], obs["bbox"][1]
        label   = f"{obs['class']} | {obs['proximity']} | risk:{obs['risk_score']}"
        color   = (0, 0, 255) if obs["in_path"] else (0, 200, 200)
        cv2.putText(annotated, label, (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # 8. Traffic signal labels
    for tdet in traffic_data["traffic_detections"]:
        x1, y1     = tdet["bbox"][0], tdet["bbox"][1]
        label_text = f"{tdet['state']} → {tdet['action']}"
        if tdet["detail"]:
            label_text += f" ({tdet['detail']})"
        text_color = {"STOP": (0, 0, 255), "SLOW": (0, 165, 255),
                      "GO": (0, 255, 0), "CAUTION": (0, 255, 255)}.get(
                      tdet["action"], (255, 255, 255))
        cv2.putText(annotated, label_text, (x1, max(y1 - 25, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    # 9. Path offset info
    cv2.putText(
        annotated,
        f"path: {path_data['path_action']}  offset:{path_data['path_offset']:+.2f}",
        (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 255, 180), 2,
    )

    # 10. Final action banner (top-left, black background for readability)
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

    # FPS counter (top-right)
    fps_text = f"FPS: {1000/max(inference_ms,1):.1f}"
    cv2.putText(annotated, fps_text,
                (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    return annotated


def mask_to_polygon(mask, fallback_w, fallback_h):
    """Converts the largest contour of a binary mask to a polygon array."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback: static trapezoid
        w, h = fallback_w, fallback_h
        return np.array([[int(0.04*w), h-1], [int(0.96*w), h-1],
                          [int(0.62*w), int(0.5*h)], [int(0.38*w), int(0.5*h)]])
    largest = max(contours, key=cv2.contourArea)
    hull    = cv2.convexHull(largest)
    epsilon = 0.015 * cv2.arcLength(hull, True)
    approx  = cv2.approxPolyDP(hull, epsilon, True)
    return approx.reshape(-1, 2)


# ──────────────────────────────────────────────────────────────
# THREADED PIPELINE
# ──────────────────────────────────────────────────────────────

class FramePipeline:
    """
    Runs segmentation and YOLO detection in parallel threads.

    How it works
    ────────────
    Both threads share a common input frame (set via submit()).
    Each thread has its own result slot and a threading.Event to signal
    when the result is ready.

    The main thread calls submit(frame) to kick both threads off,
    then calls get_results() which blocks until both are done.

    Why not multiprocessing?
    Threads are cheaper to create and share memory.  Since OpenCV and
    NumPy release the GIL during computation, real parallel execution
    happens for the heavy parts (K-means, YOLO forward pass).
    """

    def __init__(self, detector: YOLODetector,
                 car_zone_mask, danger_zone_mask,
                 road_smoother: RoadMaskSmoother):
        self.detector        = detector
        self.car_zone_mask   = car_zone_mask
        self.danger_zone_mask = danger_zone_mask
        self.road_smoother   = road_smoother

        # Shared frame — written by main thread, read by workers
        self._frame       = None
        self._frame_lock  = threading.Lock()

        # Results
        self._seg_result  = None
        self._det_result  = None

        # Events: main sets _start, workers set _seg_done / _det_done
        self._start_event = threading.Event()
        self._seg_done    = threading.Event()
        self._det_done    = threading.Event()
        self._stop_flag   = False

        # Start worker threads (daemon=True so they die when main exits)
        self._seg_thread = threading.Thread(target=self._seg_worker, daemon=True)
        self._det_thread = threading.Thread(target=self._det_worker, daemon=True)
        self._seg_thread.start()
        self._det_thread.start()

    def _seg_worker(self):
        """
        Segmentation worker thread.
        Waits for a frame, runs color_segmentation, signals done.
        """
        while not self._stop_flag:
            # Wait for the main thread to provide a new frame
            self._start_event.wait(timeout=1.0)
            if self._stop_flag:
                break

            with self._frame_lock:
                frame = self._frame

            if frame is None:
                continue

            # Run K-means road segmentation
            seg     = color_segmentation(frame, clusters=4, spatial_weight=0.25)
            raw     = seg["road_mask"]
            # Apply temporal smoothing to reduce flicker
            smooth  = self.road_smoother.update(raw)
            polygon = mask_to_polygon(smooth, frame.shape[1], frame.shape[0])
            coverage = round(float(np.sum(smooth > 0)) / float(frame.shape[0] * frame.shape[1]), 3)

            self._seg_result = {
                "mask":        smooth,
                "roi_polygon": polygon,
                "coverage":    coverage,
            }
            self._seg_done.set()

    def _det_worker(self):
        """
        YOLO detection worker thread.
        Waits for a frame, runs both YOLO models, signals done.
        """
        while not self._stop_flag:
            self._start_event.wait(timeout=1.0)
            if self._stop_flag:
                break

            with self._frame_lock:
                frame = self._frame

            if frame is None:
                continue

            # Run YOLO inference (both COCO and barrier models)
            self._det_result = self.detector.process_frame(
                frame, self.car_zone_mask, self.danger_zone_mask
            )
            self._det_done.set()

    def submit(self, frame: np.ndarray):
        """
        Sends a new frame to both worker threads.
        Clears the done-events so get_results() waits for fresh output.
        """
        self._seg_done.clear()
        self._det_done.clear()
        with self._frame_lock:
            self._frame = frame
        # Wake both workers simultaneously
        self._start_event.set()
        self._start_event.clear()

    def get_results(self, timeout: float = 5.0):
        """
        Blocks until both threads have finished processing the current frame.
        Returns (seg_result, det_result) or (None, None) on timeout.
        """
        seg_ok = self._seg_done.wait(timeout=timeout)
        det_ok = self._det_done.wait(timeout=timeout)
        if not (seg_ok and det_ok):
            return None, None
        return self._seg_result, self._det_result

    def stop(self):
        """Signals worker threads to exit cleanly."""
        self._stop_flag = True
        self._start_event.set()   # unblock any waiting thread


# ──────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────

def main():
    # ── Load assets ──────────────────────────────────────────
    car_icon = _load_car_icon(CAR_ICON_PATH)
    detector = YOLODetector(COCO_WEIGHTS, BARRIER_WEIGHTS)

    # ── Open video ───────────────────────────────────────────
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: could not open video at '{VIDEO_PATH}'")
        print("Check that the path is correct and the file exists.")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 25.0

    # ── Video writer (optional) ──────────────────────────────
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

    # ── Build per-frame constants (car zone, masks) ──────────
    car_zone_bbox  = get_car_zone_bbox(FRAME_WIDTH, FRAME_HEIGHT)
    car_mask, danger_mask = get_car_zone_masks(FRAME_WIDTH, FRAME_HEIGHT, car_icon)
    road_smoother  = RoadMaskSmoother()

    # ── Start threaded pipeline ──────────────────────────────
    pipeline = FramePipeline(detector, car_mask, danger_mask, road_smoother)

    print("Running — press Q to quit\n")
    frame_count  = 0
    total_ms     = 0.0

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("\nVideo finished.")
            break

        # Resize to working resolution
        frame_small = cv2.resize(raw_frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Submit to both worker threads (they run in parallel from here)
        t_frame_start = time.time()
        pipeline.submit(frame_small)

        # While threads are running, the main thread is free.
        # For now we just wait — in a real robot we'd do other work here.
        seg_result, det_result = pipeline.get_results(timeout=5.0)

        if seg_result is None or det_result is None:
            # Timeout — skip this frame
            print("\nWarning: worker thread timed out, skipping frame.")
            continue

        frame_ms = (time.time() - t_frame_start) * 1000
        total_ms += frame_ms
        frame_count += 1

        # ── Decision ─────────────────────────────────────────
        path_data = get_path_guidance(seg_result["mask"], car_zone_bbox)

        final_action, avoid_action, primary_risky = decide_final_action(
            path_data       = path_data,
            obstacle_data   = det_result,
            traffic_data    = det_result,
            road_mask       = seg_result["mask"],
            car_zone_bbox   = car_zone_bbox,
        )

        # ── Annotate ─────────────────────────────────────────
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

        # ── Terminal output (single line, overwrites itself) ──
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

        # ── Save + Display ────────────────────────────────────
        if writer:
            writer.write(annotated)

        if SHOW_DISPLAY:
            cv2.imshow("Self-Driving Pipeline", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nStopped by user.")
                break

    # ── Cleanup ───────────────────────────────────────────────
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
