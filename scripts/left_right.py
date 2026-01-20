#!/usr/bin/env python3
"""
Headless person detector that splits detections into LEFT vs RIGHT.

- Uses RealSense color stream
- Uses YOLOv8 pose for person detection + nose keypoint
- Logs once per second:
    - total people
    - left count / right count
    - per-person confidence

Run:
    python scripts/left_right.py

Stop:
    Ctrl+C
"""

import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO


@dataclass
class PersonDet:
    x1: float
    y1: float
    x2: float
    y2: float
    nose_x: float
    nose_y: float
    conf: float

    @property
    def area(self) -> float:
        return max(0.0, (self.x2 - self.x1)) * max(0.0, (self.y2 - self.y1))


class SplitPeopleLogger:
    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        rotate_90_clockwise: bool = True,
        conf_thresh: float = 0.25,
        log_period_s: float = 1.0,
    ):
        self.rotate_90_clockwise = rotate_90_clockwise
        self.conf_thresh = conf_thresh
        self.log_period_s = log_period_s

        print("Loading YOLO pose model...")
        self.model = YOLO(model_path)

        print("Initializing RealSense camera...")
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)

        # Align is not strictly necessary if we only use color, but harmless.
        self.align = rs.align(rs.stream.color)

        # Compute image geometry (after optional rotation)
        self.img_width = 480 if rotate_90_clockwise else 640
        self.img_height = 640 if rotate_90_clockwise else 480
        self.center_x = self.img_width / 2.0

        self._last_log_t = 0.0
        print("Initialization complete (headless). Logging once per second.\n")

    def _get_color_frame(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None

        img = np.asanyarray(color_frame.get_data())
        if self.rotate_90_clockwise:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return img

    def detect_people(self, frame: np.ndarray) -> List[PersonDet]:
        """
        Returns person detections with nose keypoint.
        Uses YOLO pose output:
          - result.boxes.xyxy, result.boxes.conf
          - result.keypoints.xy (COCO: 0 = nose)
        """
        results = self.model(frame, verbose=False)
        dets: List[PersonDet] = []

        for r in results:
            if r.keypoints is None or r.boxes is None:
                continue

            kpts = r.keypoints.xy.cpu().numpy()          # (N, K, 2)
            boxes = r.boxes.xyxy.cpu().numpy()           # (N, 4)
            confs = r.boxes.conf.cpu().numpy()           # (N,)

            for i in range(len(confs)):
                conf = float(confs[i])
                if conf < self.conf_thresh:
                    continue

                # Nose (keypoint 0)
                nose_x, nose_y = kpts[i][0]

                # Some frames may have missing/invalid keypoints
                if nose_x <= 0 or nose_y <= 0:
                    continue

                x1, y1, x2, y2 = boxes[i]
                dets.append(PersonDet(x1, y1, x2, y2, float(nose_x), float(nose_y), conf))

        return dets

    def split_left_right(self, dets: List[PersonDet]) -> Tuple[List[PersonDet], List[PersonDet]]:
        left, right = [], []
        for d in dets:
            # If nose is left of image center => LEFT side, else RIGHT
            if d.nose_x < self.center_x:
                left.append(d)
            else:
                right.append(d)
        return left, right

    def log_once_per_second(self, left: List[PersonDet], right: List[PersonDet]) -> None:
        now = time.time()
        if now - self._last_log_t < self.log_period_s:
            return
        self._last_log_t = now

        total = len(left) + len(right)

        # Sort detections by confidence (or area if you prefer)
        all_dets = sorted(left + right, key=lambda d: d.conf, reverse=True)

        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] total={total} left={len(left)} right={len(right)}")

        # Per-person logs (confidence + side + position info)
        for idx, d in enumerate(all_dets, start=1):
            side = "LEFT" if d.nose_x < self.center_x else "RIGHT"
            # Optional: include bbox area as a rough proxy for distance
            print(
                f"  - person#{idx:02d} side={side:<5} conf={d.conf:.2f} "
                f"nose=({d.nose_x:.0f},{d.nose_y:.0f}) area={d.area:.0f}"
            )

        if total == 0:
            print("  - (no people detected)")
        print("")  # blank line for readability

    def run(self):
        try:
            while True:
                frame = self._get_color_frame()
                if frame is None:
                    continue

                dets = self.detect_people(frame)
                left, right = self.split_left_right(dets)
                self.log_once_per_second(left, right)

                # Tight loop; logging is rate-limited
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nStopping (Ctrl+C).")

        finally:
            self.cleanup()

    def cleanup(self):
        print("Cleaning up...")
        try:
            self.pipeline.stop()
        except Exception:
            pass
        print("Cleanup complete.")


def main():
    SplitPeopleLogger().run()


if __name__ == "__main__":
    main()
