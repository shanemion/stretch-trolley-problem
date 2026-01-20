"""
People counting module for trolley problem.

Supports both mock (terminal input) and real camera (RealSense + YOLO) modes.
"""

import time
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

# Camera imports are lazy-loaded to allow mock mode without dependencies


@dataclass
class PersonDet:
    """Person detection with position and confidence."""
    x1: float
    y1: float
    x2: float
    y2: float
    nose_x: float
    nose_y: float
    conf: float
    side: str = ""  # "LEFT" or "RIGHT"

    @property
    def area(self) -> float:
        return max(0.0, (self.x2 - self.x1)) * max(0.0, (self.y2 - self.y1))


class PeopleCounter:
    """
    Counts people on left vs right side of camera view.
    
    Supports two modes:
    - Mock mode: terminal input for development/testing
    - Camera mode: RealSense + YOLO pose detection
    """
    
    def __init__(
        self,
        conf_thresh: float = 0.35,
        model_path: str = "scripts/yolov8n-pose.pt",
        rotate_90_clockwise: bool = True,
        use_mock: bool = True,
    ):
        """
        Initialize people counter.
        
        Args:
            conf_thresh: Confidence threshold for YOLO detection
            model_path: Path to YOLO model
            rotate_90_clockwise: Camera rotation flag
            use_mock: If True, use terminal input; if False, use camera
        """
        self.conf_thresh = conf_thresh
        self.model_path = model_path
        self.rotate_90_clockwise = rotate_90_clockwise
        self.use_mock = use_mock
        
        # Camera-related attributes (initialized if not mock)
        self.model = None
        self.pipeline = None
        self.align = None
        self.img_width = 0
        self.img_height = 0
        self.center_x = 0.0
        
        if not use_mock:
            self._init_camera()
        else:
            print("[PeopleCounter] Initialized in MOCK mode (terminal input)")
    
    def _init_camera(self):
        """Initialize RealSense camera and YOLO model."""
        import numpy as np
        import cv2
        import pyrealsense2 as rs
        from ultralytics import YOLO
        
        # Store imports for later use
        self._np = np
        self._cv2 = cv2
        self._rs = rs
        
        print("[PeopleCounter] Loading YOLO pose model...")
        self.model = YOLO(self.model_path)
        
        print("[PeopleCounter] Initializing RealSense camera...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        
        # Align to color stream
        self.align = rs.align(rs.stream.color)
        
        # Compute image geometry (after optional rotation)
        if self.rotate_90_clockwise:
            self.img_width = 480
            self.img_height = 640
        else:
            self.img_width = 640
            self.img_height = 480
        self.center_x = self.img_width / 2.0
        
        print(f"[PeopleCounter] Camera initialized (center_x={self.center_x:.1f})")
        print(f"[PeopleCounter] Confidence threshold: {self.conf_thresh}")
        print("[PeopleCounter] Ready for camera mode.\n")
    
    def _get_color_frame(self):
        """Get current color frame from RealSense."""
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        
        img = self._np.asanyarray(color_frame.get_data())
        if self.rotate_90_clockwise:
            img = self._cv2.rotate(img, self._cv2.ROTATE_90_CLOCKWISE)
        return img
    
    def _detect_people(self, frame) -> List[PersonDet]:
        """
        Run YOLO pose detection on frame.
        
        Returns list of PersonDet with confidence and position.
        """
        results = self.model(frame, verbose=False)
        dets: List[PersonDet] = []
        
        for r in results:
            if r.keypoints is None or r.boxes is None:
                continue
            
            kpts = r.keypoints.xy.cpu().numpy()  # (N, K, 2)
            boxes = r.boxes.xyxy.cpu().numpy()   # (N, 4)
            confs = r.boxes.conf.cpu().numpy()   # (N,)
            
            for i in range(len(confs)):
                conf = float(confs[i])
                if conf < self.conf_thresh:
                    continue
                
                # Nose (keypoint 0)
                nose_x, nose_y = kpts[i][0]
                
                # Skip invalid keypoints
                if nose_x <= 0 or nose_y <= 0:
                    continue
                
                x1, y1, x2, y2 = boxes[i]
                
                # Determine side based on nose position
                side = "LEFT" if nose_x < self.center_x else "RIGHT"
                
                dets.append(PersonDet(
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    nose_x=float(nose_x), nose_y=float(nose_y),
                    conf=conf,
                    side=side
                ))
        
        return dets
    
    def get_counts(self) -> Tuple[int, int, Dict]:
        """
        Get current left and right person counts.
        
        Returns:
            Tuple of (left_count, right_count, metadata_dict)
            
        Metadata dict contains:
            - timestamp: float (current time)
            - total_detections: int
            - confidences: list of floats (per-person confidence)
            - left_confidences: list of floats
            - right_confidences: list of floats
            - detections: list of PersonDet objects (for detailed logging)
            - sample_valid: bool
            - source: str ("camera" or "terminal_input")
        """
        if self.use_mock:
            return self._get_counts_mock()
        else:
            return self._get_counts_camera()
    
    def _get_counts_camera(self) -> Tuple[int, int, Dict]:
        """Get counts from camera using YOLO detection."""
        frame = self._get_color_frame()
        
        if frame is None:
            # Invalid frame
            return 0, 0, {
                "timestamp": time.time(),
                "total_detections": 0,
                "confidences": [],
                "left_confidences": [],
                "right_confidences": [],
                "detections": [],
                "sample_valid": False,
                "source": "camera",
            }
        
        # Run detection
        dets = self._detect_people(frame)
        
        # Split by side
        left_dets = [d for d in dets if d.side == "LEFT"]
        right_dets = [d for d in dets if d.side == "RIGHT"]
        
        left_count = len(left_dets)
        right_count = len(right_dets)
        
        # Extract confidences
        all_confs = [d.conf for d in dets]
        left_confs = [d.conf for d in left_dets]
        right_confs = [d.conf for d in right_dets]
        
        meta = {
            "timestamp": time.time(),
            "total_detections": len(dets),
            "confidences": all_confs,
            "left_confidences": left_confs,
            "right_confidences": right_confs,
            "detections": dets,
            "sample_valid": True,
            "source": "camera",
        }
        
        return left_count, right_count, meta
    
    def _get_counts_mock(self) -> Tuple[int, int, Dict]:
        """Get counts via terminal input."""
        while True:
            try:
                left_input = input("Enter number of people on LEFT (int): ").strip()
                left_count = int(left_input)
                if left_count < 0:
                    print("  Error: count must be >= 0. Retrying...")
                    continue
                break
            except ValueError:
                print("  Error: invalid input. Please enter an integer. Retrying...")
        
        while True:
            try:
                right_input = input("Enter number of people on RIGHT (int): ").strip()
                right_count = int(right_input)
                if right_count < 0:
                    print("  Error: count must be >= 0. Retrying...")
                    continue
                break
            except ValueError:
                print("  Error: invalid input. Please enter an integer. Retrying...")
        
        meta = {
            "timestamp": time.time(),
            "total_detections": left_count + right_count,
            "confidences": [],
            "left_confidences": [],
            "right_confidences": [],
            "detections": [],
            "sample_valid": True,
            "source": "terminal_input",
        }
        
        return left_count, right_count, meta
    
    def cleanup(self) -> None:
        """Clean up resources (camera, model, etc.)."""
        if not self.use_mock and self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        print("[PeopleCounter] Cleanup complete")
