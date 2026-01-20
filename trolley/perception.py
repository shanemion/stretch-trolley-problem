"""
People counting module for trolley problem.

Supports:
- Mock mode: terminal input for development/testing
- Static camera mode: RealSense + YOLO with nose-based left/right split
- Scanning camera mode: Robot pans head left/right to expand FOV
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
    Counts people on left vs right side.
    
    Modes:
    - Mock: terminal input
    - Static camera: nose-based split
    - Scanning camera: head pans left/right, all detections in each view go to that side
    """
    
    # Head pan positions (radians) for scanning mode
    PAN_LEFT = 0.4      # ~23 degrees left
    PAN_RIGHT = -0.4    # ~23 degrees right
    PAN_CENTER = 0.0    # Center position
    SCAN_SETTLE_TIME = 0.3  # Time to wait after head movement
    
    def __init__(
        self,
        conf_thresh: float = 0.35,
        model_path: str = "scripts/yolov8n-pose.pt",
        rotate_90_clockwise: bool = True,
        use_mock: bool = True,
        use_scanning: bool = False,
    ):
        """
        Initialize people counter.
        
        Args:
            conf_thresh: Confidence threshold for YOLO detection
            model_path: Path to YOLO model
            rotate_90_clockwise: Camera rotation flag
            use_mock: If True, use terminal input
            use_scanning: If True, use head scanning (requires robot)
        """
        self.conf_thresh = conf_thresh
        self.model_path = model_path
        self.rotate_90_clockwise = rotate_90_clockwise
        self.use_mock = use_mock
        self.use_scanning = use_scanning
        
        # Camera-related attributes
        self.model = None
        self.pipeline = None
        self.align = None
        self.img_width = 0
        self.img_height = 0
        self.center_x = 0.0
        
        # Robot for scanning
        self.robot = None
        self.robot_initialized = False
        
        if not use_mock:
            self._init_camera()
            if use_scanning:
                self._init_robot()
        else:
            print("[PeopleCounter] Initialized in MOCK mode (terminal input)")
    
    def _init_camera(self):
        """Initialize RealSense camera and YOLO model."""
        import numpy as np
        import cv2
        import pyrealsense2 as rs
        from ultralytics import YOLO
        
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
        self.align = rs.align(rs.stream.color)
        
        if self.rotate_90_clockwise:
            self.img_width = 480
            self.img_height = 640
        else:
            self.img_width = 640
            self.img_height = 480
        self.center_x = self.img_width / 2.0
        
        mode = "SCANNING" if self.use_scanning else "STATIC"
        print(f"[PeopleCounter] Camera initialized ({mode} mode)")
    
    def _init_robot(self):
        """Initialize robot for head scanning."""
        try:
            print("[PeopleCounter] Initializing robot for head scanning...")
            import stretch_body.robot as robot_module
            
            self.robot = robot_module.Robot()
            did_startup = self.robot.startup()
            if not did_startup:
                print("[PeopleCounter] Warning: Robot startup returned False")
            
            self.robot_initialized = True
            self._move_head_to(self.PAN_CENTER)
            print("[PeopleCounter] Robot ready for head scanning!")
            
        except Exception as e:
            self.robot_initialized = False
            print(f"[PeopleCounter] Robot init failed: {e}")
            print("[PeopleCounter] Falling back to static camera mode")
    
    def _move_head_to(self, pan_position: float):
        """Move head to specified pan position."""
        if not self.robot_initialized or self.robot is None:
            return
        try:
            self.robot.head.move_to('head_pan', pan_position)
            self.robot.push_command()
        except Exception as e:
            print(f"[PeopleCounter] Head movement error: {e}")
    
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
    
    def _detect_people(self, frame, side: str = "") -> List[PersonDet]:
        """Run YOLO pose detection on frame."""
        results = self.model(frame, verbose=False)
        dets: List[PersonDet] = []
        
        for r in results:
            if r.keypoints is None or r.boxes is None:
                continue
            
            kpts = r.keypoints.xy.cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            
            for i in range(len(confs)):
                conf = float(confs[i])
                if conf < self.conf_thresh:
                    continue
                
                nose_x, nose_y = kpts[i][0]
                if nose_x <= 0 or nose_y <= 0:
                    continue
                
                x1, y1, x2, y2 = boxes[i]
                
                # If side is specified (scanning mode), use it
                # Otherwise, determine by nose position (static mode)
                if side:
                    det_side = side
                else:
                    det_side = "LEFT" if nose_x < self.center_x else "RIGHT"
                
                dets.append(PersonDet(
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    nose_x=float(nose_x), nose_y=float(nose_y),
                    conf=conf,
                    side=det_side
                ))
        
        return dets
    
    def get_counts(self) -> Tuple[int, int, Dict]:
        """
        Get current left and right person counts.
        
        Returns:
            Tuple of (left_count, right_count, metadata_dict)
        """
        if self.use_mock:
            return self._get_counts_mock()
        elif self.use_scanning and self.robot_initialized:
            return self._get_counts_scanning()
        else:
            return self._get_counts_static()
    
    def _get_counts_scanning(self) -> Tuple[int, int, Dict]:
        """Get counts using head scanning (pans left, then right)."""
        left_dets = []
        right_dets = []
        
        # Look LEFT
        self._move_head_to(self.PAN_LEFT)
        time.sleep(self.SCAN_SETTLE_TIME)
        
        for _ in range(3):  # Multiple captures for robustness
            frame = self._get_color_frame()
            if frame is not None:
                dets = self._detect_people(frame, side="LEFT")
                if len(dets) > len(left_dets):
                    left_dets = dets
            time.sleep(0.1)
        
        # Look RIGHT
        self._move_head_to(self.PAN_RIGHT)
        time.sleep(self.SCAN_SETTLE_TIME)
        
        for _ in range(3):
            frame = self._get_color_frame()
            if frame is not None:
                dets = self._detect_people(frame, side="RIGHT")
                if len(dets) > len(right_dets):
                    right_dets = dets
            time.sleep(0.1)
        
        # Return to center
        self._move_head_to(self.PAN_CENTER)
        
        left_count = len(left_dets)
        right_count = len(right_dets)
        left_confs = [d.conf for d in left_dets]
        right_confs = [d.conf for d in right_dets]
        
        meta = {
            "timestamp": time.time(),
            "total_detections": left_count + right_count,
            "confidences": left_confs + right_confs,
            "left_confidences": left_confs,
            "right_confidences": right_confs,
            "detections": left_dets + right_dets,
            "sample_valid": True,
            "source": "scanning_camera",
        }
        
        return left_count, right_count, meta
    
    def _get_counts_static(self) -> Tuple[int, int, Dict]:
        """Get counts from static camera (nose-based split)."""
        frame = self._get_color_frame()
        
        if frame is None:
            return 0, 0, {
                "timestamp": time.time(),
                "total_detections": 0,
                "confidences": [],
                "left_confidences": [],
                "right_confidences": [],
                "detections": [],
                "sample_valid": False,
                "source": "static_camera",
            }
        
        dets = self._detect_people(frame)
        
        left_dets = [d for d in dets if d.side == "LEFT"]
        right_dets = [d for d in dets if d.side == "RIGHT"]
        
        left_count = len(left_dets)
        right_count = len(right_dets)
        left_confs = [d.conf for d in left_dets]
        right_confs = [d.conf for d in right_dets]
        
        meta = {
            "timestamp": time.time(),
            "total_detections": len(dets),
            "confidences": [d.conf for d in dets],
            "left_confidences": left_confs,
            "right_confidences": right_confs,
            "detections": dets,
            "sample_valid": True,
            "source": "static_camera",
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
        """Clean up resources."""
        if self.robot_initialized and self.robot is not None:
            try:
                self._move_head_to(self.PAN_CENTER)
                time.sleep(0.3)
                self.robot.stop()
            except Exception:
                pass
        
        if not self.use_mock and self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        
        print("[PeopleCounter] Cleanup complete")
