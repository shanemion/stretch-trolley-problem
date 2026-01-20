"""
GUI visualization for the trolley problem system with head scanning.

The robot pans its head left and right to expand the field of view:
- Looking LEFT: captures people on the left track
- Looking RIGHT: captures people on the right track

Displays:
- Top Left: Confidence view (left/right aggregate confidence scores)
- Bottom Left: Camera feed showing current view + scan direction
- Right Side: Reserved for future use

Usage:
    python -m trolley.gui
"""

import time
import numpy as np
import cv2
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Detection:
    """Person detection with position and confidence."""
    x1: float
    y1: float
    x2: float
    y2: float
    nose_x: float
    nose_y: float
    conf: float
    side: str  # "LEFT" or "RIGHT" (based on which direction camera is looking)


class TrolleyGUI:
    """
    GUI visualization for trolley problem detection with head scanning.
    
    The robot pans left and right to expand FOV:
    - When looking LEFT: all detections count as LEFT
    - When looking RIGHT: all detections count as RIGHT
    
    Layout:
    +-------------------+-------------------+
    |  CONFIDENCE VIEW  |                   |
    |  LEFT  |  RIGHT   |    (Reserved)     |
    +--------+----------+                   |
    |   CAMERA FEED     |                   |
    |   [Scanning...]   |                   |
    +-------------------+-------------------+
    """
    
    # Color scheme (BGR format)
    BG_COLOR = (30, 30, 30)        # Dark gray background
    PANEL_BG = (45, 45, 45)        # Slightly lighter panel background
    BORDER_COLOR = (100, 100, 100) # Panel borders
    TEXT_COLOR = (255, 255, 255)   # White text
    LEFT_COLOR = (0, 165, 255)     # Orange for left
    RIGHT_COLOR = (255, 100, 100)  # Blue for right
    DIVIDER_COLOR = (0, 255, 255)  # Yellow divider
    SUCCESS_COLOR = (0, 255, 0)    # Green
    ERROR_COLOR = (0, 0, 255)      # Red
    
    # Head pan positions (radians)
    PAN_LEFT = 0.4      # ~23 degrees left
    PAN_RIGHT = -0.4    # ~23 degrees right
    PAN_CENTER = 0.0    # Center position
    
    # Scanning parameters
    SCAN_SETTLE_TIME = 0.5   # Time to wait after head movement (seconds)
    SCAN_HOLD_TIME = 5.0     # Time to hold each position before switching (seconds)
    
    def __init__(
        self,
        window_width: int = 1280,
        window_height: int = 720,
        model_path: str = "scripts/yolov8n-pose.pt",
        conf_thresh: float = 0.35,
        rotate_90_clockwise: bool = True,
        use_robot: bool = True,
    ):
        """
        Initialize the GUI with head scanning.
        
        Args:
            window_width: Total window width
            window_height: Total window height
            model_path: Path to YOLO model
            conf_thresh: Confidence threshold for detection
            rotate_90_clockwise: Whether to rotate camera feed
            use_robot: Whether to use robot for head movement
        """
        self.window_width = window_width
        self.window_height = window_height
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.rotate_90_clockwise = rotate_90_clockwise
        self.use_robot = use_robot
        
        # Layout dimensions
        self.left_panel_width = window_width // 2
        self.right_panel_width = window_width - self.left_panel_width
        self.confidence_height = window_height // 3
        self.camera_height = window_height - self.confidence_height
        
        # Camera state
        self.camera_initialized = False
        self.camera_error = False
        self.camera_error_msg = ""
        self.pipeline = None
        self.align = None
        self.model = None
        
        # Robot state
        self.robot = None
        self.robot_initialized = False
        
        # Scanning state machine
        self.current_scan_side = "LEFT"  # Which side we're currently looking at
        self.scan_state = "INIT"  # INIT, LOOKING_LEFT, LOOKING_RIGHT
        self.last_switch_time = 0  # When we last switched sides
        self.head_moved = False  # Whether head has moved to current position
        
        # Detection state (accumulated from both scans)
        self.left_detections: List[Detection] = []
        self.right_detections: List[Detection] = []
        self.left_count = 0
        self.right_count = 0
        self.left_confidences: List[float] = []
        self.right_confidences: List[float] = []
        
        # Current frame state
        self._current_frame = None
        self._left_frame = None   # Last frame captured looking left
        self._right_frame = None  # Last frame captured looking right
        
        # Camera geometry (after rotation)
        if rotate_90_clockwise:
            self.cam_width = 480
            self.cam_height = 640
        else:
            self.cam_width = 640
            self.cam_height = 480
        
        # Initialize components
        self._init_camera()
        if use_robot:
            self._init_robot()
    
    def _init_camera(self):
        """Initialize RealSense camera and YOLO model."""
        import traceback
        
        try:
            print("[GUI] Importing pyrealsense2...")
            import pyrealsense2 as rs
            self._rs = rs
            
            print("[GUI] Importing YOLO...")
            from ultralytics import YOLO
            
            print(f"[GUI] Loading YOLO pose model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            print("[GUI] YOLO model loaded!")
            
            print("[GUI] Creating RealSense pipeline...")
            self.pipeline = rs.pipeline()
            
            print("[GUI] Configuring RealSense stream...")
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            print("[GUI] Starting RealSense pipeline...")
            self.pipeline.start(config)
            
            print("[GUI] Creating alignment...")
            self.align = rs.align(rs.stream.color)
            
            self.camera_initialized = True
            self.camera_error = False
            print("[GUI] Camera initialized successfully!")
            
        except Exception as e:
            self.camera_initialized = False
            self.camera_error = True
            self.camera_error_msg = str(e)
            print(f"[GUI] Camera initialization failed: {e}")
            print("[GUI] Full traceback:")
            traceback.print_exc()
    
    def _init_robot(self):
        """Initialize Stretch robot for head control."""
        try:
            print("[GUI] Importing stretch_body...")
            import stretch_body.robot as robot_module
            
            print("[GUI] Creating robot instance...")
            self.robot = robot_module.Robot()
            
            print("[GUI] Starting robot...")
            did_startup = self.robot.startup()
            if not did_startup:
                print("[GUI] Warning: Robot startup returned False")
            
            self.robot_initialized = True
            print("[GUI] Robot initialized for head control!")
            
            # Move head to center position
            self._move_head_to(self.PAN_CENTER)
            
        except Exception as e:
            self.robot_initialized = False
            print(f"[GUI] Robot initialization failed: {e}")
            print("[GUI] Will run without head scanning.")
    
    def _move_head_to(self, pan_position: float):
        """Move head to specified pan position."""
        if not self.robot_initialized or self.robot is None:
            return
        
        try:
            self.robot.head.move_to('head_pan', pan_position)
            self.robot.push_command()
        except Exception as e:
            print(f"[GUI] Head movement error: {e}")
    
    def _get_frame(self) -> Optional[np.ndarray]:
        """Get current frame from camera."""
        if not self.camera_initialized or self.pipeline is None:
            return None
        
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            
            if not color_frame:
                return None
            
            img = np.asanyarray(color_frame.get_data())
            
            if self.rotate_90_clockwise:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            
            self.camera_error = False
            return img
            
        except Exception as e:
            self.camera_error = True
            self.camera_error_msg = str(e)
            print(f"[GUI] Frame capture error: {e}")
            return None
    
    def _detect_people(self, frame: np.ndarray, side: str) -> List[Detection]:
        """
        Run YOLO detection on frame.
        All detections are assigned to the specified side (based on head direction).
        """
        if self.model is None:
            return []
        
        results = self.model(frame, verbose=False)
        detections: List[Detection] = []
        
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
                
                # All detections assigned to the side we're looking at
                detections.append(Detection(
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    nose_x=float(nose_x), nose_y=float(nose_y),
                    conf=conf,
                    side=side  # Based on head direction, not nose position
                ))
        
        return detections
    
    def _update_scan_state(self):
        """
        Update scanning state machine (non-blocking).
        Switches between LEFT and RIGHT every SCAN_HOLD_TIME seconds.
        """
        current_time = time.time()
        
        if not self.robot_initialized:
            # No robot - just capture center and split by nose position
            frame = self._get_frame()
            if frame is not None:
                self._current_frame = frame
                self._left_frame = frame.copy()
                self._right_frame = frame.copy()
                all_dets = self._detect_people_split(frame)
                self.left_detections = [d for d in all_dets if d.side == "LEFT"]
                self.right_detections = [d for d in all_dets if d.side == "RIGHT"]
            return
        
        # State machine for scanning
        if self.scan_state == "INIT":
            # Initialize - move to left position
            self.current_scan_side = "LEFT"
            self._move_head_to(self.PAN_LEFT)
            self.last_switch_time = current_time
            self.head_moved = True
            self.scan_state = "LOOKING_LEFT"
            return
        
        time_in_position = current_time - self.last_switch_time
        
        # Check if we need to switch sides
        if time_in_position >= self.SCAN_HOLD_TIME:
            if self.current_scan_side == "LEFT":
                # Switch to RIGHT
                self.current_scan_side = "RIGHT"
                self._move_head_to(self.PAN_RIGHT)
                self.scan_state = "LOOKING_RIGHT"
            else:
                # Switch to LEFT
                self.current_scan_side = "LEFT"
                self._move_head_to(self.PAN_LEFT)
                self.scan_state = "LOOKING_LEFT"
            
            self.last_switch_time = current_time
            self.head_moved = True
            return
        
        # Wait for head to settle after movement
        if self.head_moved and time_in_position < self.SCAN_SETTLE_TIME:
            return  # Still settling
        
        self.head_moved = False
        
        # Capture frame and detect for current side
        frame = self._get_frame()
        if frame is not None:
            self._current_frame = frame
            dets = self._detect_people(frame, self.current_scan_side)
            
            if self.current_scan_side == "LEFT":
                self._left_frame = frame.copy()
                # Keep best detection (more people = better)
                if len(dets) >= len(self.left_detections):
                    self.left_detections = dets
            else:
                self._right_frame = frame.copy()
                if len(dets) >= len(self.right_detections):
                    self.right_detections = dets
    
    def _detect_people_split(self, frame: np.ndarray) -> List[Detection]:
        """
        Fallback: detect people and split by nose position (no robot).
        """
        if self.model is None:
            return []
        
        results = self.model(frame, verbose=False)
        detections: List[Detection] = []
        center_x = self.cam_width / 2.0
        
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
                side = "LEFT" if nose_x < center_x else "RIGHT"
                
                detections.append(Detection(
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    nose_x=float(nose_x), nose_y=float(nose_y),
                    conf=conf,
                    side=side
                ))
        
        return detections
    
    def _draw_confidence_panel(self, canvas: np.ndarray):
        """Draw the confidence view panel (top left)."""
        x1, y1 = 0, 0
        x2, y2 = self.left_panel_width, self.confidence_height
        mid_x = self.left_panel_width // 2
        
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.PANEL_BG, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.BORDER_COLOR, 2)
        cv2.line(canvas, (mid_x, y1), (mid_x, y2), self.DIVIDER_COLOR, 2)
        
        # Title with scan status and timer
        if self.robot_initialized:
            time_in_pos = time.time() - self.last_switch_time
            time_remaining = max(0, self.SCAN_HOLD_TIME - time_in_pos)
            scan_status = f"Looking {self.current_scan_side} ({time_remaining:.1f}s)"
            status_color = self.LEFT_COLOR if self.current_scan_side == "LEFT" else self.RIGHT_COLOR
        else:
            scan_status = "STATIC VIEW"
            status_color = self.TEXT_COLOR
        cv2.putText(canvas, f"CONFIDENCE VIEW", (x1 + 10, y1 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_COLOR, 2)
        cv2.putText(canvas, scan_status, (x1 + 200, y1 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # LEFT section
        left_section_x = x1 + 20
        cv2.putText(canvas, "LEFT TRACK", (left_section_x, y1 + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.LEFT_COLOR, 2)
        
        if self.left_confidences:
            total_conf = sum(self.left_confidences)
            avg_conf = total_conf / len(self.left_confidences)
            cv2.putText(canvas, f"People: {self.left_count}", (left_section_x, y1 + 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_COLOR, 1)
            cv2.putText(canvas, f"Sum Conf: {total_conf:.2f}", (left_section_x, y1 + 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_COLOR, 1)
            cv2.putText(canvas, f"Avg Conf: {avg_conf:.2f}", (left_section_x, y1 + 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_COLOR, 1)
            
            bar_y = y1 + 160
            bar_width = int(avg_conf * 150)
            cv2.rectangle(canvas, (left_section_x, bar_y), 
                         (left_section_x + bar_width, bar_y + 15), self.LEFT_COLOR, -1)
        else:
            cv2.putText(canvas, "No people", (left_section_x, y1 + 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # RIGHT section
        right_section_x = mid_x + 20
        cv2.putText(canvas, "RIGHT TRACK", (right_section_x, y1 + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.RIGHT_COLOR, 2)
        
        if self.right_confidences:
            total_conf = sum(self.right_confidences)
            avg_conf = total_conf / len(self.right_confidences)
            cv2.putText(canvas, f"People: {self.right_count}", (right_section_x, y1 + 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_COLOR, 1)
            cv2.putText(canvas, f"Sum Conf: {total_conf:.2f}", (right_section_x, y1 + 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_COLOR, 1)
            cv2.putText(canvas, f"Avg Conf: {avg_conf:.2f}", (right_section_x, y1 + 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_COLOR, 1)
            
            bar_y = y1 + 160
            bar_width = int(avg_conf * 150)
            cv2.rectangle(canvas, (right_section_x, bar_y), 
                         (right_section_x + bar_width, bar_y + 15), self.RIGHT_COLOR, -1)
        else:
            cv2.putText(canvas, "No people", (right_section_x, y1 + 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    def _draw_camera_panel(self, canvas: np.ndarray):
        """Draw the camera feed panel showing both left and right captures."""
        x1, y1 = 0, self.confidence_height
        x2, y2 = self.left_panel_width, self.window_height
        panel_width = x2 - x1
        panel_height = y2 - y1
        
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.PANEL_BG, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.BORDER_COLOR, 2)
        
        # Title
        # Title
        if self.robot_initialized:
            cv2.putText(canvas, "CAMERA FEED (5s per side)", (x1 + 10, y1 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_COLOR, 2)
        else:
            cv2.putText(canvas, "CAMERA FEED (Static)", (x1 + 10, y1 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_COLOR, 2)
        
        # Split panel into left and right views
        half_width = (panel_width - 30) // 2
        view_height = panel_height - 60
        
        # Scale factor for frames
        if self.cam_height > 0:
            scale = min(half_width / self.cam_width, view_height / self.cam_height)
        else:
            scale = 0.5
        
        new_w = int(self.cam_width * scale)
        new_h = int(self.cam_height * scale)
        
        # LEFT VIEW
        left_view_x = x1 + 10
        left_view_y = y1 + 45
        
        # Highlight active view
        left_active = self.current_scan_side == "LEFT" and self.robot_initialized
        left_label = "LEFT VIEW [ACTIVE]" if left_active else "LEFT VIEW"
        left_thickness = 2 if left_active else 1
        cv2.putText(canvas, left_label, (left_view_x, left_view_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.LEFT_COLOR, left_thickness)
        
        if self._left_frame is not None:
            resized = cv2.resize(self._left_frame, (new_w, new_h))
            # Draw detections
            for det in self.left_detections:
                sx1, sy1 = int(det.x1 * scale), int(det.y1 * scale)
                sx2, sy2 = int(det.x2 * scale), int(det.y2 * scale)
                cv2.rectangle(resized, (sx1, sy1), (sx2, sy2), self.LEFT_COLOR, 2)
                cv2.putText(resized, f"{det.conf:.2f}", (sx1, sy1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.LEFT_COLOR, 1)
            canvas[left_view_y:left_view_y + new_h, left_view_x:left_view_x + new_w] = resized
        else:
            cv2.rectangle(canvas, (left_view_x, left_view_y), 
                         (left_view_x + new_w, left_view_y + new_h), self.BORDER_COLOR, 1)
            cv2.putText(canvas, "Waiting...", (left_view_x + 10, left_view_y + new_h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # RIGHT VIEW
        right_view_x = x1 + 20 + half_width
        right_view_y = y1 + 45
        
        # Highlight active view
        right_active = self.current_scan_side == "RIGHT" and self.robot_initialized
        right_label = "RIGHT VIEW [ACTIVE]" if right_active else "RIGHT VIEW"
        right_thickness = 2 if right_active else 1
        cv2.putText(canvas, right_label, (right_view_x, right_view_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.RIGHT_COLOR, right_thickness)
        
        if self._right_frame is not None:
            resized = cv2.resize(self._right_frame, (new_w, new_h))
            # Draw detections
            for det in self.right_detections:
                sx1, sy1 = int(det.x1 * scale), int(det.y1 * scale)
                sx2, sy2 = int(det.x2 * scale), int(det.y2 * scale)
                cv2.rectangle(resized, (sx1, sy1), (sx2, sy2), self.RIGHT_COLOR, 2)
                cv2.putText(resized, f"{det.conf:.2f}", (sx1, sy1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.RIGHT_COLOR, 1)
            canvas[right_view_y:right_view_y + new_h, right_view_x:right_view_x + new_w] = resized
        else:
            cv2.rectangle(canvas, (right_view_x, right_view_y), 
                         (right_view_x + new_w, right_view_y + new_h), self.BORDER_COLOR, 1)
            cv2.putText(canvas, "Waiting...", (right_view_x + 10, right_view_y + new_h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Divider
        mid_x = x1 + panel_width // 2
        cv2.line(canvas, (mid_x, y1 + 40), (mid_x, y2 - 10), self.DIVIDER_COLOR, 2)
    
    def _draw_right_panel(self, canvas: np.ndarray):
        """Draw the right panel (reserved for future use)."""
        x1, y1 = self.left_panel_width, 0
        x2, y2 = self.window_width, self.window_height
        
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.PANEL_BG, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.BORDER_COLOR, 2)
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.putText(canvas, "(Reserved)", (center_x - 60, center_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
    
    def update(self) -> np.ndarray:
        """
        Update scan state and return rendered frame.
        Non-blocking - holds each position for SCAN_HOLD_TIME seconds.
        """
        # Update scanning state (non-blocking)
        self._update_scan_state()
        
        # Update counts and confidences
        self.left_count = len(self.left_detections)
        self.right_count = len(self.right_detections)
        self.left_confidences = [d.conf for d in self.left_detections]
        self.right_confidences = [d.conf for d in self.right_detections]
        
        # Create canvas
        canvas = np.full((self.window_height, self.window_width, 3), 
                        self.BG_COLOR, dtype=np.uint8)
        
        # Draw panels
        self._draw_confidence_panel(canvas)
        self._draw_camera_panel(canvas)
        self._draw_right_panel(canvas)
        
        return canvas
    
    def get_counts(self) -> Tuple[int, int, dict]:
        """
        Get current left and right counts (for integration with decider).
        
        Returns:
            Tuple of (left_count, right_count, metadata)
        """
        meta = {
            "timestamp": time.time(),
            "total_detections": self.left_count + self.right_count,
            "left_confidences": self.left_confidences.copy(),
            "right_confidences": self.right_confidences.copy(),
            "sample_valid": True,
            "source": "scanning_camera" if self.robot_initialized else "static_camera",
        }
        return self.left_count, self.right_count, meta
    
    def run(self):
        """Run the GUI loop."""
        print("[GUI] Starting GUI with head scanning...")
        print("[GUI] Press 'q' to quit")
        
        window_name = "Trolley Problem - Scanning Detection GUI"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_width, self.window_height)
        
        scan_count = 0
        last_status_time = time.time()
        
        try:
            while True:
                # Update and render
                canvas = self.update()
                scan_count += 1
                
                # Print status every 10 seconds
                if time.time() - last_status_time >= 10.0:
                    print(f"[GUI] Scans: {scan_count}, Left: {self.left_count} ({sum(self.left_confidences):.2f}), "
                          f"Right: {self.right_count} ({sum(self.right_confidences):.2f})")
                    last_status_time = time.time()
                
                # Display
                cv2.imshow(window_name, canvas)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[GUI] Quitting...")
                    break
                
        except KeyboardInterrupt:
            print("[GUI] Interrupted (Ctrl+C)")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("[GUI] Cleaning up...")
        
        # Return head to center
        if self.robot_initialized and self.robot is not None:
            try:
                self._move_head_to(self.PAN_CENTER)
                time.sleep(0.5)
                self.robot.stop()
            except Exception:
                pass
        
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        
        cv2.destroyAllWindows()
        print("[GUI] Cleanup complete!")


def main():
    """Run the GUI standalone."""
    gui = TrolleyGUI(
        window_width=1280,
        window_height=720,
        model_path="scripts/yolov8n-pose.pt",
        conf_thresh=0.35,
        use_robot=True,  # Enable head scanning
    )
    gui.run()


if __name__ == "__main__":
    main()
