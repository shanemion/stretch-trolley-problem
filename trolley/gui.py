"""
GUI visualization for the trolley problem system.

Displays:
- Top Left: Confidence view (left/right aggregate confidence scores)
- Bottom Left: Camera feed with bounding boxes and center divider
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
    side: str  # "LEFT" or "RIGHT"


class TrolleyGUI:
    """
    GUI visualization for trolley problem detection.
    
    Layout:
    +-------------------+-------------------+
    |  CONFIDENCE VIEW  |                   |
    |  LEFT  |  RIGHT   |    (Reserved)     |
    +--------+----------+                   |
    |   CAMERA FEED     |                   |
    |  LEFT  |  RIGHT   |                   |
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
    
    def __init__(
        self,
        window_width: int = 1280,
        window_height: int = 720,
        model_path: str = "scripts/yolov8n-pose.pt",
        conf_thresh: float = 0.35,
        rotate_90_clockwise: bool = True,
    ):
        """
        Initialize the GUI.
        
        Args:
            window_width: Total window width
            window_height: Total window height
            model_path: Path to YOLO model
            conf_thresh: Confidence threshold for detection
            rotate_90_clockwise: Whether to rotate camera feed
        """
        self.window_width = window_width
        self.window_height = window_height
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.rotate_90_clockwise = rotate_90_clockwise
        
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
        
        # Detection state
        self.detections: List[Detection] = []
        self.left_count = 0
        self.right_count = 0
        self.left_confidences: List[float] = []
        self.right_confidences: List[float] = []
        self._current_frame = None
        
        # Camera geometry (after rotation)
        if rotate_90_clockwise:
            self.cam_width = 480
            self.cam_height = 640
        else:
            self.cam_width = 640
            self.cam_height = 480
        self.center_x = self.cam_width / 2.0
        
        # Initialize components
        self._init_camera()
    
    def _init_camera(self):
        """Initialize RealSense camera and YOLO model."""
        import traceback
        
        try:
            print("[GUI] Importing pyrealsense2...")
            import pyrealsense2 as rs
            self._rs = rs  # Store for later use
            
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
    
    def _get_frame(self) -> Optional[np.ndarray]:
        """Get current frame from camera."""
        if not self.camera_initialized or self.pipeline is None:
            return None
        
        try:
            # Use same approach as working script - no timeout
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            
            if not color_frame:
                return None
            
            # Convert to numpy array (same as working script)
            img = np.asanyarray(color_frame.get_data())
            
            # Rotate if needed
            if self.rotate_90_clockwise:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            
            # Frame retrieved successfully - clear any previous errors
            self.camera_error = False
            return img
            
        except Exception as e:
            # Only set error if we have persistent failures
            self.camera_error = True
            self.camera_error_msg = str(e)
            print(f"[GUI] Frame capture error: {e}")
            return None
    
    def _detect_people(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLO detection on frame."""
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
                side = "LEFT" if nose_x < self.center_x else "RIGHT"
                
                detections.append(Detection(
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    nose_x=float(nose_x), nose_y=float(nose_y),
                    conf=conf, side=side
                ))
        
        return detections
    
    def _draw_confidence_panel(self, canvas: np.ndarray):
        """Draw the confidence view panel (top left)."""
        # Panel boundaries
        x1, y1 = 0, 0
        x2, y2 = self.left_panel_width, self.confidence_height
        mid_x = self.left_panel_width // 2
        
        # Draw panel background
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.PANEL_BG, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.BORDER_COLOR, 2)
        
        # Draw divider
        cv2.line(canvas, (mid_x, y1), (mid_x, y2), self.DIVIDER_COLOR, 2)
        
        # Title
        cv2.putText(canvas, "CONFIDENCE VIEW", (x1 + 10, y1 + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.TEXT_COLOR, 2)
        
        # LEFT section
        left_section_x = x1 + 20
        cv2.putText(canvas, "LEFT", (left_section_x, y1 + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.LEFT_COLOR, 2)
        
        # Left confidence aggregate
        if self.left_confidences:
            total_conf = sum(self.left_confidences)
            avg_conf = total_conf / len(self.left_confidences)
            cv2.putText(canvas, f"Count: {self.left_count}", (left_section_x, y1 + 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_COLOR, 1)
            cv2.putText(canvas, f"Sum Conf: {total_conf:.2f}", (left_section_x, y1 + 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_COLOR, 1)
            cv2.putText(canvas, f"Avg Conf: {avg_conf:.2f}", (left_section_x, y1 + 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_COLOR, 1)
            
            # Draw confidence bars
            bar_y = y1 + 180
            bar_width = int((total_conf / max(len(self.left_confidences), 1)) * 150)
            cv2.rectangle(canvas, (left_section_x, bar_y), 
                         (left_section_x + bar_width, bar_y + 20), self.LEFT_COLOR, -1)
        else:
            cv2.putText(canvas, "No detections", (left_section_x, y1 + 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # RIGHT section
        right_section_x = mid_x + 20
        cv2.putText(canvas, "RIGHT", (right_section_x, y1 + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.RIGHT_COLOR, 2)
        
        # Right confidence aggregate
        if self.right_confidences:
            total_conf = sum(self.right_confidences)
            avg_conf = total_conf / len(self.right_confidences)
            cv2.putText(canvas, f"Count: {self.right_count}", (right_section_x, y1 + 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_COLOR, 1)
            cv2.putText(canvas, f"Sum Conf: {total_conf:.2f}", (right_section_x, y1 + 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_COLOR, 1)
            cv2.putText(canvas, f"Avg Conf: {avg_conf:.2f}", (right_section_x, y1 + 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_COLOR, 1)
            
            # Draw confidence bars
            bar_y = y1 + 180
            bar_width = int((total_conf / max(len(self.right_confidences), 1)) * 150)
            cv2.rectangle(canvas, (right_section_x, bar_y), 
                         (right_section_x + bar_width, bar_y + 20), self.RIGHT_COLOR, -1)
        else:
            cv2.putText(canvas, "No detections", (right_section_x, y1 + 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    def _draw_camera_panel(self, canvas: np.ndarray, frame: Optional[np.ndarray]):
        """Draw the camera feed panel (bottom left)."""
        # Panel boundaries
        x1, y1 = 0, self.confidence_height
        x2, y2 = self.left_panel_width, self.window_height
        panel_width = x2 - x1
        panel_height = y2 - y1
        mid_x = x1 + panel_width // 2
        
        # Draw panel background
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.PANEL_BG, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.BORDER_COLOR, 2)
        
        if frame is not None:
            # Resize frame to fit panel (maintaining aspect ratio)
            frame_h, frame_w = frame.shape[:2]
            scale = min((panel_width - 20) / frame_w, (panel_height - 60) / frame_h)
            new_w, new_h = int(frame_w * scale), int(frame_h * scale)
            
            resized_frame = cv2.resize(frame, (new_w, new_h))
            
            # Draw bounding boxes and annotations on resized frame
            for det in self.detections:
                # Scale coordinates
                sx1 = int(det.x1 * scale)
                sy1 = int(det.y1 * scale)
                sx2 = int(det.x2 * scale)
                sy2 = int(det.y2 * scale)
                snose_x = int(det.nose_x * scale)
                snose_y = int(det.nose_y * scale)
                
                # Choose color based on side
                color = self.LEFT_COLOR if det.side == "LEFT" else self.RIGHT_COLOR
                
                # Draw bounding box
                cv2.rectangle(resized_frame, (sx1, sy1), (sx2, sy2), color, 2)
                
                # Draw nose keypoint
                cv2.circle(resized_frame, (snose_x, snose_y), 5, (255, 0, 255), -1)
                
                # Draw confidence label
                label = f"{det.side}: {det.conf:.2f}"
                cv2.putText(resized_frame, label, (sx1, sy1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw center divider on resized frame
            center_line_x = int(self.center_x * scale)
            cv2.line(resized_frame, (center_line_x, 0), (center_line_x, new_h), 
                    self.DIVIDER_COLOR, 2)
            
            # Draw LEFT/RIGHT labels on frame
            cv2.putText(resized_frame, "LEFT", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.LEFT_COLOR, 2)
            cv2.putText(resized_frame, "RIGHT", (center_line_x + 10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.RIGHT_COLOR, 2)
            
            # Calculate position to center the frame in the panel
            offset_x = x1 + (panel_width - new_w) // 2
            offset_y = y1 + 40 + (panel_height - 60 - new_h) // 2
            
            # Place frame on canvas
            canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized_frame
            
            # Title
            cv2.putText(canvas, "CAMERA FEED", (x1 + 10, y1 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.TEXT_COLOR, 2)
        else:
            # Camera error placeholder
            cv2.putText(canvas, "CAMERA FEED", (x1 + 10, y1 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.TEXT_COLOR, 2)
            
            # Error message box
            err_box_x1 = x1 + 20
            err_box_y1 = y1 + 60
            err_box_x2 = x2 - 20
            err_box_y2 = y2 - 20
            
            cv2.rectangle(canvas, (err_box_x1, err_box_y1), (err_box_x2, err_box_y2), 
                         self.ERROR_COLOR, 2)
            
            # Error message
            cv2.putText(canvas, "CAMERA FEED FAILED TO MOUNT", 
                       (err_box_x1 + 20, err_box_y1 + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ERROR_COLOR, 2)
            
            if self.camera_error_msg:
                # Split error message into multiple lines
                err_lines = []
                msg = self.camera_error_msg
                max_chars = 60
                while msg:
                    if len(msg) <= max_chars:
                        err_lines.append(msg)
                        break
                    else:
                        # Find a good break point
                        break_at = msg.rfind(' ', 0, max_chars)
                        if break_at == -1:
                            break_at = max_chars
                        err_lines.append(msg[:break_at])
                        msg = msg[break_at:].lstrip()
                
                y_offset = err_box_y1 + 80
                for i, line in enumerate(err_lines[:5]):  # Max 5 lines
                    cv2.putText(canvas, line, 
                               (err_box_x1 + 20, y_offset + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            # Hint
            cv2.putText(canvas, "Check terminal for full error details", 
                       (err_box_x1 + 20, err_box_y2 - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            cv2.putText(canvas, "Try: Check if another process is using the camera", 
                       (err_box_x1 + 20, err_box_y2 - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    def _draw_right_panel(self, canvas: np.ndarray):
        """Draw the right panel (reserved for future use)."""
        # Panel boundaries
        x1, y1 = self.left_panel_width, 0
        x2, y2 = self.window_width, self.window_height
        
        # Draw panel background
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.PANEL_BG, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.BORDER_COLOR, 2)
        
        # Reserved label
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.putText(canvas, "(Reserved)", (center_x - 60, center_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
    
    def update(self) -> np.ndarray:
        """
        Update detection and return rendered frame.
        
        Returns:
            np.ndarray: The rendered GUI frame
        """
        # Get camera frame
        frame = self._get_frame()
        
        # Store current frame for display
        self._current_frame = frame
        
        # Run detection if we have a frame
        if frame is not None:
            self.detections = self._detect_people(frame)
            
            # Update counts and confidences
            left_dets = [d for d in self.detections if d.side == "LEFT"]
            right_dets = [d for d in self.detections if d.side == "RIGHT"]
            
            self.left_count = len(left_dets)
            self.right_count = len(right_dets)
            self.left_confidences = [d.conf for d in left_dets]
            self.right_confidences = [d.conf for d in right_dets]
        
        # Create canvas
        canvas = np.full((self.window_height, self.window_width, 3), 
                        self.BG_COLOR, dtype=np.uint8)
        
        # Draw panels
        self._draw_confidence_panel(canvas)
        self._draw_camera_panel(canvas, frame)
        self._draw_right_panel(canvas)
        
        return canvas
    
    def run(self):
        """Run the GUI loop."""
        print("[GUI] Starting GUI...")
        print("[GUI] Press 'q' to quit")
        
        window_name = "Trolley Problem - Detection GUI"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_width, self.window_height)
        
        frame_count = 0
        last_status_time = time.time()
        
        try:
            while True:
                # Update and render
                canvas = self.update()
                frame_count += 1
                
                # Print status every 5 seconds
                if time.time() - last_status_time >= 5.0:
                    has_frame = self._current_frame is not None
                    print(f"[GUI] Frames: {frame_count}, Camera OK: {has_frame}, "
                          f"Detections: {len(self.detections)}, Error: {self.camera_error}")
                    last_status_time = time.time()
                
                # Display
                cv2.imshow(window_name, canvas)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[GUI] Quitting...")
                    break
                
                # Small delay
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("[GUI] Interrupted (Ctrl+C)")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("[GUI] Cleaning up...")
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
    )
    gui.run()


if __name__ == "__main__":
    main()
