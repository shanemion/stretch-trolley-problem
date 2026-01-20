#!/usr/bin/env python3
"""
Comprehensive Trolley Problem GUI

Combines:
- Scanning camera detection (left/right views)
- 10-second countdown with decision logic
- Interactive start/stop/restart buttons
- Lever action execution

Usage:
    python run_trolley_gui.py [OPTIONS]

Press 'q' to quit, or use the on-screen buttons.
"""

import argparse
import time
import numpy as np
import cv2
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from trolley import config


class TrolleyState(Enum):
    """State machine for trolley scenario."""
    IDLE = "IDLE"           # Waiting to start
    COUNTDOWN = "COUNTDOWN" # Counting down, scanning for people
    DECIDING = "DECIDING"   # Making decision
    EXECUTING = "EXECUTING" # Executing lever action
    COMPLETE = "COMPLETE"   # Scenario complete


@dataclass
class Detection:
    """Person detection."""
    x1: float
    y1: float
    x2: float
    y2: float
    nose_x: float
    nose_y: float
    conf: float
    side: str


class Button:
    """Simple clickable button."""
    def __init__(self, x: int, y: int, w: int, h: int, label: str, 
                 color: tuple, hover_color: tuple):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.label = label
        self.color = color
        self.hover_color = hover_color
        self.hovered = False
        self.enabled = True
    
    def contains(self, px: int, py: int) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h
    
    def draw(self, canvas: np.ndarray):
        if not self.enabled:
            color = (80, 80, 80)
            text_color = (120, 120, 120)
        elif self.hovered:
            color = self.hover_color
            text_color = (255, 255, 255)
        else:
            color = self.color
            text_color = (255, 255, 255)
        
        cv2.rectangle(canvas, (self.x, self.y), (self.x + self.w, self.y + self.h), color, -1)
        cv2.rectangle(canvas, (self.x, self.y), (self.x + self.w, self.y + self.h), (200, 200, 200), 2)
        
        # Center text
        text_size = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = self.x + (self.w - text_size[0]) // 2
        text_y = self.y + (self.h + text_size[1]) // 2
        cv2.putText(canvas, self.label, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)


class TrolleyGUIComplete:
    """
    Complete Trolley Problem GUI with countdown and decision logic.
    """
    
    # Colors
    BG_COLOR = (30, 30, 30)
    PANEL_BG = (45, 45, 45)
    BORDER_COLOR = (100, 100, 100)
    TEXT_COLOR = (255, 255, 255)
    LEFT_COLOR = (0, 165, 255)     # Orange
    RIGHT_COLOR = (255, 100, 100)  # Blue
    DIVIDER_COLOR = (0, 255, 255)  # Yellow
    SUCCESS_COLOR = (0, 255, 0)    # Green
    ERROR_COLOR = (0, 0, 255)      # Red
    WARNING_COLOR = (0, 200, 255)  # Orange-yellow
    
    # Scanning parameters
    PAN_LEFT = 0.4
    PAN_RIGHT = -0.4
    PAN_CENTER = 0.0
    SCAN_SETTLE_TIME = 0.5
    SCAN_HOLD_TIME = 5.0
    
    # Countdown parameters
    COUNTDOWN_TIME = 10.0
    
    def __init__(
        self,
        window_width: int = 1280,
        window_height: int = 720,
        model_path: str = "scripts/yolov8n-pose.pt",
        conf_thresh: float = 0.35,
        rotate_90_clockwise: bool = True,
        use_robot: bool = True,
        dry_run: bool = True,
    ):
        self.window_width = window_width
        self.window_height = window_height
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.rotate_90_clockwise = rotate_90_clockwise
        self.use_robot = use_robot
        self.dry_run = dry_run
        
        # Layout
        self.left_panel_width = window_width // 2
        self.right_panel_width = window_width - self.left_panel_width
        self.button_height = 50
        self.confidence_height = (window_height - self.button_height) // 3
        self.camera_height = window_height - self.confidence_height - self.button_height
        
        # Camera state
        self.camera_initialized = False
        self.pipeline = None
        self.align = None
        self.model = None
        
        # Robot state
        self.robot = None
        self.robot_initialized = False
        
        # Scanning state
        self.current_scan_side = "LEFT"
        self.scan_state = "INIT"
        self.last_switch_time = 0
        self.head_moved = False
        
        # Detection state
        self.left_detections: List[Detection] = []
        self.right_detections: List[Detection] = []
        self.left_count = 0
        self.right_count = 0
        self.left_confidences: List[float] = []
        self.right_confidences: List[float] = []
        self._left_frame = None
        self._right_frame = None
        
        # Trolley state
        self.trolley_state = TrolleyState.IDLE
        self.countdown_start_time = 0
        self.decision = None  # "DIVERT_RIGHT", "STAY_LEFT", or None
        self.decision_reason = ""
        self.final_left_conf = 0.0
        self.final_right_conf = 0.0
        
        # Camera geometry
        if rotate_90_clockwise:
            self.cam_width = 480
            self.cam_height = 640
        else:
            self.cam_width = 640
            self.cam_height = 480
        self.center_x = self.cam_width / 2.0
        
        # Buttons
        self._create_buttons()
        
        # Mouse state
        self.mouse_pos = (0, 0)
        
        # Initialize components
        self._init_camera()
        if use_robot:
            self._init_robot()
    
    def _create_buttons(self):
        """Create control buttons."""
        btn_width = 120
        btn_spacing = 20
        start_x = self.left_panel_width + 50
        btn_y = 15
        
        self.btn_start = Button(
            start_x, btn_y, btn_width, 35,
            "START", (0, 150, 0), (0, 200, 0)
        )
        self.btn_stop = Button(
            start_x + btn_width + btn_spacing, btn_y, btn_width, 35,
            "STOP", (0, 0, 180), (0, 0, 220)
        )
        self.btn_restart = Button(
            start_x + 2 * (btn_width + btn_spacing), btn_y, btn_width, 35,
            "RESTART", (150, 100, 0), (200, 140, 0)
        )
        
        self.buttons = [self.btn_start, self.btn_stop, self.btn_restart]
    
    def _init_camera(self):
        """Initialize RealSense camera and YOLO model."""
        try:
            import pyrealsense2 as rs
            from ultralytics import YOLO
            
            print("[GUI] Loading YOLO model...")
            self.model = YOLO(self.model_path)
            
            print("[GUI] Initializing RealSense...")
            self.pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline.start(cfg)
            self.align = rs.align(rs.stream.color)
            self._rs = rs
            
            self.camera_initialized = True
            print("[GUI] Camera ready!")
        except Exception as e:
            print(f"[GUI] Camera init failed: {e}")
            self.camera_initialized = False
    
    def _init_robot(self):
        """Initialize robot for head control and lever action."""
        try:
            import stretch_body.robot as robot_module
            
            print("[GUI] Initializing robot...")
            self.robot = robot_module.Robot()
            self.robot.startup()
            self.robot_initialized = True
            self._move_head_to(self.PAN_CENTER)
            print("[GUI] Robot ready!")
        except Exception as e:
            print(f"[GUI] Robot init failed: {e}")
            self.robot_initialized = False
    
    def _move_head_to(self, pan: float):
        if self.robot_initialized and self.robot:
            try:
                self.robot.head.move_to('head_pan', pan)
                self.robot.push_command()
            except Exception as e:
                print(f"[GUI] Head error: {e}")
    
    def _get_frame(self) -> Optional[np.ndarray]:
        if not self.camera_initialized:
            return None
        try:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color = aligned.get_color_frame()
            if not color:
                return None
            img = np.asanyarray(color.get_data())
            if self.rotate_90_clockwise:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            return img
        except:
            return None
    
    def _detect_people(self, frame: np.ndarray, side: str) -> List[Detection]:
        if self.model is None:
            return []
        
        results = self.model(frame, verbose=False)
        dets = []
        
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
                dets.append(Detection(
                    float(x1), float(y1), float(x2), float(y2),
                    float(nose_x), float(nose_y), conf, side
                ))
        return dets
    
    def _detect_people_split(self, frame: np.ndarray) -> List[Detection]:
        """Detect and split by nose position (no robot mode)."""
        if self.model is None:
            return []
        
        results = self.model(frame, verbose=False)
        dets = []
        
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
                dets.append(Detection(
                    float(x1), float(y1), float(x2), float(y2),
                    float(nose_x), float(nose_y), conf, side
                ))
        return dets
    
    def _update_scan(self):
        """Update scanning state machine."""
        current_time = time.time()
        
        if not self.robot_initialized:
            frame = self._get_frame()
            if frame is not None:
                self._left_frame = frame.copy()
                self._right_frame = frame.copy()
                all_dets = self._detect_people_split(frame)
                self.left_detections = [d for d in all_dets if d.side == "LEFT"]
                self.right_detections = [d for d in all_dets if d.side == "RIGHT"]
            return
        
        if self.scan_state == "INIT":
            self.current_scan_side = "LEFT"
            self._move_head_to(self.PAN_LEFT)
            self.last_switch_time = current_time
            self.head_moved = True
            self.scan_state = "LOOKING_LEFT"
            return
        
        time_in_pos = current_time - self.last_switch_time
        
        if time_in_pos >= self.SCAN_HOLD_TIME:
            if self.current_scan_side == "LEFT":
                self.current_scan_side = "RIGHT"
                self._move_head_to(self.PAN_RIGHT)
                self.scan_state = "LOOKING_RIGHT"
            else:
                self.current_scan_side = "LEFT"
                self._move_head_to(self.PAN_LEFT)
                self.scan_state = "LOOKING_LEFT"
            self.last_switch_time = current_time
            self.head_moved = True
            return
        
        if self.head_moved and time_in_pos < self.SCAN_SETTLE_TIME:
            return
        self.head_moved = False
        
        frame = self._get_frame()
        if frame is not None:
            dets = self._detect_people(frame, self.current_scan_side)
            if self.current_scan_side == "LEFT":
                self._left_frame = frame.copy()
                if len(dets) >= len(self.left_detections):
                    self.left_detections = dets
            else:
                self._right_frame = frame.copy()
                if len(dets) >= len(self.right_detections):
                    self.right_detections = dets
    
    def _execute_divert(self):
        """Execute the lever divert action."""
        if not self.robot_initialized or self.dry_run:
            print("[GUI] DIVERT ACTION (dry-run or no robot)")
            return
        
        from trolley.actions import divert_lever
        print("[GUI] Executing divert lever sequence...")
        divert_lever(self.robot, config.DIVERT_DISTANCE_M, dry_run=False)
        print("[GUI] Divert complete!")
    
    def _update_trolley_state(self):
        """Update trolley scenario state machine."""
        if self.trolley_state == TrolleyState.COUNTDOWN:
            elapsed = time.time() - self.countdown_start_time
            if elapsed >= self.COUNTDOWN_TIME:
                self.trolley_state = TrolleyState.DECIDING
        
        elif self.trolley_state == TrolleyState.DECIDING:
            # Cache final confidence sums
            self.final_left_conf = sum(self.left_confidences)
            self.final_right_conf = sum(self.right_confidences)
            
            # Decision logic
            if self.final_right_conf < self.final_left_conf:
                self.decision = "DIVERT_RIGHT"
                self.decision_reason = f"Right ({self.final_right_conf:.2f}) < Left ({self.final_left_conf:.2f})"
            else:
                self.decision = "STAY_LEFT"
                if self.final_right_conf == self.final_left_conf:
                    self.decision_reason = f"Tie ({self.final_left_conf:.2f} = {self.final_right_conf:.2f})"
                else:
                    self.decision_reason = f"Left ({self.final_left_conf:.2f}) <= Right ({self.final_right_conf:.2f})"
            
            self.trolley_state = TrolleyState.EXECUTING
        
        elif self.trolley_state == TrolleyState.EXECUTING:
            if self.decision == "DIVERT_RIGHT":
                self._execute_divert()
            self.trolley_state = TrolleyState.COMPLETE
    
    def _handle_click(self, x: int, y: int):
        """Handle mouse click."""
        if self.btn_start.enabled and self.btn_start.contains(x, y):
            self._start_scenario()
        elif self.btn_stop.enabled and self.btn_stop.contains(x, y):
            self._stop_scenario()
        elif self.btn_restart.enabled and self.btn_restart.contains(x, y):
            self._restart_scenario()
    
    def _start_scenario(self):
        """Start the trolley scenario."""
        if self.trolley_state == TrolleyState.IDLE:
            self.trolley_state = TrolleyState.COUNTDOWN
            self.countdown_start_time = time.time()
            self.decision = None
            self.decision_reason = ""
            print("[GUI] Scenario started!")
    
    def _stop_scenario(self):
        """Stop the current scenario."""
        self.trolley_state = TrolleyState.IDLE
        self.decision = None
        print("[GUI] Scenario stopped.")
    
    def _restart_scenario(self):
        """Restart the scenario."""
        self._stop_scenario()
        self.left_detections = []
        self.right_detections = []
        self._start_scenario()
        print("[GUI] Scenario restarted!")
    
    def _update_buttons(self):
        """Update button states based on scenario state."""
        if self.trolley_state == TrolleyState.IDLE:
            self.btn_start.enabled = True
            self.btn_stop.enabled = False
            self.btn_restart.enabled = False
        elif self.trolley_state in [TrolleyState.COUNTDOWN, TrolleyState.DECIDING, TrolleyState.EXECUTING]:
            self.btn_start.enabled = False
            self.btn_stop.enabled = True
            self.btn_restart.enabled = False
        else:  # COMPLETE
            self.btn_start.enabled = False
            self.btn_stop.enabled = False
            self.btn_restart.enabled = True
        
        # Update hover states
        for btn in self.buttons:
            btn.hovered = btn.contains(*self.mouse_pos)
    
    def _draw_confidence_panel(self, canvas: np.ndarray):
        """Draw confidence view (top left)."""
        x1, y1 = 0, self.button_height
        x2, y2 = self.left_panel_width, self.button_height + self.confidence_height
        mid_x = self.left_panel_width // 2
        
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.PANEL_BG, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.BORDER_COLOR, 2)
        cv2.line(canvas, (mid_x, y1), (mid_x, y2), self.DIVIDER_COLOR, 2)
        
        # Title
        if self.robot_initialized:
            time_in_pos = time.time() - self.last_switch_time
            time_rem = max(0, self.SCAN_HOLD_TIME - time_in_pos)
            status = f"Looking {self.current_scan_side} ({time_rem:.1f}s)"
            color = self.LEFT_COLOR if self.current_scan_side == "LEFT" else self.RIGHT_COLOR
        else:
            status = "Static View"
            color = self.TEXT_COLOR
        
        cv2.putText(canvas, "DETECTION STATUS", (x1 + 10, y1 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_COLOR, 2)
        cv2.putText(canvas, status, (x1 + 220, y1 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # LEFT stats
        left_x = x1 + 20
        cv2.putText(canvas, "LEFT TRACK", (left_x, y1 + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.LEFT_COLOR, 2)
        
        left_sum = sum(self.left_confidences) if self.left_confidences else 0
        cv2.putText(canvas, f"People: {self.left_count}", (left_x, y1 + 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_COLOR, 1)
        cv2.putText(canvas, f"Sum Conf: {left_sum:.2f}", (left_x, y1 + 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_COLOR, 1)
        
        # Confidence bar
        bar_width = int(min(left_sum, 3.0) / 3.0 * 150)
        cv2.rectangle(canvas, (left_x, y1 + 125), (left_x + bar_width, y1 + 140), self.LEFT_COLOR, -1)
        
        # RIGHT stats
        right_x = mid_x + 20
        cv2.putText(canvas, "RIGHT TRACK", (right_x, y1 + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.RIGHT_COLOR, 2)
        
        right_sum = sum(self.right_confidences) if self.right_confidences else 0
        cv2.putText(canvas, f"People: {self.right_count}", (right_x, y1 + 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_COLOR, 1)
        cv2.putText(canvas, f"Sum Conf: {right_sum:.2f}", (right_x, y1 + 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_COLOR, 1)
        
        bar_width = int(min(right_sum, 3.0) / 3.0 * 150)
        cv2.rectangle(canvas, (right_x, y1 + 125), (right_x + bar_width, y1 + 140), self.RIGHT_COLOR, -1)
    
    def _draw_camera_panel(self, canvas: np.ndarray):
        """Draw camera views (bottom left)."""
        x1 = 0
        y1 = self.button_height + self.confidence_height
        x2 = self.left_panel_width
        y2 = self.window_height
        panel_width = x2 - x1
        panel_height = y2 - y1
        
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.PANEL_BG, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.BORDER_COLOR, 2)
        
        cv2.putText(canvas, "CAMERA VIEWS", (x1 + 10, y1 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_COLOR, 2)
        
        half_width = (panel_width - 30) // 2
        view_height = panel_height - 60
        
        if self.cam_height > 0:
            scale = min(half_width / self.cam_width, view_height / self.cam_height)
        else:
            scale = 0.3
        
        new_w = int(self.cam_width * scale)
        new_h = int(self.cam_height * scale)
        
        # LEFT view
        lv_x, lv_y = x1 + 10, y1 + 45
        active_l = self.current_scan_side == "LEFT" and self.robot_initialized
        label_l = "LEFT [ACTIVE]" if active_l else "LEFT"
        cv2.putText(canvas, label_l, (lv_x, lv_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.LEFT_COLOR, 2 if active_l else 1)
        
        if self._left_frame is not None:
            resized = cv2.resize(self._left_frame, (new_w, new_h))
            for det in self.left_detections:
                sx1, sy1 = int(det.x1 * scale), int(det.y1 * scale)
                sx2, sy2 = int(det.x2 * scale), int(det.y2 * scale)
                cv2.rectangle(resized, (sx1, sy1), (sx2, sy2), self.LEFT_COLOR, 2)
            canvas[lv_y:lv_y + new_h, lv_x:lv_x + new_w] = resized
        else:
            cv2.rectangle(canvas, (lv_x, lv_y), (lv_x + new_w, lv_y + new_h), self.BORDER_COLOR, 1)
        
        # RIGHT view
        rv_x, rv_y = x1 + 20 + half_width, y1 + 45
        active_r = self.current_scan_side == "RIGHT" and self.robot_initialized
        label_r = "RIGHT [ACTIVE]" if active_r else "RIGHT"
        cv2.putText(canvas, label_r, (rv_x, rv_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.RIGHT_COLOR, 2 if active_r else 1)
        
        if self._right_frame is not None:
            resized = cv2.resize(self._right_frame, (new_w, new_h))
            for det in self.right_detections:
                sx1, sy1 = int(det.x1 * scale), int(det.y1 * scale)
                sx2, sy2 = int(det.x2 * scale), int(det.y2 * scale)
                cv2.rectangle(resized, (sx1, sy1), (sx2, sy2), self.RIGHT_COLOR, 2)
            canvas[rv_y:rv_y + new_h, rv_x:rv_x + new_w] = resized
        else:
            cv2.rectangle(canvas, (rv_x, rv_y), (rv_x + new_w, rv_y + new_h), self.BORDER_COLOR, 1)
        
        cv2.line(canvas, (x1 + panel_width // 2, y1 + 40), 
                (x1 + panel_width // 2, y2 - 10), self.DIVIDER_COLOR, 2)
    
    def _draw_control_panel(self, canvas: np.ndarray):
        """Draw the right panel with countdown and status."""
        x1 = self.left_panel_width
        y1 = self.button_height
        x2 = self.window_width
        y2 = self.window_height
        panel_width = x2 - x1
        center_x = x1 + panel_width // 2
        
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.PANEL_BG, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.BORDER_COLOR, 2)
        
        # Title
        cv2.putText(canvas, "TROLLEY CONTROL", (x1 + 20, y1 + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.TEXT_COLOR, 2)
        
        if self.trolley_state == TrolleyState.IDLE:
            # Waiting to start
            cv2.putText(canvas, "Press START to begin scenario", (x1 + 50, y1 + 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
            cv2.putText(canvas, "The trolley will arrive at the", (x1 + 50, y1 + 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_COLOR, 1)
            cv2.putText(canvas, "LEFT track by default.", (x1 + 50, y1 + 210),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.LEFT_COLOR, 2)
            cv2.putText(canvas, "You have 10 seconds to decide", (x1 + 50, y1 + 260),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_COLOR, 1)
            cv2.putText(canvas, "whether to divert to the right.", (x1 + 50, y1 + 290),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_COLOR, 1)
        
        elif self.trolley_state == TrolleyState.COUNTDOWN:
            elapsed = time.time() - self.countdown_start_time
            remaining = max(0, self.COUNTDOWN_TIME - elapsed)
            
            # Big countdown
            cv2.putText(canvas, "TROLLEY ARRIVING AT", (x1 + 60, y1 + 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.WARNING_COLOR, 2)
            cv2.putText(canvas, "LEFT TRACK", (x1 + 120, y1 + 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.LEFT_COLOR, 3)
            
            # Countdown number
            cv2.putText(canvas, f"{remaining:.1f}", (center_x - 60, y1 + 280),
                       cv2.FONT_HERSHEY_SIMPLEX, 4.0, self.WARNING_COLOR, 5)
            cv2.putText(canvas, "seconds", (center_x - 60, y1 + 320),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.TEXT_COLOR, 2)
            
            # Current detection summary
            left_sum = sum(self.left_confidences) if self.left_confidences else 0
            right_sum = sum(self.right_confidences) if self.right_confidences else 0
            
            cv2.putText(canvas, "CURRENT DETECTION:", (x1 + 50, y1 + 400),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_COLOR, 2)
            cv2.putText(canvas, f"Left: {self.left_count} people ({left_sum:.2f})", (x1 + 50, y1 + 435),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.LEFT_COLOR, 1)
            cv2.putText(canvas, f"Right: {self.right_count} people ({right_sum:.2f})", (x1 + 50, y1 + 465),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.RIGHT_COLOR, 1)
            
            # Prediction
            if right_sum < left_sum:
                prediction = "Will DIVERT to RIGHT"
                pred_color = self.RIGHT_COLOR
            else:
                prediction = "Will STAY on LEFT"
                pred_color = self.LEFT_COLOR
            cv2.putText(canvas, f"Prediction: {prediction}", (x1 + 50, y1 + 510),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, pred_color, 2)
        
        elif self.trolley_state in [TrolleyState.DECIDING, TrolleyState.EXECUTING]:
            cv2.putText(canvas, "MAKING DECISION...", (x1 + 100, y1 + 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.WARNING_COLOR, 2)
        
        elif self.trolley_state == TrolleyState.COMPLETE:
            # Show decision result
            if self.decision == "DIVERT_RIGHT":
                cv2.putText(canvas, "DECISION:", (x1 + 50, y1 + 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.TEXT_COLOR, 2)
                cv2.putText(canvas, "DIVERTED TO RIGHT", (x1 + 50, y1 + 160),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.RIGHT_COLOR, 3)
                cv2.putText(canvas, "Lever was pulled!", (x1 + 50, y1 + 210),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.SUCCESS_COLOR, 2)
            else:
                cv2.putText(canvas, "DECISION:", (x1 + 50, y1 + 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.TEXT_COLOR, 2)
                cv2.putText(canvas, "STAYED ON LEFT", (x1 + 50, y1 + 160),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.LEFT_COLOR, 3)
                cv2.putText(canvas, "No action taken.", (x1 + 50, y1 + 210),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
            
            # Show reason
            cv2.putText(canvas, "Reason:", (x1 + 50, y1 + 280),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_COLOR, 1)
            cv2.putText(canvas, self.decision_reason, (x1 + 50, y1 + 310),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Final counts
            cv2.putText(canvas, f"Final Left: {self.left_count} ({self.final_left_conf:.2f})", (x1 + 50, y1 + 380),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.LEFT_COLOR, 1)
            cv2.putText(canvas, f"Final Right: {self.right_count} ({self.final_right_conf:.2f})", (x1 + 50, y1 + 410),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.RIGHT_COLOR, 1)
            
            cv2.putText(canvas, "Press RESTART for new scenario", (x1 + 50, y1 + 500),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    
    def _draw_button_bar(self, canvas: np.ndarray):
        """Draw the top button bar."""
        cv2.rectangle(canvas, (0, 0), (self.window_width, self.button_height), (40, 40, 40), -1)
        cv2.line(canvas, (0, self.button_height), (self.window_width, self.button_height), self.BORDER_COLOR, 2)
        
        # Title on left side
        cv2.putText(canvas, "TROLLEY PROBLEM", (15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.TEXT_COLOR, 2)
        
        # Dry-run indicator
        if self.dry_run:
            cv2.putText(canvas, "[DRY-RUN]", (220, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.WARNING_COLOR, 1)
        
        # Draw buttons
        for btn in self.buttons:
            btn.draw(canvas)
    
    def update(self) -> np.ndarray:
        """Update and render frame."""
        # Update scanning
        self._update_scan()
        
        # Update counts
        self.left_count = len(self.left_detections)
        self.right_count = len(self.right_detections)
        self.left_confidences = [d.conf for d in self.left_detections]
        self.right_confidences = [d.conf for d in self.right_detections]
        
        # Update trolley state
        self._update_trolley_state()
        
        # Update buttons
        self._update_buttons()
        
        # Create canvas
        canvas = np.full((self.window_height, self.window_width, 3), self.BG_COLOR, dtype=np.uint8)
        
        # Draw panels
        self._draw_button_bar(canvas)
        self._draw_confidence_panel(canvas)
        self._draw_camera_panel(canvas)
        self._draw_control_panel(canvas)
        
        return canvas
    
    def run(self):
        """Run the GUI."""
        print("[GUI] Starting Trolley Problem GUI...")
        print("[GUI] Press 'q' to quit")
        
        window_name = "Trolley Problem - Complete System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_width, self.window_height)
        
        def mouse_callback(event, x, y, flags, param):
            self.mouse_pos = (x, y)
            if event == cv2.EVENT_LBUTTONDOWN:
                self._handle_click(x, y)
        
        cv2.setMouseCallback(window_name, mouse_callback)
        
        try:
            while True:
                canvas = self.update()
                cv2.imshow(window_name, canvas)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[GUI] Quitting...")
                    break
                elif key == ord('s'):
                    self._start_scenario()
                elif key == ord('x'):
                    self._stop_scenario()
                elif key == ord('r'):
                    self._restart_scenario()
        
        except KeyboardInterrupt:
            print("[GUI] Interrupted")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("[GUI] Cleaning up...")
        if self.robot_initialized and self.robot:
            try:
                self._move_head_to(self.PAN_CENTER)
                time.sleep(0.3)
                self.robot.stop()
            except:
                pass
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass
        cv2.destroyAllWindows()
        print("[GUI] Done!")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trolley Problem - Complete GUI System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--width", type=int, default=1280, help="Window width")
    parser.add_argument("--height", type=int, default=720, help="Window height")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--no-robot", action="store_true", help="Disable robot (static view)")
    parser.add_argument("--no-dry-run", action="store_true", help="Enable real lever action")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("TROLLEY PROBLEM - COMPLETE SYSTEM")
    print("=" * 60)
    print(f"Window: {args.width}x{args.height}")
    print(f"Robot: {'ENABLED' if not args.no_robot else 'DISABLED'}")
    print(f"Lever action: {'REAL' if args.no_dry_run else 'DRY-RUN'}")
    print("=" * 60)
    print("Controls:")
    print("  Click START or press 's' - Begin scenario")
    print("  Click STOP or press 'x'  - Stop scenario")
    print("  Click RESTART or press 'r' - Restart")
    print("  Press 'q' - Quit")
    print("=" * 60)
    print()
    
    gui = TrolleyGUIComplete(
        window_width=args.width,
        window_height=args.height,
        conf_thresh=args.conf,
        use_robot=not args.no_robot,
        dry_run=not args.no_dry_run,
    )
    gui.run()


if __name__ == "__main__":
    main()
