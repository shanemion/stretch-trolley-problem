"""
People counting module for trolley problem.

Currently implements mock terminal-input mode for development.
Future: will swap in RealSense + YOLO implementation.
"""

import time
from typing import Tuple, Dict


class PeopleCounter:
    """
    Counts people on left vs right side of camera view.
    
    Currently uses terminal input (mock mode).
    Future: will use RealSense camera + YOLO pose detection.
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
            conf_thresh: Confidence threshold (for future YOLO use)
            model_path: Path to YOLO model (for future use)
            rotate_90_clockwise: Camera rotation flag (for future use)
            use_mock: If True, use terminal input; if False, use camera (future)
        """
        self.conf_thresh = conf_thresh
        self.model_path = model_path
        self.rotate_90_clockwise = rotate_90_clockwise
        self.use_mock = use_mock
        
        if not use_mock:
            # Future: initialize RealSense + YOLO here
            raise NotImplementedError("Real camera mode not yet implemented")
        
        print("[PeopleCounter] Initialized in MOCK mode (terminal input)")
    
    def get_counts(self) -> Tuple[int, int, Dict]:
        """
        Get current left and right person counts.
        
        Returns:
            Tuple of (left_count, right_count, metadata_dict)
            
        Metadata dict contains:
            - timestamp: float (current time)
            - total_detections: int
            - confidences: list (empty for mock mode)
            - sample_valid: bool
            - source: str ("terminal_input" for mock)
        """
        if self.use_mock:
            return self._get_counts_mock()
        else:
            # Future: return self._get_counts_camera()
            raise NotImplementedError("Real camera mode not yet implemented")
    
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
            "confidences": [],  # Empty for mock mode
            "sample_valid": True,
            "source": "terminal_input",
        }
        
        return left_count, right_count, meta
    
    def cleanup(self) -> None:
        """Clean up resources (camera, model, etc.)."""
        if not self.use_mock:
            # Future: stop RealSense pipeline
            pass
        print("[PeopleCounter] Cleanup complete")
