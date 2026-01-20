"""
Centralized configuration for trolley problem system.
"""

# Timing parameters
TOTAL_WAIT_S = 10.0  # Total wait time before decision
FINAL_WINDOW_S = 2.0  # Decision window (last N seconds)
SAMPLE_HZ = 5  # Sampling frequency during final window
SAMPLE_DT = 1.0 / SAMPLE_HZ  # Time between samples

# Perception parameters
CONF_THRESH = 0.35  # YOLO confidence threshold
MODEL_PATH = "scripts/yolov8n-pose.pt"  # YOLO model path (relative to repo root)
ROTATE_90_CLOCKWISE = True  # Camera rotation flag

# RealSense camera parameters
RS_WIDTH = 640
RS_HEIGHT = 480
RS_FPS = 30

# Trolley logic
DEFAULT_TRACK = "left"  # Default trolley track
# Decision rule: divert RIGHT if right_count < left_count

# Arm/gripper parameters (from working scripts)
INCH = 0.0254  # meters per inch
DIVERT_DISTANCE_M = 6 * INCH  # Arm extension distance (5 inches)

# Gripper parameters
OPEN_POS = 100  # Fully open
CLOSE_START = 40  # Start from partially open for gentle close
CLOSE_MIN = -100  # Fully closed
CLOSE_STEP = 10  # Step size toward closed
STEP_SLEEP = 0.25  # Sleep between gripper steps
EFFORT_THRESHOLD = 0.25  # Light pressure threshold (if available)
MAX_STEPS = 20  # Maximum steps for gentle close

# Safety defaults
DRY_RUN = True  # Default to safe mode (no hardware movement)
USE_MOCK_PERCEPTION = True  # Default to terminal input (for development)
