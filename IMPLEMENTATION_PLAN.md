# Trolley Problem Implementation Plan

## Overview
Build an orchestrator that combines perception (people counting) with action (lever pulling) to recreate the trolley problem scenario.

## Architecture

### Module Structure
```
stretch-trolley-problem/
├── trolley/
│   ├── __init__.py          # Package marker
│   ├── config.py            # Centralized configuration
│   ├── perception.py        # PeopleCounter wrapper (from left_right.py)
│   ├── actions.py           # Lever divert action (from grip_drag.py)
│   └── decider.py           # 10s timer + final-window decision logic
├── run_trolley.py           # CLI entrypoint
└── scripts/                 # Reference scripts (DO NOT MODIFY)
```

## Component Details

### 1. `trolley/config.py`
**Purpose**: Centralized configuration constants

**Key Parameters**:
- `TOTAL_WAIT_S = 10.0` - Total wait time before decision
- `FINAL_WINDOW_S = 2.0` - Decision window (last N seconds)
- `SAMPLE_HZ = 5` - Sampling frequency during final window
- `CONF_THRESH = 0.35` - YOLO confidence threshold
- `DEFAULT_TRACK = "left"` - Default trolley track
- `DIVERT_DISTANCE_M = 5 * 0.0254` - Arm extension distance (5 inches)
- `DRY_RUN = True` - Default to safe mode

**Gripper Parameters** (from working scripts):
- `OPEN_POS = 100`
- `CLOSE_START = 40`
- `CLOSE_MIN = -100`
- `CLOSE_STEP = 10`
- `STEP_SLEEP = 0.25`
- `EFFORT_THRESHOLD = 0.25`
- `MAX_STEPS = 20`

**Camera Parameters**:
- `MODEL_PATH = "scripts/yolov8n-pose.pt"`
- `ROTATE_90_CLOCKWISE = True`
- `RS_WIDTH = 640`
- `RS_HEIGHT = 480`
- `RS_FPS = 30`

### 2. `trolley/perception.py`
**Purpose**: Wrap working left/right detection into reusable class

**Class: `PeopleCounter`**
- **`__init__(conf_thresh, model_path, rotate_90_clockwise)`**
  - Initialize RealSense pipeline
  - Load YOLO model
  - Set up image geometry

- **`get_counts() -> tuple[int, int, dict]`**
  - Returns: `(left_count, right_count, metadata)`
  - Metadata includes:
    - `timestamp`: current time
    - `total_detections`: raw detection count
    - `confidences`: list of confidences per person
    - `sample_valid`: bool (whether frame was valid)

- **`cleanup()`**
  - Stop RealSense pipeline

**Key Points**:
- Headless (no cv2.imshow)
- Extract core logic from `left_right.py`
- Return counts, not full detection lists (for efficiency)

### 3. `trolley/actions.py`
**Purpose**: Wrap lever pulling sequence into reusable function

**Function: `divert_lever(robot, distance_m, dry_run=False) -> None`**

**Sequence** (from `grip_drag.py`):
1. Robot startup + home (assume already done by caller)
2. Open gripper to `OPEN_POS` (100)
3. Wait for gripper to settle
4. Extend arm by `distance_m` (5 inches)
5. Wait for arm motion to complete
6. Close gripper gently (stepped approach)
7. Retract arm by `distance_m` (5 inches)
8. Wait for arm motion to complete
9. Open gripper to `OPEN_POS` (100)

**Helper Functions**:
- `get_gripper_status(robot) -> tuple[float|None, float|None]`
- `move_gripper(robot, pos) -> None`
- `open_gripper(robot) -> None`
- `close_gripper_gently(robot) -> None`

**Dry Run Mode**:
- Print each step without executing hardware commands
- Use `print("[DRY RUN] ...")` prefix

**Key Points**:
- Assume arm height is correct (no lift motion)
- Use exact gripper parameters from working scripts
- Robust error handling (try/except around hardware calls)

### 4. `trolley/decider.py`
**Purpose**: Orchestrate 10s wait + final-window decision

**Function: `decide_and_act(counter, robot, config) -> dict`**

**Algorithm**:
```
start_time = time.time()
buffer_left = []
buffer_right = []

# Phase 1: Warm-up period (first 8 seconds)
# - Log counts but don't record
# - Optional: print progress every 2 seconds

# Phase 2: Final window (last 2 seconds)
# - Sample at SAMPLE_HZ
# - Append to buffers

# Phase 3: Decision
# - Compute median of final window samples
# - Apply decision rule
# - Execute action if needed
```

**Decision Logic**:
```python
L_final = median(buffer_left) if buffer_left else 0
R_final = median(buffer_right) if buffer_right else 0

# Guardrails
if len(buffer_left) < 3 or len(buffer_right) < 3:
    # Fallback: use last observed count
    L_final = buffer_left[-1] if buffer_left else 0
    R_final = buffer_right[-1] if buffer_right else 0

# Decision rule: divert to side with fewer people
# Default track is LEFT, so divert RIGHT if right < left
if R_final < L_final:
    decision = "DIVERT_RIGHT"
    execute_divert = True
else:
    decision = "STAY_LEFT"
    execute_divert = False
```

**Logging**:
- Progress updates during warm-up: `[t=2.3s] L=2 R=1 (warm-up, not recording)`
- Final window start: `[t=8.0s] Entering decision window (last 2.0s)...`
- Sample logs: `[t=8.2s] Sample: L=2 R=1`
- Final decision: `[t=10.0s] FINAL DECISION: L_final=2 R_final=1 -> DIVERT_RIGHT`
- Action execution: `Executing divert sequence...`

**Return Value**:
```python
{
    "decision": "DIVERT_RIGHT" | "STAY_LEFT",
    "left_count": int,
    "right_count": int,
    "samples_collected": int,
    "executed": bool
}
```

### 5. `run_trolley.py`
**Purpose**: CLI entrypoint with argument parsing

**Arguments**:
- `--dry-run` (default: True) - Don't execute hardware actions
- `--total-wait` (default: 10.0) - Total wait time
- `--final-window` (default: 2.0) - Decision window duration
- `--sample-hz` (default: 5) - Sampling frequency
- `--conf` (default: 0.35) - Confidence threshold
- `--model-path` (default: "scripts/yolov8n-pose.pt") - YOLO model path

**Execution Flow**:
1. Parse arguments
2. Create config object
3. Initialize `PeopleCounter`
4. Initialize robot (if not dry-run)
5. Call `decide_and_act(counter, robot, config)`
6. Print final results
7. Cleanup (counter.cleanup(), robot.stop())

## Implementation Steps

### Step 1: Create Module Structure
- Create `trolley/` directory
- Create `__init__.py` (empty or minimal exports)
- Create `config.py` with all constants

### Step 2: Implement `perception.py`
- Extract detection logic from `left_right.py`
- Create `PeopleCounter` class
- Test independently: `python -c "from trolley.perception import PeopleCounter; c = PeopleCounter(); print(c.get_counts())"`

### Step 3: Implement `actions.py`
- Extract action sequence from `grip_drag.py`
- Create `divert_lever()` function
- Test with `dry_run=True` first
- Test with real hardware (carefully!)

### Step 4: Implement `decider.py`
- Implement timer logic
- Implement buffer management
- Implement decision rule
- Test with mock counter (no hardware)

### Step 5: Implement `run_trolley.py`
- Add argument parsing (argparse)
- Wire everything together
- Add comprehensive logging

### Step 6: Integration Testing
- Stage A: Perception only (`--dry-run`, verify counts)
- Stage B: Action only (test `divert_lever` standalone)
- Stage C: Full integration (`--dry-run False`)

## Safety Considerations

1. **Dry Run Default**: Always default to `dry_run=True`
2. **Robot Locking**: Check for existing processes before startup
3. **Graceful Shutdown**: Handle Ctrl+C properly
4. **Error Handling**: Wrap hardware calls in try/except
5. **Logging**: Clear console output for demos

## Testing Checklist

- [ ] Perception counts people correctly (left/right split)
- [ ] Final window only uses last 2 seconds
- [ ] Median calculation works correctly
- [ ] Decision rule: divert when right < left
- [ ] Dry-run mode prints but doesn't move hardware
- [ ] Lever pull sequence matches `grip_drag.py` behavior
- [ ] Runs headless over SSH (no display errors)
- [ ] Handles edge cases (no people, unstable detections)

## Questions to Clarify

1. **Divert direction**: If right has fewer people, pull lever right? (Assuming YES)
2. **Confidence threshold**: Use 0.35 or 0.25? (Using 0.35 as specified)
3. **Model path**: Relative to repo root or scripts/? (Using "scripts/yolov8n-pose.pt")
4. **Arm position**: Should we verify arm is at correct height, or assume? (Assuming as specified)

## Next Steps

1. Review this plan
2. Answer any clarifying questions
3. Begin implementation starting with Step 1
