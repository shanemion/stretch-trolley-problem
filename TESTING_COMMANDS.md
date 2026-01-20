# Trolley Problem Testing Commands

Quick reference for all testing commands and options.

---

## GUI Visualization (with Head Scanning)

### Launch Detection GUI with Head Scanning
```bash
python run_gui.py
```
- Robot pans head LEFT and RIGHT to expand field of view
- LEFT view = people on left track
- RIGHT view = people on right track
- Shows both camera views side by side
- Displays confidence scores for each side
- Press 'q' to quit

### GUI without Robot (Static View)
```bash
python run_gui.py --no-robot
```
- No head movement (camera stays centered)
- Splits by nose position (like before)
- Use if robot is unavailable

### GUI Options
```bash
python run_gui.py --width 1920 --height 1080  # Larger window
python run_gui.py --conf 0.5                   # Higher confidence threshold
python run_gui.py --no-rotate                  # Don't rotate camera feed
python run_gui.py --no-robot                   # Disable head scanning
```

---

## Basic Commands

### Dry-Run with Mock Perception (Safest - No Hardware)
```bash
python run_trolley.py --dry-run
```
- Uses terminal input for people counts
- No hardware movement
- Good for testing decision logic

### Dry-Run with Real Camera (Static View)
```bash
python run_trolley.py --no-mock-perception --dry-run
```
- Uses RealSense camera + YOLO detection
- Splits by nose position (static view)
- No hardware movement
- Good for testing perception pipeline

### Dry-Run with Camera + Head Scanning (Wider FOV)
```bash
python run_trolley.py --no-mock-perception --scanning --dry-run
```
- Robot pans head left and right to expand FOV
- LEFT view = people on left track
- RIGHT view = people on right track
- No lever action (dry-run)

### Real Hardware with Mock Perception (Test Actions)
```bash
python run_trolley.py --no-dry-run --mock-perception
```
- Uses terminal input for counts
- Executes lever pull sequence on robot
- Good for testing action sequence

### Full System Test (Real Camera + Real Hardware)
```bash
python run_trolley.py --no-mock-perception --no-dry-run
```
- Uses RealSense camera + YOLO detection
- Executes lever pull sequence on robot
- **Full end-to-end test**

## Timing Options

### Shorter Wait Times (Faster Testing)
```bash
python run_trolley.py --dry-run --total-wait 5 --final-window 1
```
- 5 second total wait
- 1 second decision window
- Useful for quick iteration

### Custom Timing
```bash
python run_trolley.py --dry-run --total-wait 15 --final-window 3
```
- 15 second total wait
- 3 second decision window

## Detection Parameters

### Adjust Confidence Threshold
```bash
python run_trolley.py --no-mock-perception --dry-run --conf 0.5
```
- Higher threshold = fewer false positives
- Lower threshold = more detections (may include false positives)
- Default: 0.35

### Custom Model Path
```bash
python run_trolley.py --no-mock-perception --dry-run --model-path path/to/model.pt
```
- Use different YOLO model file
- Default: `scripts/yolov8n-pose.pt`

## Sampling Parameters

### Adjust Sampling Frequency
```bash
python run_trolley.py --dry-run --sample-hz 10
```
- Higher frequency = more samples during final window
- Default: 5 Hz (samples every 0.2s)
- Note: Higher frequency may be slower with camera

## Common Testing Scenarios

### 1. Quick Logic Test (Mock, Fast)
```bash
python run_trolley.py --dry-run --total-wait 3 --final-window 1
```
- Fastest way to test decision logic
- Enter counts manually

### 2. Camera Detection Test (No Hardware)
```bash
python run_trolley.py --no-mock-perception --dry-run --total-wait 5
```
- Test camera detection and confidence logging
- Verify people are detected correctly

### 3. Action Sequence Test (Mock Counts, Real Hardware)
```bash
python run_trolley.py --no-dry-run --mock-perception --total-wait 5
```
- Test lever pull sequence
- Enter counts manually, robot executes action

### 4. Full System Test (Everything Real)
```bash
python run_trolley.py --no-mock-perception --no-dry-run
```
- Complete end-to-end test
- Camera detects, robot acts

## All Available Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--dry-run` | Don't execute hardware actions | `True` |
| `--no-dry-run` | Execute hardware actions | - |
| `--mock-perception` | Use terminal input for counts | `True` |
| `--no-mock-perception` | Use RealSense camera | - |
| `--scanning` | Enable head scanning for wider FOV | `False` |
| `--no-scanning` | Disable head scanning (static view) | - |
| `--total-wait N` | Total wait time (seconds) | `10.0` |
| `--final-window N` | Decision window duration (seconds) | `2.0` |
| `--sample-hz N` | Sampling frequency (Hz) | `5.0` |
| `--conf N` | Confidence threshold | `0.35` |
| `--model-path PATH` | Path to YOLO model | `scripts/yolov8n-pose.pt` |

## Safety Checklist

Before running with `--no-dry-run`:

- [ ] Robot is free (`free` command or `stretch_free_robot_process.py`)
- [ ] Arm path is clear
- [ ] Hand near runstop button
- [ ] Tested with `--dry-run` first
- [ ] Camera view is clear (if using `--no-mock-perception`)

## Troubleshooting

### "Another process is already using Stretch"
```bash
free  # or stretch_free_robot_process.py
ps aux | grep -E "stretch|python" | grep -v grep  # Check for lingering processes
```

### Camera not detected
- Check USB connection
- Verify RealSense is powered
- Try: `rs-enumerate-devices` (if available)

### Model file not found
- Ensure `scripts/yolov8n-pose.pt` exists
- Or specify custom path with `--model-path`

## Example Full Command

```bash
# Test with real camera, 5s wait, 1s window, higher confidence, no hardware
python run_trolley.py \
  --no-mock-perception \
  --dry-run \
  --total-wait 5 \
  --final-window 1 \
  --conf 0.5 \
  --sample-hz 10
```

## Decision Logic

- **Default track**: LEFT
- **Divert condition**: RIGHT count < LEFT count → DIVERT_RIGHT
- **Otherwise**: STAY_LEFT (no action)
- **Ties**: STAY_LEFT (no action)
