# Trolley Problem Implementation

## Overview

This system recreates the trolley problem scenario using the Stretch robot:
- **Perception**: Counts people on left vs right side
- **Decision**: Waits 10 seconds, uses last 2 seconds to decide
- **Action**: Diverts trolley to side with fewer people (default track is LEFT)

## Quick Start

### Basic Usage (Mock Perception + Dry Run)

```bash
python run_trolley.py
```

This will:
- Use terminal input for people counts (mock mode)
- Not execute any hardware actions (dry-run mode)
- Wait 10 seconds total
- Sample during last 2 seconds
- Make decision based on median counts

### With Real Hardware (Dry Run First!)

```bash
# Test with dry-run first
python run_trolley.py --no-dry-run --mock-perception

# Later: with real camera (when implemented)
python run_trolley.py --no-dry-run --no-mock-perception
```

## Command-Line Options

- `--dry-run` / `--no-dry-run`: Enable/disable hardware execution (default: dry-run)
- `--total-wait N`: Total wait time in seconds (default: 10.0)
- `--final-window N`: Decision window duration in seconds (default: 2.0)
- `--sample-hz N`: Sampling frequency during final window (default: 5.0)
- `--conf N`: Confidence threshold (default: 0.35)
- `--mock-perception` / `--no-mock-perception`: Use terminal input vs camera (default: mock)

## Example Session

```bash
$ python run_trolley.py

Initializing perception system...
[PeopleCounter] Initialized in MOCK mode (terminal input)
Dry-run mode: Robot hardware will not be used

=== TROLLEY PROBLEM DECISION WINDOW ===
Total wait: 10.0s
Warm-up period: 8.0s (not recording)
Decision window: last 2.0s (recording samples)
Sampling frequency: 5.0 Hz
Default track: LEFT
Dry run mode: True
========================================

[t=0.0s] Enter number of people on LEFT (int): 2
Enter number of people on RIGHT (int): 1
[t=0.0s] L=2 R=1 (warm-up, not recording)
[t=2.0s] Enter number of people on LEFT (int): 2
Enter number of people on RIGHT (int): 1
[t=2.0s] L=2 R=1 (warm-up, not recording)
...
[t=8.0s] Enter number of people on LEFT (int): 2
Enter number of people on RIGHT (int): 1
[t=8.0s] Sample: L=2 R=1 (recording)
[t=8.2s] Enter number of people on LEFT (int): 2
Enter number of people on RIGHT (int): 1
[t=8.2s] Sample: L=2 R=1 (recording)
...
[t=10.0s] Sample: L=2 R=1 (recording)

========================================
DECISION PHASE
========================================
Samples collected in final window: 10
Final counts (median): L=2 R=1
DECISION: DIVERT_RIGHT
  Reasoning: R=1 < L=2 -> divert to RIGHT track

[DRY RUN] === DIVERT LEVER SEQUENCE ===
...
```

## Architecture

- **`trolley/config.py`**: Centralized configuration
- **`trolley/perception.py`**: People counting (currently mock, future: RealSense+YOLO)
- **`trolley/actions.py`**: Lever pulling sequence
- **`trolley/decider.py`**: Decision logic (10s timer + final-window sampling)
- **`run_trolley.py`**: CLI entrypoint

## Decision Logic

1. **Warm-up period** (first 8 seconds): Counts are logged but not used
2. **Decision window** (last 2 seconds): Samples collected at 5 Hz
3. **Decision**: 
   - Compute median of final window samples
   - If `right_count < left_count`: **DIVERT_RIGHT**
   - Otherwise: **STAY_LEFT** (default track)
4. **Action**: Execute lever pull sequence if diverting

## Safety

- **Default is dry-run mode**: No hardware movement unless explicitly disabled
- **Mock perception by default**: Uses terminal input (safe for testing)
- **Always test with `--dry-run` first** before using real hardware

## Development Notes

- Reference scripts in `scripts/` are **not modified** (as requested)
- Mock perception uses same interface as future real implementation
- Easy to swap: change `use_mock=False` in `PeopleCounter` initialization
