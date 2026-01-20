#!/usr/bin/env python3
"""
Gripper-only test (per official Stretch Body docs).

- Gripper command range: -100 (fully closed) to +100 (fully open)
- Uses move_to('stretch_gripper', value)

Run:
  python scripts/gripper_only.py
Stop:
  Ctrl+C
"""

import time
import stretch_body.robot as stretch_robot

# Tuneables
OPEN_POS = 100          # fully open
CLOSE_START = 40        # start from partially open for gentle close
CLOSE_MIN = -100        # fully closed (we won't necessarily go this far)
CLOSE_STEP = 10         # step size toward closed
STEP_SLEEP = 0.25
EFFORT_THRESHOLD = 0.25 # "light pressure" threshold (if available)
MAX_STEPS = 20


def get_gripper_status(r):
    """
    Returns (pos, effort_or_current) if available.
    Note: status structure can differ slightly; we print helpful debug if missing.
    """
    # Many builds: end_of_arm.get_joint('stretch_gripper').status
    try:
        j = r.end_of_arm.get_joint("stretch_gripper")
        st = j.status
        pos = st.get("pos", None)
        eff = st.get("effort", st.get("current", None))
        return pos, eff
    except Exception as e:
        print(f"[debug] Could not read gripper joint status: {e}")
        return None, None


def move_gripper(r, pos):
    r.end_of_arm.move_to("stretch_gripper", pos)
    # Dynamixel commands execute immediately, but giving time helps
    time.sleep(0.2)


def main():
    r = stretch_robot.Robot()
    did_startup = r.startup()
    print(f"Robot connected to hardware: {did_startup}")

    try:
        print("Homing robot (safe to do; ensures calibration)...")
        r.home()
        time.sleep(1.0)

        print("Opening gripper fully...")
        move_gripper(r, OPEN_POS)
        time.sleep(1.0)
        pos, eff = get_gripper_status(r)
        print(f"  after open: pos={pos} effort/current={eff}")

        print("Closing gently until light pressure (if measurable)...")
        # Start from a partially open position so we don't slam shut
        move_gripper(r, CLOSE_START)
        time.sleep(0.5)

        current = CLOSE_START
        for i in range(MAX_STEPS):
            pos, eff = get_gripper_status(r)

            # If effort/current is available, stop when it increases past threshold
            if eff is not None and eff >= EFFORT_THRESHOLD:
                print(f"  stopping: effort/current={eff} >= {EFFORT_THRESHOLD} at pos={pos}")
                break

            # Step more closed
            current = max(CLOSE_MIN, current - CLOSE_STEP)
            print(f"  step {i+1}: commanding {current}")
            move_gripper(r, current)
            time.sleep(STEP_SLEEP)

        print("Re-opening gripper fully...")
        move_gripper(r, OPEN_POS)
        time.sleep(1.0)
        pos, eff = get_gripper_status(r)
        print(f"  after re-open: pos={pos} effort/current={eff}")

        print("Done.")

    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl+C).")

    finally:
        r.stop()
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
