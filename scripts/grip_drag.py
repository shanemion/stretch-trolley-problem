#!/usr/bin/env python3
"""
Extend 5 inches with gripper open, close gripper, retract 5 inches while closed, then open.

Run:
  python scripts/extend_grip_retract_release.py

Stop:
  Ctrl+C
"""

import time
import stretch_body.robot as stretch_robot

INCH = 0.0254
DIST_M = 5 * INCH  # 5 inches in meters

# Gripper commands per docs: -100 (closed) to +100 (open)
OPEN_POS = 100

# Gentle close tuning
CLOSE_START = 40         # start partially open
CLOSE_MIN = -100         # fully closed bound
CLOSE_STEP = 10          # step toward closed each iteration
STEP_SLEEP = 0.25
EFFORT_THRESHOLD = 0.25  # if available; otherwise loop just runs steps
MAX_STEPS = 20


def get_gripper_status(r):
    """Return (pos, effort/current) if available; else (None, None)."""
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
    time.sleep(0.2)  # Dynamixel acts immediately; small delay helps


def open_gripper(r):
    print("Opening gripper fully...")
    move_gripper(r, OPEN_POS)
    time.sleep(0.6)
    pos, eff = get_gripper_status(r)
    print(f"  after open: pos={pos} effort/current={eff}")


def close_gripper_gently(r):
    print("Closing gripper gently...")
    move_gripper(r, CLOSE_START)
    time.sleep(0.4)

    current = CLOSE_START
    for i in range(MAX_STEPS):
        pos, eff = get_gripper_status(r)

        if eff is not None and eff >= EFFORT_THRESHOLD:
            print(f"  stopping: effort/current={eff} >= {EFFORT_THRESHOLD} at pos={pos}")
            return

        current = max(CLOSE_MIN, current - CLOSE_STEP)
        print(f"  step {i+1}: commanding {current}")
        move_gripper(r, current)
        time.sleep(STEP_SLEEP)

    pos, eff = get_gripper_status(r)
    print(f"  close loop ended (max steps). final pos={pos} effort/current={eff}")


def main():
    r = stretch_robot.Robot()
    did_startup = r.startup()
    print(f"Robot connected to hardware: {did_startup}")

    try:
        # Home everything (calibrate) — simpler + more reliable than piecemeal
        print("Homing robot (arm + end_of_arm)...")
        r.home()
        time.sleep(1.5)

        # 1) Extend with gripper open
        open_gripper(r)

        print("Extending arm 5 inches...")
        r.arm.move_by(DIST_M)
        r.push_command()
        r.wait_command()  # waits for arm motion to complete

        # 2) Close
        close_gripper_gently(r)

        # 3) Retract while closed
        print("Retracting arm 5 inches while gripper closed...")
        r.arm.move_by(-DIST_M)
        r.push_command()
        r.wait_command()

        # 4) Open
        open_gripper(r)

        print("Done.")

    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl+C).")

    finally:
        r.stop()
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
