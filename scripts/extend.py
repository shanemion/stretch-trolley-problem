#!/usr/bin/env python3
"""
Move Stretch arm out ~1 inch and back.

Run:
    python scripts/extend.py

Safety:
- Make sure arm is clear
- Keep hand near runstop
"""

import time
import stretch_body.robot as robot


def main():
    print("Initializing Stretch robot...")
    r = robot.Robot()
    r.startup()

    try:
        print("Homing arm...")
        r.arm.home()
        r.push_command()
        time.sleep(2.0)

        # 1 inch ≈ 0.0254 meters
        distance_m = 0.0254

        print("Extending arm ~1 inch...")
        r.arm.move_by(distance_m)
        r.push_command()
        time.sleep(2.0)

        print("Retracting arm...")
        r.arm.move_by(-distance_m)
        r.push_command()
        time.sleep(2.0)

        print("Stowing robot...")
        r.stow()
        r.push_command()
        time.sleep(2.0)

    finally:
        print("Stopping robot.")
        r.stop()
        print("Done.")


if __name__ == "__main__":
    main()
