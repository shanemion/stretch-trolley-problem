"""
Lever pulling actions for trolley problem.

Implements the divert sequence: open gripper → extend → close → retract → open.
"""

import time
from typing import Optional

from trolley.config import (
    OPEN_POS,
    CLOSE_START,
    CLOSE_MIN,
    CLOSE_STEP,
    STEP_SLEEP,
    EFFORT_THRESHOLD,
    MAX_STEPS,
)

# Import stretch_body only when needed (not at top level to allow dry-run without hardware)
def _get_robot_module():
    import stretch_body.robot as robot_module
    return robot_module


def get_gripper_status(r) -> tuple:
    """
    Return (pos, effort/current) if available; else (None, None).
    
    Args:
        r: Stretch robot instance
        
    Returns:
        Tuple of (position, effort/current) or (None, None) if unavailable
    """
    try:
        j = r.end_of_arm.get_joint("stretch_gripper")
        st = j.status
        pos = st.get("pos", None)
        eff = st.get("effort", st.get("current", None))
        return pos, eff
    except Exception as e:
        print(f"[debug] Could not read gripper joint status: {e}")
        return None, None


def move_gripper(r, pos: float) -> None:
    """
    Move gripper to specified position.
    
    Args:
        r: Stretch robot instance
        pos: Gripper position (-100 to +100)
    """
    r.end_of_arm.move_to("stretch_gripper", pos)
    time.sleep(0.2)  # Dynamixel acts immediately; small delay helps


def open_gripper(r, dry_run: bool = False) -> None:
    """
    Open gripper fully.
    
    Args:
        r: Stretch robot instance
        dry_run: If True, only print action without executing
    """
    if dry_run:
        print("[DRY RUN] Opening gripper fully...")
        print(f"[DRY RUN]   Command: move_to('stretch_gripper', {OPEN_POS})")
        time.sleep(0.1)  # Simulate delay
        return
    
    print("Opening gripper fully...")
    move_gripper(r, OPEN_POS)
    time.sleep(0.6)
    pos, eff = get_gripper_status(r)
    print(f"  after open: pos={pos} effort/current={eff}")


def close_gripper_gently(r, dry_run: bool = False) -> None:
    """
    Close gripper gently using stepped approach.
    
    Args:
        r: Stretch robot instance
        dry_run: If True, only print action without executing
    """
    if dry_run:
        print("[DRY RUN] Closing gripper gently...")
        print(f"[DRY RUN]   Start from {CLOSE_START}, step toward {CLOSE_MIN}")
        print(f"[DRY RUN]   Step size: {CLOSE_STEP}, max steps: {MAX_STEPS}")
        time.sleep(0.1)  # Simulate delay
        return
    
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


def divert_lever(
    r: Optional[object],
    distance_m: float,
    dry_run: bool = False,
) -> None:
    """
    Execute lever pulling sequence to divert trolley.
    
    Sequence:
    1. Open gripper
    2. Extend arm by distance_m
    3. Close gripper gently
    4. Retract arm by distance_m
    5. Open gripper
    
    Args:
        r: Stretch robot instance (can be None if dry_run=True)
        distance_m: Distance to extend/retract arm (meters)
        dry_run: If True, only print actions without executing
        
    Note:
        Assumes robot is already started, homed, and at correct height.
        When dry_run=True, robot can be None.
    """
    # Validate: if not dry_run, robot must be provided
    if not dry_run and r is None:
        raise ValueError("Robot instance required when dry_run=False")
    
    if dry_run:
        print("\n[DRY RUN] === DIVERT LEVER SEQUENCE ===")
        print(f"[DRY RUN] Distance: {distance_m:.4f} m ({distance_m / 0.0254:.1f} inches)")
    
    # 1) Open gripper
    open_gripper(r, dry_run=dry_run)
    
    if not dry_run:
        time.sleep(0.5)
    
    # 2) Extend arm
    if dry_run:
        print(f"[DRY RUN] Extending arm {distance_m:.4f} m...")
    else:
        print(f"Extending arm {distance_m:.4f} m...")
        r.arm.move_by(distance_m)
        r.push_command()
        r.wait_command()  # Wait for arm motion to complete
    
    if not dry_run:
        time.sleep(0.5)
    
    # 3) Close gripper gently
    close_gripper_gently(r, dry_run=dry_run)
    
    if not dry_run:
        time.sleep(0.5)
    
    # 4) Retract arm
    if dry_run:
        print(f"[DRY RUN] Retracting arm {distance_m:.4f} m...")
    else:
        print(f"Retracting arm {distance_m:.4f} m...")
        r.arm.move_by(-distance_m)
        r.push_command()
        r.wait_command()
    
    if not dry_run:
        time.sleep(0.5)
    
    # 5) Open gripper
    open_gripper(r, dry_run=dry_run)
    
    if dry_run:
        print("[DRY RUN] === DIVERT SEQUENCE COMPLETE ===\n")
    else:
        print("Divert sequence complete.")
