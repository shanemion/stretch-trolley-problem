"""
Decision logic for trolley problem.

Implements 10-second wait with final-window sampling and decision making.
"""

import time
import statistics
from typing import Dict, Optional

from trolley.config import TOTAL_WAIT_S, FINAL_WINDOW_S, SAMPLE_DT
from trolley.actions import divert_lever


def median_or_fallback(values: list) -> int:
    """
    Compute median of values, or return last value if too few samples.
    
    Args:
        values: List of integer counts
        
    Returns:
        Median value, or last value if < 3 samples, or 0 if empty
    """
    if not values:
        return 0
    if len(values) < 3:
        return values[-1]
    return int(statistics.median(values))


def decide_and_act(
    counter,
    robot: Optional[object],
    config: Dict,
) -> Dict:
    """
    Main decision and action orchestrator.
    
    Waits TOTAL_WAIT_S seconds, samples counts during final FINAL_WINDOW_S,
    makes decision based on median counts, and executes action if needed.
    
    For mock mode (terminal input): asks for counts once, then waits.
    For camera mode: samples at regular intervals.
    
    Args:
        counter: PeopleCounter instance with get_counts() method
        robot: Stretch robot instance (or None if dry_run)
        config: Configuration dict with keys:
            - TOTAL_WAIT_S: float
            - FINAL_WINDOW_S: float
            - SAMPLE_DT: float
            - DIVERT_DISTANCE_M: float
            - DRY_RUN: bool
            - DEFAULT_TRACK: str
    
    Returns:
        Dict with keys:
            - decision: str ("DIVERT_RIGHT" or "STAY_LEFT")
            - left_count: int
            - right_count: int
            - samples_collected: int
            - executed: bool
    """
    total_wait = config.get("TOTAL_WAIT_S", TOTAL_WAIT_S)
    final_window = config.get("FINAL_WINDOW_S", FINAL_WINDOW_S)
    sample_dt = config.get("SAMPLE_DT", SAMPLE_DT)
    distance_m = config.get("DIVERT_DISTANCE_M", 5 * 0.0254)
    dry_run = config.get("DRY_RUN", True)
    default_track = config.get("DEFAULT_TRACK", "left")
    
    buffer_left = []
    buffer_right = []
    
    warmup_duration = total_wait - final_window
    
    print(f"\n=== TROLLEY PROBLEM DECISION WINDOW ===")
    print(f"Total wait: {total_wait:.1f}s")
    print(f"Warm-up period: {warmup_duration:.1f}s (not recording)")
    print(f"Decision window: last {final_window:.1f}s (recording samples)")
    print(f"Sampling frequency: {1.0/sample_dt:.1f} Hz")
    print(f"Default track: {default_track.upper()}")
    print(f"Dry run mode: {dry_run}")
    print("=" * 40)
    print()
    
    # Check if counter is in mock mode
    is_mock = getattr(counter, 'use_mock', False)
    
    if is_mock:
        # MOCK MODE: Ask for counts once, then wait with countdown
        print("MOCK MODE: Enter the scenario counts (this simulates what the camera would see)")
        print()
        
        # Get counts (this will block for terminal input)
        left_count, right_count, meta = counter.get_counts()
        
        print(f"\nScenario: L={left_count}, R={right_count}")
        print(f"\nStarting {total_wait:.0f}-second countdown...\n")
        
        # Countdown with progress updates
        start_time = time.time()
        end_time = start_time + total_wait
        final_start_time = start_time + warmup_duration
        last_print = -1
        
        while time.time() < end_time:
            elapsed = time.time() - start_time
            remaining = total_wait - elapsed
            second = int(elapsed)
            
            # Print update once per second
            if second > last_print:
                last_print = second
                in_final = elapsed >= warmup_duration
                phase = "(RECORDING)" if in_final else "(warm-up)"
                
                # During final window, record samples
                if in_final:
                    buffer_left.append(left_count)
                    buffer_right.append(right_count)
                
                print(f"[t={elapsed:5.1f}s] L={left_count} R={right_count} {phase} | {remaining:.1f}s remaining")
            
            time.sleep(0.1)
        
        print(f"\n[t={total_wait:.1f}s] Time's up!")
        
    else:
        # CAMERA MODE: Sample at regular intervals
        print("CAMERA MODE: Sampling from camera...")
        print()
        
        start_time = time.time()
        end_time = start_time + total_wait
        final_start_time = start_time + warmup_duration
        next_sample_time = start_time
        last_progress_log = start_time
        
        while time.time() < end_time:
            current_time = time.time()
            
            if current_time >= next_sample_time:
                # Get counts from camera (non-blocking)
                left_count, right_count, meta = counter.get_counts()
                
                sample_time = time.time()
                elapsed = sample_time - start_time
                in_final_window = sample_time >= final_start_time
                
                if in_final_window:
                    buffer_left.append(left_count)
                    buffer_right.append(right_count)
                    print(f"[t={elapsed:.1f}s] Sample: L={left_count} R={right_count} (recording)")
                else:
                    if sample_time - last_progress_log >= 2.0:
                        print(f"[t={elapsed:.1f}s] L={left_count} R={right_count} (warm-up)")
                        last_progress_log = sample_time
                
                next_sample_time = sample_time + sample_dt
            
            time.sleep(0.01)
        
        print(f"\n[t={total_wait:.1f}s] Sampling complete.")
    
    # Decision phase
    print()
    print("=" * 40)
    print("DECISION PHASE")
    print("=" * 40)
    
    # Compute final counts using median
    L_final = median_or_fallback(buffer_left)
    R_final = median_or_fallback(buffer_right)
    
    samples_collected = len(buffer_left)
    
    print(f"Samples collected in final window: {samples_collected}")
    print(f"Raw buffer L: {buffer_left}")
    print(f"Raw buffer R: {buffer_right}")
    print(f"Final counts (median): L={L_final} R={R_final}")
    print()
    
    # Decision rule: divert to side with fewer people
    # Default track is LEFT, so divert RIGHT if right < left
    if R_final < L_final:
        decision = "DIVERT_RIGHT"
        execute_divert = True
        print(f"DECISION: {decision}")
        print(f"  Reasoning: R={R_final} < L={L_final} -> divert to RIGHT track (fewer people)")
    else:
        decision = "STAY_LEFT"
        execute_divert = False
        if R_final == L_final:
            print(f"DECISION: {decision}")
            print(f"  Reasoning: R={R_final} == L={L_final} -> stay on default LEFT track (tie)")
        else:
            print(f"DECISION: {decision}")
            print(f"  Reasoning: R={R_final} > L={L_final} -> stay on default LEFT track (fewer people)")
    
    print()
    
    # Execute action if needed
    executed = False
    if execute_divert:
        print("Executing divert sequence...")
        # In dry-run mode, we can still call divert_lever with dry_run=True
        # It will just print the steps without moving hardware
        divert_lever(robot, distance_m, dry_run=dry_run)
        executed = not dry_run  # Only mark as executed if not dry-run
    else:
        print("No action needed (staying on default track).")
    
    print()
    print("=" * 40)
    print("DECISION COMPLETE")
    print("=" * 40)
    print()
    
    return {
        "decision": decision,
        "left_count": L_final,
        "right_count": R_final,
        "samples_collected": samples_collected,
        "executed": executed,
    }
