#!/usr/bin/env python3
"""
Trolley Problem CLI Entrypoint

Orchestrates perception and action to recreate the trolley problem scenario.
"""

import argparse
import sys
import time

from trolley import config
from trolley.perception import PeopleCounter
from trolley.decider import decide_and_act


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Trolley Problem: Count people and decide whether to divert trolley",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=config.DRY_RUN,
        help="Don't execute hardware actions (default: %(default)s)",
    )
    parser.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Execute hardware actions (opposite of --dry-run)",
    )
    
    parser.add_argument(
        "--total-wait",
        type=float,
        default=config.TOTAL_WAIT_S,
        help="Total wait time before decision (seconds)",
    )
    
    parser.add_argument(
        "--final-window",
        type=float,
        default=config.FINAL_WINDOW_S,
        help="Decision window duration (last N seconds)",
    )
    
    parser.add_argument(
        "--sample-hz",
        type=float,
        default=config.SAMPLE_HZ,
        help="Sampling frequency during final window (Hz)",
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=config.CONF_THRESH,
        help="Confidence threshold for detection",
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=config.MODEL_PATH,
        help="Path to YOLO model file",
    )
    
    parser.add_argument(
        "--mock-perception",
        action="store_true",
        default=config.USE_MOCK_PERCEPTION,
        help="Use terminal input for perception (default: %(default)s)",
    )
    parser.add_argument(
        "--no-mock-perception",
        dest="mock_perception",
        action="store_false",
        help="Use real camera for perception (opposite of --mock-perception)",
    )
    
    return parser.parse_args()


def build_config(args) -> dict:
    """Build configuration dict from arguments."""
    sample_dt = 1.0 / args.sample_hz
    
    return {
        "TOTAL_WAIT_S": args.total_wait,
        "FINAL_WINDOW_S": args.final_window,
        "SAMPLE_DT": sample_dt,
        "DIVERT_DISTANCE_M": config.DIVERT_DISTANCE_M,
        "DRY_RUN": args.dry_run,
        "DEFAULT_TRACK": config.DEFAULT_TRACK,
    }


def main():
    """Main entrypoint."""
    args = parse_args()
    
    # Validate arguments
    if args.final_window >= args.total_wait:
        print("Error: --final-window must be less than --total-wait")
        sys.exit(1)
    
    if args.sample_hz <= 0:
        print("Error: --sample-hz must be positive")
        sys.exit(1)
    
    # Build config
    cfg = build_config(args)
    
    # Initialize perception
    print("Initializing perception system...")
    counter = PeopleCounter(
        conf_thresh=args.conf,
        model_path=args.model_path,
        rotate_90_clockwise=config.ROTATE_90_CLOCKWISE,
        use_mock=args.mock_perception,
    )
    
    # Initialize robot (if not dry-run)
    r = None
    if not args.dry_run:
        print("Initializing robot...")
        try:
            import stretch_body.robot as robot_module
            r = robot_module.Robot()
            did_startup = r.startup()
            if not did_startup:
                print("Error: Failed to connect to robot hardware")
                sys.exit(1)
            
            print("Homing robot...")
            r.home()
            time.sleep(1.5)
            print("Robot ready.")
        except ImportError as e:
            print(f"Error: Could not import stretch_body: {e}")
            print("Make sure you're running on the Stretch robot with the correct environment.")
            sys.exit(1)
    else:
        print("Dry-run mode: Robot hardware will not be used")
    
    print()
    
    try:
        # Run decision and action
        result = decide_and_act(counter, r, cfg)
        
        # Print final results
        print("\n" + "=" * 40)
        print("FINAL RESULTS")
        print("=" * 40)
        print(f"Decision: {result['decision']}")
        print(f"Left count: {result['left_count']}")
        print(f"Right count: {result['right_count']}")
        print(f"Samples collected: {result['samples_collected']}")
        print(f"Action executed: {result['executed']}")
        print("=" * 40)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        print("\nCleaning up...")
        counter.cleanup()
        if r is not None:
            r.stop()
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
