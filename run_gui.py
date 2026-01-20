#!/usr/bin/env python3
"""
Launch the Trolley Problem GUI visualization.

This displays:
- Top Left: Confidence view (left/right aggregate confidence scores)
- Bottom Left: Camera feed with bounding boxes and center divider
- Right Side: Reserved for future use

Usage:
    python run_gui.py [OPTIONS]

Press 'q' to quit the GUI.
"""

import argparse
import sys

from trolley.gui import TrolleyGUI
from trolley import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trolley Problem GUI - Visual detection display",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Window width in pixels",
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Window height in pixels",
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
        "--no-rotate",
        action="store_true",
        help="Don't rotate camera feed 90 degrees",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 50)
    print("TROLLEY PROBLEM - DETECTION GUI")
    print("=" * 50)
    print(f"Window size: {args.width}x{args.height}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Model: {args.model_path}")
    print(f"Rotate camera: {not args.no_rotate}")
    print("=" * 50)
    print()
    
    try:
        gui = TrolleyGUI(
            window_width=args.width,
            window_height=args.height,
            model_path=args.model_path,
            conf_thresh=args.conf,
            rotate_90_clockwise=not args.no_rotate,
        )
        gui.run()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
