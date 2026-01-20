
import sys
import os
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("sys.path:", sys.path)

try:
    import cv2
    print("cv2 imported:", cv2.__file__)
except ImportError as e:
    print("cv2 import failed:", e)

try:
    import pygame
    print("pygame imported:", pygame.__file__)
except ImportError as e:
    print("pygame import failed:", e)
