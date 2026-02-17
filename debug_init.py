
import sys
import os
import time

print(f"Python Version: {sys.version}")

print("Step 1: Importing basic libs...")
try:
    import cv2
    import numpy as np
    print("Basic libs imported.")
except ImportError as e:
    print(f"FAILED to import basic libs: {e}")

print("Step 2: Importing TensorFlow...")
start = time.time()
try:
    import tensorflow as tf
    print(f"TensorFlow imported. Version: {tf.__version__}")
except ImportError as e:
    print(f"FAILED to import TensorFlow: {e}")
except Exception as e:
    print(f"CRASH during TensorFlow import: {e}")
print(f"TF Import Time: {time.time() - start:.2f}s")

print("Step 3: Importing Ultralytics (YOLO)...")
start = time.time()
try:
    from ultralytics import YOLO
    print("Ultralytics imported.")
except ImportError as e:
    print(f"FAILED to import Ultralytics: {e}")
except Exception as e:
    print(f"CRASH during Ultralytics import: {e}")
print(f"YOLO Import Time: {time.time() - start:.2f}s")

print("Step 4: Importing MediaPipe...")
start = time.time()
try:
    import mediapipe as mp
    print("MediaPipe imported.")
except ImportError as e:
    print(f"FAILED to import MediaPipe: {e}")
print(f"MP Import Time: {time.time() - start:.2f}s")

# --- Initialize Modules ---

print("\n--- Testing Initialization ---")

print("Init FaceDetector (YOLO)...")
try:
    from src.logic import FaceDetector
    fd = FaceDetector("yolo11n.pt")
    print("FaceDetector Initialized.")
except Exception as e:
    print(f"FAILED FaceDetector: {e}")

print("Init LandmarkProcessor (MediaPipe)...")
try:
    from src.logic import LandmarkProcessor
    lp = LandmarkProcessor()
    print("LandmarkProcessor Initialized.")
except Exception as e:
    print(f"FAILED LandmarkProcessor: {e}")

print("Init StressCalculator (Keras Model)...")
try:
    from src.logic import StressCalculator
    sc = StressCalculator()
    print("StressCalculator Initialized.")
except Exception as e:
    print(f"FAILED StressCalculator: {e}")

print("\nDIAGNOSTICS COMPLETE")
