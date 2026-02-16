import cv2
import numpy as np
import time
from src.logic import FaceDetector, LandmarkProcessor, StressCalculator

def test_pipeline():
    print("--- Starting Integration Test ---")
    
    # 1. Load Components
    print("[1/4] Loading YOLO FaceDetector...")
    try:
        detector = FaceDetector("yolo11n.pt")
        print("   ✅ YOLO Loaded")
    except Exception as e:
        print(f"   ❌ YOLO Failed: {e}")
        return

    print("[2/4] Loading LandmarkProcessor...")
    try:
        landmarker = LandmarkProcessor()
        print("   ✅ Landmarker Loaded")
    except Exception as e:
        print(f"   ❌ Landmarker Failed: {e}")
        return

    print("[3/4] Loading StressCalculator (Hybrid)...")
    try:
        stress_calc = StressCalculator()
        print("   ✅ StressCalculator Loaded (Keras Model Integration)")
    except Exception as e:
        print(f"   ❌ StressCalculator Failed: {e}")
        return

    # 2. Capture Frame
    print("[4/4] Testing on Camera Frame...")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("   ⚠️ Camera not accessible. Creating dummy image.")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a fake face so YOLO might find something (unlikely but prevents crash)
        cv2.rectangle(frame, (200, 100), (400, 300), (200, 200, 200), -1)
    else:
        print("   ✅ Frame captured")

    # 3. Process
    face_box = detector.detect(frame)
    if face_box is None:
        print("   ⚠️ No face detected in test frame. Skipping full inference.")
    else:
        print(f"   ✅ Face Detected at {face_box}")
        x1, y1, x2, y2 = map(int, face_box)
        face_crop = frame[y1:y2, x1:x2]
        
        # Landmarks
        results = landmarker.process(face_crop)
        if results.face_blendshapes:
            print("   ✅ Landmarks Detected")
            
            # Hybrid Score
            scores = stress_calc.calculate_hybrid_score(face_crop, results.face_blendshapes[0])
            print("\n--- FINAL SCORES ---")
            print(f"🔴 Stress:     {scores['stress_score']:.1%}")
            print(f"🟢 Focus:      {scores['focused_score']:.1%}")
            print(f"🟠 Distracted: {scores['distracted_score']:.1%}")
            print("\nBreakdown:")
            print(scores['details'])
        else:
            print("   ⚠️ Face found but landmarks failed (face might be too small/blurry)")

    print("\n--- Test Complete ---")

if __name__ == "__main__":
    test_pipeline()
