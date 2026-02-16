import cv2
import time
import numpy as np
import pandas as pd
from datetime import datetime
from threading import Lock
from collections import deque

# Placeholder for YOLO and MediaPipe - imports will be actual in final
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceDetector:
    def __init__(self, model_path="yolo11n.pt"):
        # Try to load requested model, fallback to standard YOLOv8n if missing
        try:
            self.model = YOLO(model_path)
            print(f"Loaded {model_path}")
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
            print("Falling back to yolov8n.pt...")
            self.model = YOLO("yolov8n.pt")
    
    def detect(self, frame):
        """
        Detects faces in the frame.
        Returns the bounding box of the largest face found (or None).
        Format: (x1, y1, x2, y2)
        """
        results = self.model(frame, verbose=False, conf=0.5)
        boxes = []
        for result in results:
            for box in result.boxes:
                # Class 0 is usually 'person' in standard COCO. 
                # If using a face-specific model, check class ID if needed.
                # Assuming yolo26n.pt behaves like standard YOLO or is face specific 
                # we'll just take the box.
                xyxy = box.xyxy[0].cpu().numpy()
                boxes.append(xyxy)
        
        if not boxes:
            return None
            
        # Return the largest box (assuming it's the main user)
        # Area = (x2-x1) * (y2-y1)
        largest_box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        return largest_box

class LandmarkProcessor:
    def __init__(self):
        self.base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        self.options = vision.FaceLandmarkerOptions(
            base_options=self.base_options,
            output_face_blendshapes=True,
            num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(self.options)

    def process(self, frame_crop):
        """
        Extracts landmarks and blendshapes from a cropped face image.
        """
        # MediaPipe expects RGB
        rgb_image = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        detection_result = self.detector.detect(mp_image)
        return detection_result

from src.emotion_model import EmotionModel

class StressCalculator:
    def __init__(self):
        self.history = []
        self.history_limit = 150 # ~5 seconds at 30fps
        # Load Keras Model
        self.emotion_model = EmotionModel()
        
        # New: Temporal Smoothing Buffer (1 second window)
        self.score_buffer = deque(maxlen=30)
        
        # New: Frame Skipping
        self.frame_count = 0
        self.skip_frames = 5 # Run Keras every 5th frame
        self.last_emotions = None
        
        # New: Blink Detection (Liveness)
        self.last_blink_time = time.time()
        self.blink_detected = False
        self.EAR_THRESHOLD = 0.25 # Tune based on testing, usually 0.2-0.3
        
    def calculate_ear(self, landmarks):
        """
        Calculates Eye Aspect Ratio (EAR) for blink detection.
        """
        # Indices for Left and Right Eye (MediaPipe 468/478 mesh)
        # Left Eye: 33 (p1), 160 (p2), 158 (p3), 133 (p4), 153 (p5), 144 (p6)
        # Right Eye: 362 (p1), 385 (p2), 387 (p3), 263 (p4), 373 (p5), 380 (p6)
        
        def eye_aspect_ratio(p1, p2, p3, p4, p5, p6):
            # Euclidian distance
            def dist(a, b): return np.linalg.norm(np.array([a.x, a.y]) - np.array([b.x, b.y]))
            
            vertical_1 = dist(p2, p5)
            vertical_2 = dist(p3, p6)
            horizontal = dist(p1, p4)
            return (vertical_1 + vertical_2) / (2.0 * horizontal)

        try:
            left_ear = eye_aspect_ratio(
                landmarks[33], landmarks[160], landmarks[158],
                landmarks[133], landmarks[153], landmarks[144]
            )
            right_ear = eye_aspect_ratio(
                landmarks[362], landmarks[385], landmarks[387],
                landmarks[263], landmarks[373], landmarks[380]
            )
            return (left_ear + right_ear) / 2.0
        except:
            return 0.3 # Default safe value if indexing fails

    def is_looking_away(self, landmarks):
        """
        Detects if user is looking away based on horizontal nose position relative to eyes.
        Returns True (Distracted) or False (Focused).
        """
        try:
            # Nose Tip: 1
            # Left Eye Outer: 33
            # Right Eye Outer: 263
            nose = landmarks[1]
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            
            # Simple horizontal processing
            # Distances
            dist_left = np.sqrt((nose.x - left_eye.x)**2 + (nose.y - left_eye.y)**2)
            dist_right = np.sqrt((nose.x - right_eye.x)**2 + (nose.y - right_eye.y)**2)
            
            # Yaw Ratio
            # If looking forward, dist_left ~= dist_right (Ratio ~= 1.0)
            # If looking left, nose moves left, dist_left decreases, ratio decreases
            
            if dist_right == 0: return True
            ratio = dist_left / dist_right
            
            # Thresholds (tuned for approximate 30-40 deg)
            # Normal range is roughly 0.6 to 1.6
            if ratio < 0.5 or ratio > 2.0:
                return True # Looking Left or Right
                
            return False
        except:
            return False

    def calculate_hybrid_score(self, face_crop, blendshapes, landmarks=None):
        """
        Calculates hybrid score with Smoothing, optimization, and blink detection.
        Arguments:
            face_crop (img): Face image for Keras model
            blendshapes (list): MediaPipe blendshapes
            landmarks (list, optional): MediaPipe landmarks for Blink/HeadPose detection
        """
        # --- 1. Geometric Score (MediaPipe) ---
        bs_map = {b.category_name: b.score for b in blendshapes}
        
        brow_down = (bs_map.get('browDownLeft', 0) + bs_map.get('browDownRight', 0)) / 2
        eye_squint = (bs_map.get('eyeSquintLeft', 0) + bs_map.get('eyeSquintRight', 0)) / 2
        lip_press = (bs_map.get('mouthPressLeft', 0) + bs_map.get('mouthPressRight', 0)) / 2
        
        def boost(val): return min(1.0, np.sqrt(val) * 1.5)
        
        geo_stress = (boost(brow_down) * 0.4) + (boost(eye_squint) * 0.3) + (boost(lip_press) * 0.3)
        geo_focused = 1.0 - geo_stress

        # --- 2. Head Pose & Looking Away ---
        looking_away_bonus = 0.0
        if landmarks:
            if self.is_looking_away(landmarks):
                looking_away_bonus = 0.8 # High distraction override

            # Blink Detection (Internal tracking only, no alert return)
            ear = self.calculate_ear(landmarks)
            if ear < self.EAR_THRESHOLD:
                self.last_blink_time = time.time() # Reset timer
        
        # --- 3. Visual Score (Keras) - WITH FRAME SKIPPING ---
        self.frame_count += 1
        
        # Only run prediction every 'skip_frames' or if we don't have a result yet
        if self.frame_count % self.skip_frames == 0 or self.last_emotions is None:
            # We assume face_crop is ready
            try:
                emotions = self.emotion_model.predict(face_crop)
            except:
                emotions = None # Handle edge cases
            self.last_emotions = emotions
        else:
            emotions = self.last_emotions # Use cached result

        vis_stress = 0.0
        vis_focused = 0.0
        vis_distracted = 0.0
        
        if emotions:
            vis_stress = emotions['Angry'] + emotions['Disgusted'] + emotions['Fearful'] + emotions['Sad']
            vis_focused = emotions['Neutral']
            vis_distracted = emotions['Happy'] + emotions['Surprised']

        # --- 4. Fusion ---
        final_stress = (vis_stress * 0.6) + (geo_stress * 0.4)
        final_focused = (vis_focused * 0.7) + (geo_focused * 0.3)
        final_distracted = vis_distracted 
        
        # Apply Looking Away logic
        # If looking away, Distracted dominates
        if looking_away_bonus > 0:
            final_distracted = max(final_distracted, looking_away_bonus)
            final_focused *= 0.2 # Penalty
            final_stress *= 0.5 # Penalty (hard to tell stress if side profile)

        # Normalize manually since we messed with values
        total = final_stress + final_focused + final_distracted
        if total > 0:
            final_stress /= total
            final_focused /= total
            final_distracted /= total
            
        # --- 5. Temporal Smoothing (Moving Average) ---
        self.score_buffer.append((final_stress, final_focused, final_distracted))
        
        # Calculate average from buffer
        avg_stress = sum([s[0] for s in self.score_buffer]) / len(self.score_buffer)
        avg_focus = sum([s[1] for s in self.score_buffer]) / len(self.score_buffer)
        avg_distracted = sum([s[2] for s in self.score_buffer]) / len(self.score_buffer)

        return {
            'stress_score': avg_stress, # Return SMOOTHED scores
            'focused_score': avg_focus,
            'distracted_score': avg_distracted,
            'details': {
                'visual_emotions': emotions,
                'geo_stress': geo_stress,
                'looking_away': looking_away_bonus > 0
            }
        }

class StudyTimer:
    def __init__(self, duration_minutes=30):
        self.total_seconds = duration_minutes * 60
        self.remaining_seconds = self.total_seconds
        self.is_running = False
        self.is_paused = False # Auto-pause state
        self.last_update_time = time.time()
        self.lock = Lock()
        
        self.no_face_start_time = None
        
    def start(self):
        self.is_running = True
        self.last_update_time = time.time()
        
    def stop(self):
        self.is_running = False
        
    def reset(self, duration_minutes):
        self.total_seconds = duration_minutes * 60
        self.remaining_seconds = self.total_seconds
        self.is_running = False
        self.is_paused = False
        self.no_face_start_time = None
        self.last_update_time = time.time()

    def update(self, face_detected):
        with self.lock:
            current_time = time.time()
            
            if not self.is_running:
                return

            if face_detected:
                self.no_face_start_time = None
                if self.is_paused:
                    self.is_paused = False # Auto-resume
                    self.last_update_time = current_time # Reset delta
            else:
                if self.no_face_start_time is None:
                    self.no_face_start_time = current_time
                elif current_time - self.no_face_start_time > 2.0:
                    self.is_paused = True
            
            if not self.is_paused:
                delta = current_time - self.last_update_time
                self.remaining_seconds = max(0, self.remaining_seconds - delta)
            
            self.last_update_time = current_time

class SessionLogger:
    def __init__(self, filepath="session_log.csv"):
        self.filepath = filepath
        # Initialize file with headers if not exists
        try:
            pd.read_csv(filepath)
        except FileNotFoundError:
            pd.DataFrame(columns=["Timestamp", "Stress_Score", "Focus_Status"]).to_csv(filepath, index=False)
            
    def log(self, stress_score, focus_status):
        new_row = {
            "Timestamp": datetime.now().isoformat(),
            "Stress_Score": stress_score,
            "Focus_Status": focus_status
        }
        df = pd.DataFrame([new_row])
        df.to_csv(self.filepath, mode='a', header=False, index=False)
