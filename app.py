import streamlit as st
import cv2
import numpy as np
import time
from src.logic import FaceDetector, LandmarkProcessor, StressCalculator, StudyTimer, SessionLogger

st.set_page_config(layout="wide", page_title="Stress Detector")

# Initialize Session State
if 'detector' not in st.session_state:
    try:
        st.session_state['detector'] = FaceDetector("yolo11n.pt")
        st.toast("YOLO11n Loaded!", icon="✅")
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        st.session_state['detector'] = None

if 'landmarker' not in st.session_state:
    st.session_state['landmarker'] = LandmarkProcessor()

if 'stress_calc' not in st.session_state:
    st.session_state['stress_calc'] = StressCalculator()

if 'timer' not in st.session_state or not hasattr(st.session_state['timer'], 'reset'):
    st.session_state['timer'] = StudyTimer(duration_minutes=30)
    
if 'logger' not in st.session_state:
    st.session_state['logger'] = SessionLogger()

if 'run' not in st.session_state:
    st.session_state['run'] = False

# Sidebar
st.sidebar.title("Controls")
if st.sidebar.button("Start/Stop Session"):
    st.session_state['run'] = not st.session_state['run']
    if st.session_state['run']:
        st.session_state['timer'].start()
    else:
        st.session_state['timer'].stop()

duration = st.sidebar.slider("Session Duration (min)", 10, 120, 30)

# Sync slider with timer
if 'last_duration' not in st.session_state:
    st.session_state['last_duration'] = 30

if duration != st.session_state['last_duration']:
    st.session_state['last_duration'] = duration
    # Only reset if not currently running to avoid glitches, or if user explicitly wants to change it
    # For now, we update the timer. The user can just STOP then change slider.
    st.session_state['timer'].reset(duration)
    st.toast(f"Timer updated to {duration} min", icon="⏱️")

# Main UI
st.title("Stress Detection & Focus Timer")

col1, col2 = st.columns([2, 1])

frame_placeholder = col1.empty()
stats_placeholder = col2.empty()

# Overlay Styles
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)

try:
    while True:
        if not st.session_state['run']:
            frame_placeholder.info("Session Stopped. Press Start to begin.")
            # Release camera if stopped to save resources/avoid conflict
            if cap.isOpened():
                cap.release()
            time.sleep(0.5)
            # Re-open if user clicks start (script reruns anyway, but just in case)
            continue
            
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)

        ret, frame = cap.read()
        if not ret:
            st.error("Camera not accessible")
            break
            
        # --- PROCESSING ---
        if st.session_state['detector'] is None:
            frame_placeholder.error("YOLO Model (yolo11n.pt) failed to load. logic.py error.")
            time.sleep(2)
            continue

        face_box = st.session_state['detector'].detect(frame)
        
        face_detected = face_box is not None
        st.session_state['timer'].update(face_detected)
        
        current_stress = 0.0
        current_focus = 0.0
        current_distracted = 0.0
        is_looking_away = False
        
        if face_detected:
            x1, y1, x2, y2 = map(int, face_box)
            # Crop
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size > 0:
                results = st.session_state['landmarker'].process(face_crop)
                
                # --- ROBUSTNESS CHECK ---
                # Only consider face "detected" if we actually found landmarks (eyes, nose, etc.)
                # This filters out random objects YOLO might have picked up.
                if results.face_landmarks:
                    st.session_state['timer'].update(True) # User is PRESENT
                    
                    if results.face_blendshapes:
                        blendshapes = results.face_blendshapes[0]
                        landmarks = results.face_landmarks[0]
                        
                        # Hybrid Calculation
                        scores = st.session_state['stress_calc'].calculate_hybrid_score(face_crop, blendshapes, landmarks)
                        current_stress = scores['stress_score']
                        current_focus = scores['focused_score']
                        current_distracted = scores['distracted_score']
                        
                        # Check Looking Away
                        is_looking_away = scores['details'].get('looking_away', False)
                
                    # Log (Log the dominant state)
                    state = "Focused"
                    if current_stress > current_focus and current_stress > current_distracted: state = "Stressed"
                    if current_distracted > current_focus and current_distracted > current_stress: state = "Distracted"
                    
                    st.session_state['logger'].log(current_stress, state)

                    # Privacy: We don't save. We just render for user feedback.
                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    st.session_state['timer'].update(False) # Face box found, but no features -> Ghost/Object
            else: # face_crop.size == 0 (e.g., box too small or out of bounds after crop)
                st.session_state['timer'].update(False)
        else:
            st.session_state['timer'].update(False) # No box found
        
        # --- RENDERING ---
        # Timer Overlay
        remaining = st.session_state['timer'].remaining_seconds
        mins, secs = divmod(int(remaining), 60)
        timer_text = f"{mins:02}:{secs:02}"
        
        if st.session_state['timer'].is_paused:
            cv2.putText(frame, "TIMER PAUSED", (50, 100), font, 1.5, (0, 0, 255), 3)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10) # Border
        else:
            cv2.putText(frame, f"Time: {timer_text}", (50, 50), font, 1, (255, 255, 255), 2)
        
        # Hybrid Scores Overlay
        # show all 3
        y_pos = frame.shape[0] - 50
        
        # Stress (Red)
        cv2.putText(frame, f"Stress: {current_stress:.0%}", (30, y_pos), font, 0.8, (0, 0, 255), 2)
        
        # Focus (Green)
        cv2.putText(frame, f"Focus: {current_focus:.0%}", (230, y_pos), font, 0.8, (0, 255, 0), 2)
        
        # Distracted (Orange/Cyan)
        cv2.putText(frame, f"Distracted: {current_distracted:.0%}", (430, y_pos), font, 0.8, (0, 165, 255), 2)
    
        if is_looking_away:
            cv2.putText(frame, "DISTRACTED: LOOKING AWAY", (30, y_pos - 40), font, 1, (0, 165, 255), 2)
    
        # Display in Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")
        
        # Stats Panel
        with stats_placeholder.container():
            m1, m2, m3 = st.columns(3)
            m1.metric("Stress", f"{current_stress:.0%}")
            m2.metric("Focus", f"{current_focus:.0%}")
            m3.metric("Distracted", f"{current_distracted:.0%}")
            
            st.metric("Time Remaining", timer_text)
            if st.session_state['timer'].is_paused:
                st.warning("⚠️ User Not Detected / Timer Paused")
            elif is_looking_away:
                st.warning("⚠️ Distracted: Looking Away")
            else:
                st.success("Analysis Active")

finally:
    cap.release()
