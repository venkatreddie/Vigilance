import cv2
import streamlit as st
import numpy as np
import mediapipe as mp
import time
import pandas as pd
import pygame
import os
from datetime import datetime

# Initialize pygame for sound
pygame.mixer.init()

# Function to play alert sound
def play_alert_sound():
    try:
        pygame.mixer.music.load("alert.wav")
        pygame.mixer.music.play()
    except:
        duration = 200  # milliseconds
        freq = 440
        os.system('play -nq -t alsa synth {} sine {}'.format(duration / 1000, freq))

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh

# EAR calculation
def eye_aspect_ratio(landmarks, eye_indices):
    p1 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p2 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
    p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p4 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    p5 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    p6 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    ear = (np.linalg.norm(p2 - p4) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p6))
    return ear

# MAR calculation
def mouth_aspect_ratio(landmarks):
    top = np.array([landmarks[13].x, landmarks[13].y])
    bottom = np.array([landmarks[14].x, landmarks[14].y])
    left = np.array([landmarks[78].x, landmarks[78].y])
    right = np.array([landmarks[308].x, landmarks[308].y])
    mar = np.linalg.norm(top - bottom) / np.linalg.norm(left - right)
    return mar

# Create or load detection log
log_file = "detection_log.csv"
if not os.path.exists(log_file):
    pd.DataFrame(columns=["Time", "Event"]).to_csv(log_file, index=False)

# Streamlit UI layout
st.set_page_config(page_title="Driver Vigilance Detection", layout="wide")
st.title("🚗 Driver Vigilance Detection System")
st.markdown("Detects **Drowsiness**, **Yawning**, and **Distraction** in Real Time")

# Columns
col1, col2 = st.columns([1, 2])
with col1:
    start_button = st.button("▶ Start Detection")
    stop_button = st.button("⏹ Stop Detection")

frame_placeholder = col2.empty()
alert_placeholder = st.empty()

# Thresholds and timers
EAR_THRESH = 0.25
MAR_THRESH = 0.6
EYE_CLOSED_TIME = 5  # seconds
YAWN_TIME = 5        # seconds
DISTRACTION_TIME = 5 # seconds

last_drowsy_time = 0
last_yawn_time = 0
last_distraction_time = 0

alert_displayed = None

if "run" not in st.session_state:
    st.session_state.run = False

if start_button:
    st.session_state.run = True
    st.rerun()
if stop_button:
    st.session_state.run = False
    st.rerun()

if st.session_state.run:
    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        start_time = time.time()
        while cap.isOpened() and st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not detected.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            event = None

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 387, 263, 373, 380]

                ear_left = eye_aspect_ratio(landmarks, left_eye_indices)
                ear_right = eye_aspect_ratio(landmarks, right_eye_indices)
                ear = (ear_left + ear_right) / 2.0
                mar = mouth_aspect_ratio(landmarks)

                # Drowsiness detection
                if ear < EAR_THRESH:
                    if time.time() - last_drowsy_time > EYE_CLOSED_TIME:
                        event = "Drowsiness Detected"
                        play_alert_sound()
                        last_drowsy_time = time.time()

                # Yawning detection
                if mar > MAR_THRESH:
                    if time.time() - last_yawn_time > YAWN_TIME:
                        event = "Yawning Detected"
                        play_alert_sound()
                        last_yawn_time = time.time()

                # Distraction detection (based on head position)
                nose = landmarks[1]
                if abs(nose.x - 0.5) > 0.25:
                    if time.time() - last_distraction_time > DISTRACTION_TIME:
                        event = "Distraction Detected"
                        play_alert_sound()
                        last_distraction_time = time.time()

                # Draw alert text
                if event:
                    cv2.putText(frame, f"⚠ {event} ⚠", (120, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    alert_placeholder.warning(event)

                    # Log event
                    pd.DataFrame([[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event]],
                                 columns=["Time", "Event"]).to_csv(log_file, mode='a', header=False, index=False)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", width=720)

        cap.release()
        cv2.destroyAllWindows()

# Show detection logs
st.markdown("---")
st.subheader("📜 Detection History")
if os.path.exists(log_file):
    log_data = pd.read_csv(log_file)
    st.dataframe(log_data[::-1], width=800)
