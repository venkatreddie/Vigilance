import cv2
import time
import math
import streamlit as st
import numpy as np
import pandas as pd
import threading
import pygame
from datetime import datetime
from collections import deque

# -------------------- INITIAL SETUP --------------------
st.set_page_config(page_title="Driver Vigilance Dashboard", layout="wide")
st.title("🚗 Driver Vigilance Detection Dashboard")

# Initialize pygame for sound
pygame.mixer.init()
alert_sound = "alert_beep.mp3"  # replace with your beep file
pygame.mixer.music.load(alert_sound)

# -------------------- ALERT SYSTEM --------------------
def play_alert_sound():
    """Play continuous beep sound in a background thread."""
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(-1)  # Loop indefinitely

def stop_alert_sound():
    """Stop alert sound."""
    pygame.mixer.music.stop()

# -------------------- DETECTION PLACEHOLDERS --------------------
def detect_drowsiness(eye_aspect_ratio):
    return eye_aspect_ratio < 0.25

def detect_yawning(mouth_aspect_ratio):
    return mouth_aspect_ratio > 0.6

def detect_distraction(face_angle):
    return abs(face_angle) > 25

# -------------------- DATA & LOGGING --------------------
log_data = deque(maxlen=1000)
stats = {"Drowsiness": 0, "Yawning": 0, "Distraction": 0}

def log_event(event_type):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_data.append({"Time": timestamp, "Event": event_type})
    stats[event_type] += 1

# -------------------- CAMERA DETECTION --------------------
FRAME_WINDOW = st.image([])
status_placeholder = st.empty()
start_button = st.button("▶️ Start Detection")
stop_button = st.button("⏹️ Stop Detection")

cap = None
detection_active = False
last_trigger_time = {"Drowsiness": 0, "Yawning": 0, "Distraction": 0}
trigger_delay = 5  # seconds delay

# -------------------- DETECTION LOOP --------------------
if start_button:
    detection_active = True
    cap = cv2.VideoCapture(0)
    st.session_state["active"] = True

if stop_button:
    detection_active = False
    stop_alert_sound()
    if cap:
        cap.release()
    st.session_state["active"] = False
    FRAME_WINDOW.image([])
    st.success("✅ Detection stopped successfully")

if st.session_state.get("active", False):
    st.info("🔍 Detection running... Please stay in front of the camera.")

    while detection_active and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---------------- Simulated detection logic ----------------
        # Replace this with your actual Mediapipe detection logic
        ear = np.random.uniform(0.2, 0.35)  # simulate EAR
        mar = np.random.uniform(0.4, 0.8)   # simulate MAR
        angle = np.random.uniform(-30, 30)  # simulate distraction

        drowsy = detect_drowsiness(ear)
        yawning = detect_yawning(mar)
        distracted = detect_distraction(angle)

        detected_event = None
        if drowsy:
            if time.time() - last_trigger_time["Drowsiness"] >= trigger_delay:
                detected_event = "Drowsiness"
                last_trigger_time["Drowsiness"] = time.time()
        elif yawning:
            if time.time() - last_trigger_time["Yawning"] >= trigger_delay:
                detected_event = "Yawning"
                last_trigger_time["Yawning"] = time.time()
        elif distracted:
            if time.time() - last_trigger_time["Distraction"] >= trigger_delay:
                detected_event = "Distraction"
                last_trigger_time["Distraction"] = time.time()
        else:
            stop_alert_sound()

        # Draw on frame
        if detected_event:
            cv2.putText(frame, f"{detected_event} Detected!", (50, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            play_alert_sound()
            log_event(detected_event)
        else:
            cv2.putText(frame, "Normal", (50, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        FRAME_WINDOW.image(frame, channels="BGR")

        # stop condition
        if not st.session_state.get("active", True):
            stop_alert_sound()
            break

    cap.release()
    stop_alert_sound()

# -------------------- TABS SECTION --------------------
tab1, tab2, tab3 = st.tabs(["📊 Live Statistics", "🕒 Event Log", "📈 Charts & Analytics"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("😴 Drowsiness Count", stats["Drowsiness"])
    col2.metric("😮 Yawning Count", stats["Yawning"])
    col3.metric("📱 Distraction Count", stats["Distraction"])

with tab2:
    st.subheader("Recent Detection Logs")
    df = pd.DataFrame(list(log_data))
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No detection logs yet.")

with tab3:
    st.subheader("Detection Trend Analytics")
    if not df.empty:
        chart_data = df["Event"].value_counts().reset_index()
        chart_data.columns = ["Event", "Count"]
        st.bar_chart(chart_data, x="Event", y="Count", use_container_width=True)

        df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce")
        time_series = df.groupby("Time").size()
        st.line_chart(time_series, use_container_width=True)
    else:
        st.info("No data to display charts yet.")
