import cv2
import streamlit as st
import numpy as np
import time
from datetime import datetime
import pandas as pd
from playsound import playsound
import threading
import tempfile
import os
from collections import deque
from streamlit.components.v1 import html
from io import BytesIO
import base64

# ===================== Streamlit Page Setup =====================
st.set_page_config(page_title="Driver Vigilance Dashboard", layout="wide")

st.markdown("""
    <style>
        body {
            background-color: #f8f9fa;
        }
        .main {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 20px;
        }
        .block-container {
            padding-top: 1rem;
        }
        h1, h2, h3 {
            color: #003366;
        }
        .card {
            background-color: #e9f2ff;
            border: 2px solid #0056b3;
            border-radius: 15px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

st.title("🚗 Driver Vigilance Detection Dashboard")

# ===================== Helper Functions =====================

def play_beep_sound():
    """Plays continuous beep in a separate thread."""
    def beep_loop():
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.write(b'\x00')
        temp_file.close()
        playsound("alert.mp3") if os.path.exists("alert.mp3") else None
    threading.Thread(target=beep_loop, daemon=True).start()

def log_event(event_type, yaw, pitch, ear, mar):
    """Logs the detection event to the dataframe."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_event = pd.DataFrame([[ts, event_type, yaw, pitch, ear, mar]],
                             columns=["Timestamp", "Event", "Yaw", "Pitch", "EAR", "MAR"])
    st.session_state.logs = pd.concat([st.session_state.logs, new_event], ignore_index=True)

# ===================== Initialize Session States =====================
if "run" not in st.session_state:
    st.session_state.run = False
if "logs" not in st.session_state:
    st.session_state.logs = pd.DataFrame(columns=["Timestamp", "Event", "Yaw", "Pitch", "EAR", "MAR"])
if "safety_score" not in st.session_state:
    st.session_state.safety_score = 100

# ===================== Sidebar Controls =====================
with st.sidebar:
    st.header("🧭 Controls")
    start_btn = st.button("▶ Start Detection", use_container_width=True)
    stop_btn = st.button("⏹ Stop Detection", use_container_width=True)
    st.divider()
    st.header("⚙️ Parameters")
    EAR_THRESHOLD = st.slider("Drowsiness EAR Threshold", 0.1, 0.3, 0.2)
    MAR_THRESHOLD = st.slider("Yawning MAR Threshold", 0.5, 0.8, 0.65)
    DISTRACTION_THRESHOLD = st.slider("Distraction Angle Threshold", 10, 25, 15)
    st.divider()
    st.caption("🎵 Alerts are triggered after 5s of continuous detection.")

# ===================== Safety Score Panel =====================
def update_safety(event_type):
    deduction = {"Drowsiness": 15, "Yawning": 10, "Distraction": 12, "Mobile Usage": 8}
    st.session_state.safety_score = max(0, st.session_state.safety_score - deduction.get(event_type, 5))
    color = "green" if st.session_state.safety_score > 70 else "orange" if st.session_state.safety_score > 40 else "red"
    st.markdown(f"""
        <div class="card">
            <h3>🛡️ Safety Score</h3>
            <div style="width:100%; background:#ddd; border-radius:10px;">
                <div style="width:{st.session_state.safety_score}%; background:{color}; color:white; text-align:center; padding:5px; border-radius:10px;">
                    {st.session_state.safety_score}/100
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ===================== Layout =====================
col1, col2 = st.columns([2, 1])

# ===================== Detection Simulation (Replace with Real Detection) =====================
def simulate_detection():
    """Simulated detection - replace with detection_system output later"""
    yaw = np.random.uniform(-15, 15)
    pitch = np.random.uniform(-10, 10)
    ear = np.random.uniform(0.15, 0.35)
    mar = np.random.uniform(0.4, 0.8)
    return yaw, pitch, ear, mar

# ===================== Real-Time Detection Loop =====================
if start_btn:
    st.session_state.run = True
if stop_btn:
    st.session_state.run = False

with col1:
    st.subheader("🎥 Live Detection Feed")
    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    if st.session_state.run:
        cap = cv2.VideoCapture(0)
        detection_start = {"Drowsiness": None, "Yawning": None, "Distraction": None}
        last_event_time = time.time()

        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not detected. Please check webcam.")
                break

            yaw, pitch, ear, mar = simulate_detection()

            # Detect Drowsiness
            if ear < EAR_THRESHOLD:
                if detection_start["Drowsiness"] is None:
                    detection_start["Drowsiness"] = time.time()
                elif time.time() - detection_start["Drowsiness"] >= 5:
                    status_placeholder.error("😴 Drowsiness Detected!")
                    log_event("Drowsiness", yaw, pitch, ear, mar)
                    update_safety("Drowsiness")
                    play_beep_sound()
            else:
                detection_start["Drowsiness"] = None

            # Detect Yawning
            if mar > MAR_THRESHOLD:
                if detection_start["Yawning"] is None:
                    detection_start["Yawning"] = time.time()
                elif time.time() - detection_start["Yawning"] >= 5:
                    status_placeholder.warning("😮 Yawning Detected!")
                    log_event("Yawning", yaw, pitch, ear, mar)
                    update_safety("Yawning")
                    play_beep_sound()
            else:
                detection_start["Yawning"] = None

            # Detect Distraction
            if abs(yaw) > DISTRACTION_THRESHOLD or abs(pitch) > DISTRACTION_THRESHOLD:
                if detection_start["Distraction"] is None:
                    detection_start["Distraction"] = time.time()
                elif time.time() - detection_start["Distraction"] >= 5:
                    status_placeholder.info("🚫 Distraction Detected!")
                    log_event("Distraction", yaw, pitch, ear, mar)
                    update_safety("Distraction")
                    play_beep_sound()
            else:
                detection_start["Distraction"] = None

            # Draw frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", width=640)
        cap.release()
        st.session_state.run = False

# ===================== Right Panel: Analytics =====================
with col2:
    st.markdown("### 📊 Real-Time Analytics")
    update_safety("")

    st.markdown("<div class='card'><h4>Detection Logs</h4></div>", unsafe_allow_html=True)
    st.dataframe(st.session_state.logs, use_container_width=True)

    if not st.session_state.logs.empty:
        st.markdown("<div class='card'><h4>📈 Detection Frequency</h4></div>", unsafe_allow_html=True)
        freq_data = st.session_state.logs["Event"].value_counts().reset_index()
        freq_data.columns = ["Event", "Count"]
        st.bar_chart(freq_data.set_index("Event"))

        st.markdown("<div class='card'><h4>📉 Mobile Usage Trends</h4></div>", unsafe_allow_html=True)
        mobile_logs = st.session_state.logs[st.session_state.logs["Event"] == "Mobile Usage"]
        if not mobile_logs.empty:
            mobile_logs["Timestamp"] = pd.to_datetime(mobile_logs["Timestamp"])
            mobile_logs["Count"] = 1
            mobile_trend = mobile_logs.resample("1T", on="Timestamp").sum(numeric_only=True)
            st.line_chart(mobile_trend["Count"])

st.success("✅ System Ready — Waiting for Camera Activation...")

