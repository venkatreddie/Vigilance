import cv2
import streamlit as st
import numpy as np
import pandas as pd
import time
from datetime import datetime

# ------------------ Streamlit UI Setup ------------------
st.set_page_config(page_title="Driver Vigilance Dashboard", layout="wide")

st.title("🚗 Driver Vigilance Detection System")
st.sidebar.header("⚙️ Controls")

# ------------------ Session State Setup ------------------
if "run_detection" not in st.session_state:
    st.session_state.run_detection = False
if "log_data" not in st.session_state:
    st.session_state.log_data = []

# ------------------ Detection Functions ------------------
def detect_drowsiness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_intensity = np.mean(gray)
    return avg_intensity < 60  # Darker frame => possible closed eyes

def detect_yawning(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_lip = np.array([0, 48, 80], dtype=np.uint8)
    upper_lip = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_lip, upper_lip)
    ratio = cv2.countNonZero(mask) / (frame.size / 3)
    return ratio > 0.02  # Adjust threshold as needed

def detect_distraction(frame):
    h, w, _ = frame.shape
    left_side = np.mean(frame[:, :w//3])
    right_side = np.mean(frame[:, 2*w//3:])
    diff = abs(left_side - right_side)
    return diff > 25

# ------------------ Sidebar Controls ------------------
start = st.sidebar.button("▶️ Start Detection")
stop = st.sidebar.button("⏹️ Stop Detection")

# Start or stop detection
if start:
    st.session_state.run_detection = True
if stop:
    st.session_state.run_detection = False

# ------------------ Main Display Containers ------------------
FRAME_WINDOW = st.empty()
log_table = st.empty()

# ------------------ Detection Loop ------------------
if st.session_state.run_detection:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Camera not accessible. Please check your webcam.")
        st.session_state.run_detection = False
    else:
        st.sidebar.success("🟢 Detection Started. Close camera window to stop.")
        start_time = time.time()

        while st.session_state.run_detection:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to capture frame from camera.")
                break

            frame = cv2.flip(frame, 1)  # Mirror view

            drowsy = detect_drowsiness(frame)
            yawn = detect_yawning(frame)
            distract = detect_distraction(frame)

            alert_message = ""
            color = (0, 255, 0)

            if drowsy:
                alert_message = "😴 Drowsiness Detected!"
                color = (0, 0, 255)
            elif yawn:
                alert_message = "😮 Yawning Detected!"
                color = (255, 0, 0)
            elif distract:
                alert_message = "⚠️ Distraction Detected!"
                color = (0, 255, 255)

            if alert_message:
                cv2.putText(frame, alert_message, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                # Log entry
                st.session_state.log_data.append({
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Alert": alert_message.replace(" Detected!", ""),
                    "Frame_Brightness": round(np.mean(frame), 2)
                })

            FRAME_WINDOW.image(frame, channels="BGR")

            # Display the recent log data
            if len(st.session_state.log_data) > 0:
                df = pd.DataFrame(st.session_state.log_data)
                log_table.dataframe(df.tail(10), use_container_width=True)

            # Auto-stop after 60 seconds or manual stop
            if time.time() - start_time > 60 or not st.session_state.run_detection:
                st.session_state.run_detection = False
                break

        cap.release()
        st.sidebar.warning("🛑 Detection stopped automatically.")
else:
    st.sidebar.info("⏸️ Detection not running. Click 'Start Detection' to begin.")

# ------------------ Optional: Save Logs as CSV ------------------
if len(st.session_state.log_data) > 0:
    df = pd.DataFrame(st.session_state.log_data)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="💾 Download Detection Log as CSV",
        data=csv,
        file_name="detection_log.csv",
        mime="text/csv",
    )
