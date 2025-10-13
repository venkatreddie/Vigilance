import cv2
import dlib
import numpy as np
import streamlit as st
import time
import sqlite3
import winsound
from datetime import datetime
from scipy.spatial import distance

# -----------------------------------
# Database Setup
# -----------------------------------
conn = sqlite3.connect("driver_data.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event TEXT,
    timestamp TEXT
)
""")
conn.commit()

# -----------------------------------
# Load Models
# -----------------------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Thresholds
EYE_AR_THRESH = 0.25
YAWN_THRESH = 20
ALERT_TIME = 10

# -----------------------------------
# Helper Functions
# -----------------------------------
def eye_aspect_ratio(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def lip_distance(shape):
    top_lip = shape[50:53] + shape[61:64]
    bottom_lip = shape[56:59] + shape[65:68]
    top_mean = np.mean(top_lip, axis=0)
    bottom_mean = np.mean(bottom_lip, axis=0)
    distance_lips = abs(top_mean[1] - bottom_mean[1])
    return distance_lips

def trigger_alert(event_type):
    winsound.Beep(1000, 800)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO detections (event, timestamp) VALUES (?, ?)", (event_type, timestamp))
    conn.commit()
    st.session_state["last_event"] = (event_type, timestamp)

# -----------------------------------
# Streamlit UI Layout
# -----------------------------------
st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")

st.title("🚗 Driver Vigilance Monitoring System")
st.markdown("Real-time **Drowsiness** and **Yawning** Detection using OpenCV, Dlib, and Streamlit")

col1, col2 = st.columns(2)

if "run" not in st.session_state:
    st.session_state["run"] = False
if "last_event" not in st.session_state:
    st.session_state["last_event"] = None

# Buttons
with col1:
    start_btn = st.button("▶️ Start Detection")
    stop_btn = st.button("⏹ Stop Detection")

if start_btn:
    st.session_state["run"] = True
if stop_btn:
    st.session_state["run"] = False

# Live video and logs
FRAME_WINDOW = col1.image([])
status_placeholder = col2.empty()
event_log = col2.empty()

# -----------------------------------
# Detection Logic
# -----------------------------------
eye_closed_start = None
yawn_start = None
last_alert_time = 0

cap = cv2.VideoCapture(0)

while st.session_state["run"]:
    ret, frame = cap.read()
    if not ret:
        st.warning("⚠️ Cannot access webcam!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    event_message = "Monitoring..."

    for face in faces:
        shape = predictor(gray, face)
        shape_np = np.zeros((68, 2), dtype=int)
        for i in range(68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)

        left_eye = shape_np[42:48]
        right_eye = shape_np[36:42]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        lip_dist = lip_distance(shape_np)

        # Drowsiness Detection
        if ear < EYE_AR_THRESH:
            if eye_closed_start is None:
                eye_closed_start = time.time()
            elif time.time() - eye_closed_start > ALERT_TIME:
                trigger_alert("Drowsiness Detected")
                event_message = "😴 Drowsiness Detected!"
        else:
            eye_closed_start = None

        # Yawning Detection
        if lip_dist > YAWN_THRESH:
            if yawn_start is None:
                yawn_start = time.time()
            elif time.time() - yawn_start > 3:
                trigger_alert("Yawning Detected")
                event_message = "😮 Yawning Detected!"
        else:
            yawn_start = None

        # Draw visual feedback
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Lip: {int(lip_dist)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, event_message, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

    # UI status
    if st.session_state["last_event"]:
        e, t = st.session_state["last_event"]
        status_placeholder.markdown(f"### 🚨 Latest Event: **{e}** at {t}")
    else:
        status_placeholder.markdown("### ✅ No Alerts Yet")

    # Event Log Table
    cursor.execute("SELECT event, timestamp FROM detections ORDER BY id DESC LIMIT 5")
    rows = cursor.fetchall()
    if rows:
        event_log.dataframe(rows, use_container_width=True)
    else:
        event_log.info("No records yet...")

    time.sleep(0.05)

cap.release()
