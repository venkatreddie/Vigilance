# dashboard.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import csv
import os
import threading
import winsound
import pandas as pd
from datetime import datetime

# ---------------------------
# Config
# ---------------------------
LOG_FILE = "detection_log.csv"
EAR_THRESHOLD = 0.25
YAWN_RATIO_THRESHOLD = 0.45  # normalized mouth open ratio
DISTRACTION_YAW = 20.0
DISTRACTION_PITCH = 15.0
ALERT_DELAY = 5.0  # seconds of sustained condition before alert
ALERT_COOLDOWN = 5.0  # seconds between sounds
FRAME_SLEEP = 0.02  # worker sleep to reduce CPU

# Setup CSV header if not exists
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["Timestamp", "Event", "Yaw", "Pitch", "EAR", "MouthRatio"])

# ---------------------------
# Utilities (detection + audio + logging)
# ---------------------------
def play_alert(alert_type: str):
    """Simple winsound beep pattern per alert type (Windows)."""
    try:
        if alert_type == "Drowsiness":
            winsound.Beep(800, 800)
        elif alert_type == "Yawning":
            winsound.Beep(1000, 700)
        elif alert_type == "Distraction":
            winsound.Beep(1200, 500)
    except Exception:
        pass  # winsound available only on Windows; ignore on others

def log_event_csv(event, yaw=0.0, pitch=0.0, ear=0.0, mouth_ratio=0.0):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([ts, event, round(yaw,2), round(pitch,2), round(ear,2), round(mouth_ratio,2)])
    return {"Timestamp": ts, "Event": event, "Yaw": round(yaw,2), "Pitch": round(pitch,2), "EAR": round(ear,2), "MouthRatio": round(mouth_ratio,2)}

# ---------------------------
# Detection math helpers
# ---------------------------
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)
LMKS_IDX = [1, 199, 33, 263, 61, 291]

def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.degrees([x, y, z])  # roll, pitch, yaw

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(lm, left_idx, right_idx):
    def ear_calc(pts):
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3])
        return (A + B) / (2.0 * C) if C != 0 else 0.0
    left = np.array([lm[i] for i in left_idx])
    right = np.array([lm[i] for i in right_idx])
    return (ear_calc(left) + ear_calc(right)) / 2.0

def mouth_open_ratio(lm):
    # Use inner lip points for vertical and corners for horizontal normalization
    # Points chosen for MediaPipe face mesh inner mouth
    top_pts = [13, 14]    # approximate inner top lip
    bottom_pts = [17, 18] # approximate inner bottom lip
    left_corner = np.array(lm[61])
    right_corner = np.array(lm[291])
    verticals = []
    for t, b in zip(top_pts, bottom_pts):
        verticals.append(np.linalg.norm(np.array(lm[t]) - np.array(lm[b])))
    vertical = np.mean(verticals) if len(verticals) else 0.0
    horizontal = np.linalg.norm(left_corner - right_corner) if (left_corner is not None and right_corner is not None) else 1.0
    return (vertical / horizontal) if horizontal > 0 else 0.0

# ---------------------------
# Worker thread that captures frames & runs detection
# ---------------------------
def detection_worker(shared):
    """
    shared: dict with keys:
      - stop_event (threading.Event)
      - frame_bgr (latest frame bytes)
      - status (dict of booleans and numbers)
      - logs (list of dicts)
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        shared["error"] = "Camera not accessible"
        return

    mp_face = mp.solutions.face_mesh
    with mp_face.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        # timers and flags for debounced alerts
        drowsy_start = None
        yawn_start = None
        distraction_start = None
        drowsy_alerted = False
        yawn_alerted = False
        distraction_alerted = False

        last_sound_time = 0.0

        while not shared["stop_event"].is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)  # mirror for natural look
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            # default status
            ear = 0.0
            mouth_ratio = 0.0
            yaw = 0.0
            pitch = 0.0

            if results.multi_face_landmarks:
                lm_raw = results.multi_face_landmarks[0].landmark
                lm = [(int(p.x * w), int(p.y * h)) for p in lm_raw]

                # EAR & mouth ratio
                try:
                    ear = eye_aspect_ratio(lm, LEFT_EYE_IDX, RIGHT_EYE_IDX)
                except Exception:
                    ear = 0.0
                try:
                    mouth_ratio = mouth_open_ratio(lm)
                except Exception:
                    mouth_ratio = 0.0

                # head pose
                try:
                    image_points = np.array([lm[i] for i in LMKS_IDX], dtype=np.float64)
                    focal_length = w
                    center = (w/2.0, h/2.0)
                    camera_matrix = np.array([[focal_length, 0, center[0]],
                                              [0, focal_length, center[1]],
                                              [0, 0, 1]], dtype=np.float64)
                    dist_coeffs = np.zeros((4,1))
                    success, rotation_vector, _ = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix, dist_coeffs)
                    R, _ = cv2.Rodrigues(rotation_vector)
                    roll, pitch, yaw = rotationMatrixToEulerAngles(R)
                except Exception:
                    yaw = pitch = 0.0

                now = time.time()

                # Drowsiness debounce (sustained EAR below threshold)
                if ear < EAR_THRESHOLD:
                    if drowsy_start is None:
                        drowsy_start = now
                    elif (now - drowsy_start) >= ALERT_DELAY and not drowsy_alerted:
                        # play sound (cooldown)
                        if now - last_sound_time >= ALERT_COOLDOWN:
                            play_alert("Drowsiness")
                            last_sound_time = now
                        entry = log_event_csv("Drowsiness", yaw, pitch, ear, mouth_ratio)
                        shared["logs"].insert(0, entry)
                        drowsy_alerted = True
                else:
                    drowsy_start = None
                    drowsy_alerted = False

                # Yawning debounce (sustained mouth ratio)
                if mouth_ratio > YAWN_RATIO_THRESHOLD:
                    if yawn_start is None:
                        yawn_start = now
                    elif (now - yawn_start) >= ALERT_DELAY and not yawn_alerted:
                        if now - last_sound_time >= ALERT_COOLDOWN:
                            play_alert("Yawning")
                            last_sound_time = now
                        entry = log_event_csv("Yawning", yaw, pitch, ear, mouth_ratio)
                        shared["logs"].insert(0, entry)
                        yawn_alerted = True
                else:
                    yawn_start = None
                    yawn_alerted = False

                # Distraction debounce (sustained head pose away)
                distracted = abs(yaw) > DISTRACTION_YAW or abs(pitch) > DISTRACTION_PITCH
                straight = abs(yaw) <= 12 and abs(pitch) <= 8
                if distracted:
                    if distraction_start is None:
                        distraction_start = now
                    elif (now - distraction_start) >= ALERT_DELAY and not distraction_alerted:
                        if now - last_sound_time >= ALERT_COOLDOWN:
                            play_alert("Distraction")
                            last_sound_time = now
                        entry = log_event_csv("Distraction", yaw, pitch, ear, mouth_ratio)
                        shared["logs"].insert(0, entry)
                        distraction_alerted = True
                elif straight:
                    distraction_start = None
                    distraction_alerted = False

                # Overlay text on frame
                cv2.putText(frame, f"EAR:{ear:.2f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                cv2.putText(frame, f"Mouth:{mouth_ratio:.2f}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                cv2.putText(frame, f"Yaw:{yaw:.2f} Pitch:{pitch:.2f}", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

                if drowsy_alerted:
                    cv2.putText(frame, "⚠ DROWSINESS", (120, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                if yawn_alerted:
                    cv2.putText(frame, "⚠ YAWNING", (120, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                if distraction_alerted:
                    cv2.putText(frame, "⚠ DISTRACTION", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

                # Update status
                shared["status"] = {
                    "drowsy": drowsy_alerted,
                    "yawn": yawn_alerted,
                    "distraction": distraction_alerted,
                    "EAR": round(ear,3),
                    "MouthRatio": round(mouth_ratio,3),
                    "Yaw": round(yaw,3),
                    "Pitch": round(pitch,3)
                }
            else:
                # No face detected: reset timers/flags for stable next detection
                drowsy_start = yawn_start = distraction_start = None
                drowsy_alerted = yawn_alerted = distraction_alerted = False
                shared["status"] = {"drowsy": False, "yawn": False, "distraction": False,
                                    "EAR": 0.0, "MouthRatio": 0.0, "Yaw": 0.0, "Pitch": 0.0}
                cv2.putText(frame, "No face detected", (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # Encode frame for UI
            ret2, jpg = cv2.imencode('.jpg', frame)
            if ret2:
                shared["frame_jpg"] = jpg.tobytes()

            time.sleep(FRAME_SLEEP)

    cap.release()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Driver Vigilance", layout="wide")
st.title("🚗 Driver Vigilance — Live Dashboard")

# Prepare session state for worker
if "shared" not in st.session_state:
    st.session_state.shared = {"frame_jpg": None, "status": {}, "logs": [], "stop_event": None, "error": None}
if "worker" not in st.session_state:
    st.session_state.worker = None

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    start_btn = st.button("▶️ Start Detection")
    stop_btn = st.button("⏹ Stop Detection")
    st.markdown("---")
    st.write("Logs saved to:", LOG_FILE)
    st.markdown("---")
    if st.button("Clear In-App Logs"):
        st.session_state.shared["logs"].clear()
        st.experimental_rerun()

# Start/Stop logic
if start_btn:
    # If a worker is already running, ignore
    if st.session_state.worker is None or not st.session_state.worker.is_alive():
        st.session_state.shared = {"frame_jpg": None, "status": {}, "logs": [], "stop_event": threading.Event(), "error": None}
        worker = threading.Thread(target=detection_worker, args=(st.session_state.shared,), daemon=True)
        st.session_state.worker = worker
        worker.start()
        time.sleep(0.2)  # small delay for first frame
        st.success("Detection started.")
    else:
        st.warning("Detection already running.")

if stop_btn:
    if st.session_state.shared.get("stop_event"):
        st.session_state.shared["stop_event"].set()
    if st.session_state.worker:
        st.session_state.worker.join(timeout=2.0)
    st.session_state.worker = None
    st.success("Detection stopped.")

# Layout: camera + status + logs
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Camera Feed (inside dashboard)")
    img_placeholder = st.empty()
    frame_bytes = st.session_state.shared.get("frame_jpg")
    if frame_bytes:
        img_placeholder.image(frame_bytes, use_column_width=True)
    else:
        img_placeholder.text("Camera feed will appear here after you click Start Detection.")

with col2:
    st.subheader("Status")
    status = st.session_state.shared.get("status", {})
    st.metric("Drowsiness", "Yes" if status.get("drowsy") else "No")
    st.metric("Yawning", "Yes" if status.get("yawn") else "No")
    st.metric("Distraction", "Yes" if status.get("distraction") else "No")
    st.write("Details:")
    st.write({
        "EAR": status.get("EAR"),
        "MouthRatio": status.get("MouthRatio"),
        "Yaw": status.get("Yaw"),
        "Pitch": status.get("Pitch")
    })

st.markdown("---")
st.subheader("Event History (most recent first)")
logs_list = st.session_state.shared.get("logs", [])
if os.path.exists(LOG_FILE):
    # combine in-app logs with csv file for completeness
    try:
        df_csv = pd.read_csv(LOG_FILE)
        df_csv = df_csv.iloc[::-1]  # show most recent first
    except Exception:
        df_csv = pd.DataFrame(columns=["Timestamp","Event","Yaw","Pitch","EAR","MouthRatio"])
else:
    df_csv = pd.DataFrame(columns=["Timestamp","Event","Yaw","Pitch","EAR","MouthRatio"])

# Merge in-app logs at top (they were already written to CSV)
if logs_list:
    df_inapp = pd.DataFrame(logs_list)
    df_display = pd.concat([df_inapp, df_csv], ignore_index=True).drop_duplicates().head(200)
else:
    df_display = df_csv.head(200)

if not df_display.empty:
    st.dataframe(df_display, use_container_width=True)
    csv_bytes = df_display.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Download Events CSV", csv_bytes, "events.csv", "text/csv")
else:
    st.info("No events logged yet.")

# Keep UI responsive: small refresh
time.sleep(0.2)
st.experimental_rerun()
