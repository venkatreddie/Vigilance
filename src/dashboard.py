import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import csv
import os
import threading
from datetime import datetime

# ---------------- SOUND HANDLER (Continuous Alarm) ----------------
try:
    import winsound
    def continuous_beep(stop_event):
        """Play continuous beeps until stop_event is set."""
        while not stop_event.is_set():
            try:
                winsound.Beep(2000, 300)
                time.sleep(0.1)
            except Exception:
                break
except Exception:
    def continuous_beep(stop_event):
        pass

def start_alarm_thread(name):
    """Start continuous alarm thread for a given detection type."""
    if name not in st.session_state.alarm_threads or not st.session_state.alarm_threads[name]["thread"].is_alive():
        stop_event = threading.Event()
        thread = threading.Thread(target=continuous_beep, args=(stop_event,), daemon=True)
        st.session_state.alarm_threads[name] = {"thread": thread, "stop_event": stop_event}
        thread.start()

def stop_alarm_thread(name):
    """Stop continuous alarm thread for a given detection type."""
    if name in st.session_state.alarm_threads:
        st.session_state.alarm_threads[name]["stop_event"].set()

# ---------------- CONFIG -----------------
LOG_FILE = "detection_log.csv"
ALERT_DELAY = 5.0
ALERT_COOLDOWN = 3.0

EAR_THRESHOLD = 0.25
YAWN_RATIO_THRESHOLD = 0.45
DISTRACTION_YAW = 20.0
DISTRACTION_PITCH = 15.0

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_IDX = [33,160,158,133,153,144]
RIGHT_EYE_IDX = [362,385,387,263,373,380]
MOUTH_TOP_INNER = [13,14]
MOUTH_BOTTOM_INNER = [17,18]
MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)
LMKS_IDX = [1, 199, 33, 263, 61, 291]

# Log setup
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp","Event","Yaw_deg","Pitch_deg","EAR","MouthRatio"])

def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.degrees([x, y, z])

def eye_aspect_ratio(landmarks, left_idx, right_idx):
    def ear_calc(points):
        A = np.linalg.norm(points[1]-points[5])
        B = np.linalg.norm(points[2]-points[4])
        C = np.linalg.norm(points[0]-points[3])
        return (A+B)/(2.0*C) if C!=0 else 0.0
    left = np.array([landmarks[i] for i in left_idx])
    right = np.array([landmarks[i] for i in right_idx])
    return (ear_calc(left) + ear_calc(right)) / 2.0

def mouth_open_ratio(landmarks):
    top_pts = [landmarks[i] for i in MOUTH_TOP_INNER if i < len(landmarks)]
    bottom_pts = [landmarks[i] for i in MOUTH_BOTTOM_INNER if i < len(landmarks)]
    if not top_pts or not bottom_pts:
        return 0.0
    verticals = [math.hypot(t[0]-b[0], t[1]-b[1]) for t,b in zip(top_pts, bottom_pts)]
    vertical = float(np.mean(verticals)) if verticals else 0.0
    left_corner = landmarks[MOUTH_LEFT_CORNER]
    right_corner = landmarks[MOUTH_RIGHT_CORNER]
    horizontal = math.hypot(left_corner[0]-right_corner[0], left_corner[1]-right_corner[1]) or 1.0
    return vertical / horizontal

def log_event(event, yaw, pitch, ear, mr):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([ts, event.strip().title(), round(yaw,2), round(pitch,2), round(ear,3), round(mr,3)])
    return {"Timestamp": ts, "Event": event.strip().title(), "Yaw": round(yaw,2), "Pitch": round(pitch,2), "EAR": round(ear,3), "MouthRatio": round(mr,3)}

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Driver Vigilance Dashboard", layout="wide")
st.title("🚘 Driver Vigilance — Continuous Alarm with 5s Delay")

st.sidebar.header("Controls")
start = st.sidebar.button("▶ Start Detection")
stop  = st.sidebar.button("⏹ Stop Detection")

if "running" not in st.session_state:
    st.session_state.running = False
if "logs_inapp" not in st.session_state:
    st.session_state.logs_inapp = []
if "alarm_threads" not in st.session_state:
    st.session_state.alarm_threads = {}

if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False

frame_slot = st.empty()
status_slot = st.empty()
cols = st.columns(4)
m_drowsy, m_yawn, m_dist, m_time = [c.empty() for c in cols]

# Initialize timers & flags
for key in ["drowsy", "yawn", "dist"]:
    if f"{key}_start" not in st.session_state:
        st.session_state[f"{key}_start"] = None
    if f"{key}_alerted" not in st.session_state:
        st.session_state[f"{key}_alerted"] = False

if st.session_state.running:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access camera.")
        st.session_state.running = False
    else:
        with mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as mesh:
            try:
                while st.session_state.running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    h, w = frame.shape[:2]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = mesh.process(rgb)

                    ear, mr, yaw_deg, pitch_deg = 0, 0, 0, 0
                    face_present = False

                    if result.multi_face_landmarks:
                        face_present = True
                        lm = result.multi_face_landmarks[0].landmark
                        lm_pts = [(int(p.x*w), int(p.y*h)) for p in lm]
                        ear = eye_aspect_ratio(lm_pts, LEFT_EYE_IDX, RIGHT_EYE_IDX)
                        mr = mouth_open_ratio(lm_pts)
                        try:
                            image_points = np.array([lm_pts[i] for i in LMKS_IDX], dtype=np.float64)
                            focal_length = w
                            center = (w/2, h/2)
                            cam_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0,0,1]], dtype=np.float64)
                            dist_coeffs = np.zeros((4,1))
                            _, rvec, _ = cv2.solvePnP(MODEL_POINTS, image_points, cam_matrix, dist_coeffs)
                            R, _ = cv2.Rodrigues(rvec)
                            roll_deg, pitch_deg, yaw_deg = rotationMatrixToEulerAngles(R)
                        except Exception:
                            pass

                    now = time.time()

                    # --- Drowsiness ---
                    if face_present and ear < EAR_THRESHOLD:
                        if st.session_state.drowsy_start is None:
                            st.session_state.drowsy_start = now
                        if now - st.session_state.drowsy_start >= ALERT_DELAY and not st.session_state.drowsy_alerted:
                            start_alarm_thread("Drowsiness")
                            entry = log_event("Drowsiness", yaw_deg, pitch_deg, ear, mr)
                            st.session_state.logs_inapp.insert(0, entry)
                            st.session_state.drowsy_alerted = True
                    else:
                        st.session_state.drowsy_start = None
                        if st.session_state.drowsy_alerted:
                            stop_alarm_thread("Drowsiness")
                            st.session_state.drowsy_alerted = False

                    # --- Yawning ---
                    if face_present and mr > YAWN_RATIO_THRESHOLD:
                        if st.session_state.yawn_start is None:
                            st.session_state.yawn_start = now
                        if now - st.session_state.yawn_start >= ALERT_DELAY and not st.session_state.yawn_alerted:
                            start_alarm_thread("Yawning")
                            entry = log_event("Yawning", yaw_deg, pitch_deg, ear, mr)
                            st.session_state.logs_inapp.insert(0, entry)
                            st.session_state.yawn_alerted = True
                    else:
                        st.session_state.yawn_start = None
                        if st.session_state.yawn_alerted:
                            stop_alarm_thread("Yawning")
                            st.session_state.yawn_alerted = False

                    # --- Distraction ---
                    head_away = abs(yaw_deg) > DISTRACTION_YAW or pitch_deg > DISTRACTION_PITCH or not face_present
                    if head_away:
                        if st.session_state.dist_start is None:
                            st.session_state.dist_start = now
                        if now - st.session_state.dist_start >= ALERT_DELAY and not st.session_state.dist_alerted:
                            start_alarm_thread("Distraction")
                            entry = log_event("Distraction", yaw_deg, pitch_deg, ear, mr)
                            st.session_state.logs_inapp.insert(0, entry)
                            st.session_state.dist_alerted = True
                    else:
                        st.session_state.dist_start = None
                        if st.session_state.dist_alerted:
                            stop_alarm_thread("Distraction")
                            st.session_state.dist_alerted = False

                    # Draw overlays
                    frame_disp = frame.copy()
                    alert_msgs = []
                    if st.session_state.drowsy_alerted: alert_msgs.append("⚠ DROWSINESS")
                    if st.session_state.yawn_alerted: alert_msgs.append("⚠ YAWNING")
                    if st.session_state.dist_alerted: alert_msgs.append("⚠ DISTRACTION")

                    for i, msg in enumerate(alert_msgs):
                        cv2.putText(frame_disp, msg, (100, 100+i*40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

                    frame_slot.image(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB), use_column_width=True)
                    m_drowsy.metric("Drowsy", "Yes" if st.session_state.drowsy_alerted else "No")
                    m_yawn.metric("Yawn", "Yes" if st.session_state.yawn_alerted else "No")
                    m_dist.metric("Distract", "Yes" if st.session_state.dist_alerted else "No")
                    m_time.metric("Time", datetime.now().strftime("%H:%M:%S"))

                    time.sleep(0.03)
            finally:
                cap.release()
                cv2.destroyAllWindows()
                for k in ["Drowsiness","Yawning","Distraction"]:
                    stop_alarm_thread(k)
else:
    frame_slot.text("Camera not running. Click ▶ Start Detection in the sidebar to begin.")
