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
from io import BytesIO

# -----------------------
# CONFIG (same thresholds as working detection)
# -----------------------
LOG_FILE = "detection_log.csv"
EAR_THRESHOLD = 0.25
YAWN_RATIO_THRESHOLD = 0.45
DISTRACTION_YAW = 20.0
DISTRACTION_PITCH = 15.0
ALERT_DELAY = 5.0  # seconds before alert triggers
BEEP_FREQ = 2000
BEEP_DUR_MS = 200

# -----------------------
# MediaPipe setup (shared by thread)
# -----------------------
mp_face_mesh = mp.solutions.face_mesh

# -----------------------
# Ensure log exists
# -----------------------
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Event", "Yaw", "Pitch", "EAR", "MouthRatio"])

# -----------------------
# Helper functions
# -----------------------
def beep():
    try:
        winsound.Beep(BEEP_FREQ, BEEP_DUR_MS)
    except:
        pass

def append_log(event, yaw=0, pitch=0, ear=0, mr=0):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([timestamp, event, round(yaw,2), round(pitch,2), round(ear,2), round(mr,2)])

def read_log(limit=200):
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame(columns=["Timestamp","Event","Yaw","Pitch","EAR","MouthRatio"])
    df = pd.read_csv(LOG_FILE)
    if not df.empty:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        return df.sort_values(by="Timestamp", ascending=False).head(limit)
    return df

# -----------------------
# Aspect ratio calculators (same approach that worked)
# -----------------------
def eye_aspect_ratio(lm, left_idx, right_idx):
    def ear_calc(eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C) if C != 0 else 0.0
    left = np.array([lm[i] for i in left_idx])
    right = np.array([lm[i] for i in right_idx])
    return (ear_calc(left) + ear_calc(right)) / 2.0

def mouth_open_ratio(lm):
    # inner upper / lower; normalized by mouth width
    # indexes chosen for inner mouth points (MediaPipe indexing)
    top_pts = [13, 14]      # approximate inner top lip points
    bottom_pts = [17, 18]   # approximate inner bottom lip points
    left_corner = lm[61]
    right_corner = lm[291]
    verticals = []
    for t,b in zip(top_pts, bottom_pts):
        verticals.append(np.linalg.norm(np.array(lm[t]) - np.array(lm[b])))
    vertical = np.mean(verticals) if len(verticals) else 0.0
    horizontal = np.linalg.norm(np.array(left_corner) - np.array(right_corner))
    return (vertical / horizontal) if horizontal > 0 else 0.0

# -----------------------
# Head-pose util (for yaw/pitch)
# -----------------------
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
    return np.degrees([x,y,z])  # roll, pitch, yaw

# -----------------------
# MediaPipe landmark indices used for eyes
# -----------------------
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# -----------------------
# Threaded detection worker
# -----------------------
def detection_worker(state):
    """
    Worker thread: captures frames, runs MediaPipe detection,
    updates shared 'state' dict with:
      - 'frame_jpg' : latest jpeg bytes
      - 'status' : dict with booleans and numeric values
    Stops when state['stop_event'] is set.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        state['error'] = "Camera not found"
        return

    with mp_face_mesh.FaceMesh(refine_landmarks=True,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as face_mesh_local:
        # timers & flags for debounced alerts (5s)
        drowsy_start = None
        yawn_start = None
        distraction_start = None
        drowsy_alerted = False
        yawn_alerted = False
        distraction_alerted = False

        while not state['stop_event'].is_set():
            ret, frame = cap.read()
            if not ret:
                continue
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh_local.process(rgb)

            # defaults
            ear = 0.0
            mouth_ratio = 0.0
            yaw = 0.0
            pitch = 0.0

            if results.multi_face_landmarks:
                lm_raw = results.multi_face_landmarks[0].landmark
                lm = [(int(p.x * w), int(p.y * h)) for p in lm_raw]

                # EAR & mouth ratio
                ear = eye_aspect_ratio(lm, LEFT_EYE_IDX, RIGHT_EYE_IDX)
                mouth_ratio = mouth_open_ratio(lm)

                # head-pose
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

                # Drowsiness logic (debounced)
                if ear < EAR_THRESHOLD:
                    if drowsy_start is None:
                        drowsy_start = now
                    elif (now - drowsy_start) >= ALERT_DELAY and not drowsy_alerted:
                        beep()
                        append_log("Drowsiness", yaw, pitch, ear, mouth_ratio)
                        drowsy_alerted = True
                else:
                    drowsy_start = None
                    drowsy_alerted = False

                # Yawn logic (debounced)
                if mouth_ratio > YAWN_RATIO_THRESHOLD:
                    if yawn_start is None:
                        yawn_start = now
                    elif (now - yawn_start) >= ALERT_DELAY and not yawn_alerted:
                        beep()
                        append_log("Yawning", yaw, pitch, ear, mouth_ratio)
                        yawn_alerted = True
                else:
                    yawn_start = None
                    yawn_alerted = False

                # Distraction logic (debounced)
                distracted = abs(yaw) > DISTRACTION_YAW or abs(pitch) > DISTRACTION_PITCH
                straight = abs(yaw) <= 12 and abs(pitch) <= 8
                if distracted:
                    if distraction_start is None:
                        distraction_start = now
                    elif (now - distraction_start) >= ALERT_DELAY and not distraction_alerted:
                        beep()
                        append_log("Distraction", yaw, pitch, ear, mouth_ratio)
                        distraction_alerted = True
                elif straight:
                    distraction_start = None
                    distraction_alerted = False

                # Draw overlays
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                cv2.putText(frame, f"MouthRatio: {mouth_ratio:.2f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                cv2.putText(frame, f"Yaw: {yaw:.2f} Pitch: {pitch:.2f}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                if drowsy_alerted:
                    cv2.putText(frame, "⚠ DROWSINESS DETECTED ⚠", (120, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                if yawn_alerted:
                    cv2.putText(frame, "⚠ YAWNING DETECTED ⚠", (120, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                if distraction_alerted:
                    cv2.putText(frame, "⚠ DISTRACTION DETECTED ⚠", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

                # update state
                state['status'] = {
                    "drowsy": drowsy_alerted,
                    "yawn": yawn_alerted,
                    "distraction": distraction_alerted,
                    "ear": round(ear,3),
                    "mouth_ratio": round(mouth_ratio,3),
                    "yaw": round(yaw,3),
                    "pitch": round(pitch,3)
                }
            else:
                # No face
                cv2.putText(frame, "No face detected", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                # reset timers / flags
                drowsy_start = yawn_start = distraction_start = None
                drowsy_alerted = yawn_alerted = distraction_alerted = False
                state['status'] = {
                    "drowsy": False, "yawn": False, "distraction": False,
                    "ear": 0.0, "mouth_ratio": 0.0, "yaw": 0.0, "pitch": 0.0
                }

            # encode frame as jpeg bytes for Streamlit
            ret2, jpg = cv2.imencode('.jpg', frame)
            if ret2:
                state['frame_jpg'] = jpg.tobytes()
            # small sleep to reduce CPU usage
            time.sleep(0.02)

    cap.release()

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Driver Vigilance Dashboard", layout="wide")
st.title("🚘 Driver Vigilance — Live Dashboard")
st.markdown("Live camera feed with Drowsiness / Yawning / Distraction detection. Alerts trigger after sustained condition (5s).")

# session state init
if 'worker' not in st.session_state:
    st.session_state['worker'] = None
if 'stop_event' not in st.session_state:
    st.session_state['stop_event'] = threading.Event()
if 'state' not in st.session_state:
    st.session_state['state'] = {'frame_jpg': None, 'status': {}, 'error': None}

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    if st.session_state['worker'] is None or not st.session_state['worker'].is_alive():
        if st.button("▶️ Start Detection"):
            # reset stop event and start worker thread
            st.session_state['stop_event'] = threading.Event()
            shared_state = {'frame_jpg': None, 'status': {}, 'stop_event': st.session_state['stop_event'], 'error': None}
            st.session_state['state'] = shared_state
            worker = threading.Thread(target=detection_worker, args=(shared_state,), daemon=True)
            st.session_state['worker'] = worker
            worker.start()
            st.success("Detection started.")
    else:
        if st.button("⏹ Stop Detection"):
            st.session_state['stop_event'].set()
            st.session_state['worker'].join(timeout=2.0)
            st.session_state['worker'] = None
            st.success("Detection stopped.")

    st.markdown("---")
    st.markdown("Developed by Venkat 🚀")
    st.markdown("Tip: Close camera window using the Stop button before running detection again.")

# layout
col1, col2 = st.columns([2, 1])

# Left: camera
with col1:
    st.subheader("Live Camera")
    img_slot = st.empty()
    # show latest frame if available
    frame_bytes = st.session_state['state'].get('frame_jpg')
    if frame_bytes:
        img_slot.image(frame_bytes, use_column_width=True)
    else:
        img_slot.text("Camera not started. Click Start Detection in the sidebar.")

# Right: status & log
with col2:
    st.subheader("Live Status")
    status = st.session_state['state'].get('status', {})
    st.metric("Drowsiness", "Yes" if status.get('drowsy') else "No")
    st.metric("Yawning", "Yes" if status.get('yawn') else "No")
    st.metric("Distraction", "Yes" if status.get('distraction') else "No")
    st.write("Details:")
    st.write({
        "EAR": status.get('ear'),
        "MouthRatio": status.get('mouth_ratio'),
        "Yaw": status.get('yaw'),
        "Pitch": status.get('pitch')
    })
    st.markdown("---")
    st.subheader("Recent Events")
    df = read_log(limit=200)
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No events logged yet.")

# auto refresh small trick (no extra package)
last_refresh = st.session_state.get("last_refresh", 0)
REFRESH_SECONDS = 0.5
if time.time() - last_refresh > REFRESH_SECONDS:
    st.session_state['last_refresh'] = time.time()
    st.experimental_rerun()
