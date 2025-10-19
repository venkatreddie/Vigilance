# dashboard.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import csv
import os
from datetime import datetime

# Try winsound (Windows) for simple beep; if not available, fallback (no sound)
try:
    import winsound
    def play_beep(freq=2000, duration=200):
        try:
            winsound.Beep(freq, duration)
        except Exception:
            pass
except Exception:
    def play_beep(freq=2000, duration=200):
        # cross-platform fallback: no-op
        pass

# ----------------- CONFIG -----------------
LOG_FILE = "detection_log.csv"
ALERT_DELAY = 5.0  # seconds of sustained condition before alert
ALERT_COOLDOWN = 3.0  # seconds minimum between repeated alerts of same type

# thresholds
EAR_THRESHOLD = 0.25           # eye aspect ratio threshold
YAWN_RATIO_THRESHOLD = 0.45    # normalized mouth open ratio threshold
DISTRACTION_YAW = 20.0         # degrees
DISTRACTION_PITCH = 15.0       # degrees

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh

# Landmark indices used
LEFT_EYE_IDX = [33,160,158,133,153,144]
RIGHT_EYE_IDX = [362,385,387,263,373,380]
# inner lip candidates and corners for mouth ratio
MOUTH_TOP_INNER = [13,14]    # approximate inner top
MOUTH_BOTTOM_INNER = [17,18] # approximate inner bottom
MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291

# head pose landmarks for solvePnP
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)
LMKS_IDX = [1, 199, 33, 263, 61, 291]  # nose tip, chin, left eye, right eye, left mouth, right mouth

# create log file header if missing
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp","Event","Yaw_deg","Pitch_deg","EAR","MouthRatio"])

# ----------- helper functions -------------
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
    return np.degrees([x, y, z])  # roll, pitch, yaw

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
    # normalized vertical using inner lip points and horizontal by mouth corners
    top_pts = [landmarks[i] for i in MOUTH_TOP_INNER if i < len(landmarks)]
    bottom_pts = [landmarks[i] for i in MOUTH_BOTTOM_INNER if i < len(landmarks)]
    if not top_pts or not bottom_pts:
        return 0.0
    verticals = []
    for t,b in zip(top_pts, bottom_pts):
        verticals.append(math.hypot(t[0]-b[0], t[1]-b[1]))
    vertical = float(np.mean(verticals)) if verticals else 0.0
    left_corner = landmarks[MOUTH_LEFT_CORNER]
    right_corner = landmarks[MOUTH_RIGHT_CORNER]
    horizontal = math.hypot(left_corner[0]-right_corner[0], left_corner[1]-right_corner[1]) if horizontal_valid(left_corner, right_corner) else 1.0
    return vertical / horizontal if horizontal>0 else 0.0

def horizontal_valid(left_corner, right_corner):
    return left_corner is not None and right_corner is not None

def log_event(event, yaw, pitch, ear, mr):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([ts, event, round(yaw,2), round(pitch,2), round(ear,3), round(mr,3)])
    return {"Timestamp": ts, "Event": event, "Yaw": round(yaw,2), "Pitch": round(pitch,2), "EAR": round(ear,3), "MouthRatio": round(mr,3)}

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Driver Vigilance Dashboard", layout="wide")
st.title("🚘 Driver Vigilance — Live Detection (5s sustained alert)")

st.sidebar.header("Controls")
start = st.sidebar.button("▶ Start Detection")
stop  = st.sidebar.button("⏹ Stop Detection")
st.sidebar.markdown("---")
st.sidebar.write("Alerts fire **only after** 5 continuous seconds of the condition.")
st.sidebar.write(f"EAR threshold: {EAR_THRESHOLD}, Yawn ratio threshold: {YAWN_RATIO_THRESHOLD}")

if "running" not in st.session_state:
    st.session_state.running = False
if "logs_inapp" not in st.session_state:
    st.session_state.logs_inapp = []

if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False

# placeholders
frame_slot = st.empty()
status_slot = st.empty()
metrics_cols = st.columns(4)

# status metrics placeholders
m_drowsy = metrics_cols[0].empty()
m_yawn = metrics_cols[1].empty()
m_dist = metrics_cols[2].empty()
m_time = metrics_cols[3].empty()

# detection state timers & flags
if "drowsy_start" not in st.session_state:
    st.session_state.drowsy_start = None
if "yawn_start" not in st.session_state:
    st.session_state.yawn_start = None
if "distraction_start" not in st.session_state:
    st.session_state.distraction_start = None

if "drowsy_alerted" not in st.session_state:
    st.session_state.drowsy_alerted = False
if "yawn_alerted" not in st.session_state:
    st.session_state.yawn_alerted = False
if "dist_alerted" not in st.session_state:
    st.session_state.dist_alerted = False

if "last_alert_times" not in st.session_state:
    st.session_state.last_alert_times = {"Drowsiness":0.0,"Yawning":0.0,"Distraction":0.0}

# helper to reset alerts when condition clears
def reset_if_cleared(cond_name):
    if cond_name == "Drowsiness":
        st.session_state.drowsy_start = None
        st.session_state.drowsy_alerted = False
    elif cond_name == "Yawning":
        st.session_state.yawn_start = None
        st.session_state.yawn_alerted = False
    elif cond_name == "Distraction":
        st.session_state.distraction_start = None
        st.session_state.dist_alerted = False

# run detection loop
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access camera. Make sure camera is free and accessible.")
        st.session_state.running = False
    else:
        # initialize MediaPipe face mesh
        with mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            try:
                while st.session_state.running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    h,w = frame.shape[:2]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb)

                    ear = 0.0
                    mouth_ratio = 0.0
                    yaw_deg = 0.0
                    pitch_deg = 0.0

                    face_present = False

                    if results.multi_face_landmarks:
                        face_present = True
                        lm = results.multi_face_landmarks[0].landmark
                        # convert to pixel coords list for convenience
                        lm_pts = [(int(p.x*w), int(p.y*h)) for p in lm]

                        # compute EAR
                        try:
                            ear = eye_aspect_ratio(lm_pts, LEFT_EYE_IDX, RIGHT_EYE_IDX)
                        except Exception:
                            ear = 0.0

                        # compute mouth ratio
                        try:
                            # pass lm_pts (list of tuples) into mouth_open routine which expects tuple coords
                            mouth_landmarks = lm_pts
                            mouth_ratio = mouth_open_ratio(mouth_landmarks)
                        except Exception:
                            mouth_ratio = 0.0

                        # head pose
                        try:
                            image_points = np.array([lm_pts[i] for i in LMKS_IDX], dtype=np.float64)
                            focal_length = w
                            center = (w/2.0, h/2.0)
                            camera_matrix = np.array([[focal_length, 0, center[0]],[0, focal_length, center[1]],[0,0,1]], dtype=np.float64)
                            dist_coeffs = np.zeros((4,1))
                            success, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                            R_mat, _ = cv2.Rodrigues(rvec)
                            roll_deg, pitch_deg, yaw_deg = rotationMatrixToEulerAngles(R_mat)
                        except Exception:
                            yaw_deg = 0.0
                            pitch_deg = 0.0

                    now = time.time()

                    # ---------- Drowsiness detection ----------
                    if face_present and ear>0.0 and ear < EAR_THRESHOLD:
                        if st.session_state.drowsy_start is None:
                            st.session_state.drowsy_start = now
                        # sustained?
                        if (now - st.session_state.drowsy_start) >= ALERT_DELAY and not st.session_state.drowsy_alerted:
                            # cooldown check to avoid repeats
                            if now - st.session_state.last_alert_times["Drowsiness"] >= ALERT_COOLDOWN:
                                play_beep(1500, 300)
                                entry = log_event("Drowsiness", yaw_deg, pitch_deg, ear, mouth_ratio)
                                st.session_state.logs_inapp.insert(0, entry)
                                st.session_state.drowsy_alerted = True
                                st.session_state.last_alert_times["Drowsiness"] = now
                    else:
                        # clear drowsy timer/flag when eyes open or no face
                        st.session_state.drowsy_start = None
                        st.session_state.drowsy_alerted = False

                    # ---------- Yawning detection ----------
                    if face_present and mouth_ratio>0.0 and mouth_ratio > YAWN_RATIO_THRESHOLD:
                        if st.session_state.yawn_start is None:
                            st.session_state.yawn_start = now
                        if (now - st.session_state.yawn_start) >= ALERT_DELAY and not st.session_state.yawn_alerted:
                            if now - st.session_state.last_alert_times["Yawning"] >= ALERT_COOLDOWN:
                                play_beep(2000, 300)
                                entry = log_event("Yawning", yaw_deg, pitch_deg, ear, mouth_ratio)
                                st.session_state.logs_inapp.insert(0, entry)
                                st.session_state.yawn_alerted = True
                                st.session_state.last_alert_times["Yawning"] = now
                    else:
                        st.session_state.yawn_start = None
                        st.session_state.yawn_alerted = False

                    # ---------- Distraction detection ----------
                    # If face is absent for sustained time OR head pose shows large yaw/pitch
                    head_away = abs(yaw_deg) > DISTRACTION_YAW or pitch_deg > DISTRACTION_PITCH
                    if (not face_present) or head_away:
                        # start timer if not started
                        if st.session_state.distraction_start is None:
                            st.session_state.distraction_start = now
                        if (now - st.session_state.distraction_start) >= ALERT_DELAY and not st.session_state.dist_alerted:
                            if now - st.session_state.last_alert_times["Distraction"] >= ALERT_COOLDOWN:
                                play_beep(1200, 300)
                                entry = log_event("Distraction", yaw_deg, pitch_deg, ear, mouth_ratio)
                                st.session_state.logs_inapp.insert(0, entry)
                                st.session_state.dist_alerted = True
                                st.session_state.last_alert_times["Distraction"] = now
                    else:
                        st.session_state.distraction_start = None
                        st.session_state.dist_alerted = False

                    # draw overlays
                    display_frame = frame.copy()
                    cv2.putText(display_frame, f"EAR:{ear:.2f}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),2)
                    cv2.putText(display_frame, f"MouthR:{mouth_ratio:.2f}", (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),2)
                    cv2.putText(display_frame, f"Yaw:{yaw_deg:.1f}", (10,85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),2)
                    cv2.putText(display_frame, f"Pitch:{pitch_deg:.1f}", (10,115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),2)

                    # show alerts text if alerted
                    alert_msgs = []
                    if st.session_state.drowsy_alerted:
                        alert_msgs.append("⚠ DROWSINESS")
                    if st.session_state.yawn_alerted:
                        alert_msgs.append("⚠ YAWNING")
                    if st.session_state.dist_alerted:
                        alert_msgs.append("⚠ DISTRACTION")
                    for i, msg in enumerate(alert_msgs):
                        cv2.putText(display_frame, msg, (120, 150 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

                    # update UI widgets
                    frame_slot.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
                    m_drowsy.metric("Drowsy", "Yes" if st.session_state.drowsy_alerted else "No")
                    m_yawn.metric("Yawning", "Yes" if st.session_state.yawn_alerted else "No")
                    m_dist.metric("Distraction", "Yes" if st.session_state.dist_alerted else "No")
                    m_time.metric("Time", datetime.now().strftime("%H:%M:%S"))

                    # small sleep & allow UI to be interactive
                    time.sleep(0.02)

            except Exception as e:
                st.error(f"Detection error: {e}")
            finally:
                cap.release()
                cv2.destroyAllWindows()
                st.session_state.running = False

# else (not running) – show placeholder image / message
else:
    frame_slot.text("Camera not running. Click ▶ Start Detection in the sidebar to begin.")

# show recent logs
st.markdown("---")
st.subheader("Recent Events (most recent first)")
if st.session_state.logs_inapp:
    import pandas as pd
    df_logs = pd.DataFrame(st.session_state.logs_inapp)
    st.dataframe(df_logs.head(200), use_container_width=True)
else:
    # If in-app logs empty, show csv if existing
    if os.path.exists(LOG_FILE):
        df_file = None
        try:
            import pandas as pd
            df_file = pd.read_csv(LOG_FILE)
            st.dataframe(df_file.tail(200).iloc[::-1], use_container_width=True)
        except Exception:
            st.info("No recent events logged yet.")
    else:
        st.info("No events logged yet.")

# Download CSV button (merge file + in-app logs)
if st.session_state.logs_inapp:
    # create combined CSV bytes
    try:
        import io, pandas as pd
        df_comb = pd.DataFrame(st.session_state.logs_inapp)
        if os.path.exists(LOG_FILE):
            df_file = pd.read_csv(LOG_FILE)
            df_combined = pd.concat([df_comb, df_file], ignore_index=True)
        else:
            df_combined = df_comb
        csv_bytes = df_combined.to_csv(index=False).encode('utf-8')
        st.download_button("Download events CSV", csv_bytes, "events.csv", "text/csv")
    except Exception:
        pass

    # ---------------------- EXISTING WORKING CODE ABOVE ----------------------
# (Do not change anything in your current working code)

# ---------------------- ANALYTICS & HISTORY SECTION ----------------------

st.markdown("## 📊 Detection Analytics & History Dashboard")

if os.path.exists(LOG_FILE):
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        from datetime import timedelta

        df_all = pd.read_csv(LOG_FILE)
        if not df_all.empty:
            df_all['Timestamp'] = pd.to_datetime(df_all['Timestamp'], errors='coerce')
            df_all = df_all.dropna(subset=['Timestamp'])

            # ------------- FILTERS -------------
            st.markdown("### 🔍 Filter Options")

            col1, col2, col3 = st.columns(3)

            # Event Type Filter
            event_types = sorted(df_all['Event'].unique())
            selected_events = col1.multiselect(
                "Select Event Types",
                event_types,
                default=event_types
            )

            # Date Range Filter
            min_date = df_all['Timestamp'].min().date()
            max_date = df_all['Timestamp'].max().date()
            date_range = col2.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

            # Recent Only Filter
            last24 = col3.checkbox("Show only last 24 hours", value=False)

            # Apply filters
            df_filtered = df_all.copy()
            df_filtered = df_filtered[df_filtered['Event'].isin(selected_events)]

            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
                df_filtered = df_filtered[
                    (df_filtered['Timestamp'].dt.date >= start_date)
                    & (df_filtered['Timestamp'].dt.date <= end_date)
                ]

            if last24:
                recent_time = datetime.now() - timedelta(hours=24)
                df_filtered = df_filtered[df_filtered['Timestamp'] >= recent_time]

            # ------------- SUMMARY METRICS -------------
            st.markdown("### 📈 Summary Overview")
            total_events = len(df_filtered)
            last_event_time = df_filtered['Timestamp'].max() if total_events > 0 else None
            unique_events = df_filtered['Event'].value_counts().to_dict()

            colA, colB, colC = st.columns(3)
            colA.metric("Total Alerts", total_events)
            colB.metric("Last Event",
                        last_event_time.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(last_event_time) else "N/A")
            colC.metric("Unique Events", ", ".join(list(unique_events.keys())) if unique_events else "None")

            # ------------- CHARTS -------------
            if total_events > 0:
                st.markdown("### 🔹 Event Distribution")
                fig1, ax1 = plt.subplots()
                counts = df_filtered['Event'].value_counts()
                ax1.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
                ax1.axis('equal')
                st.pyplot(fig1)

                st.markdown("### 🔹 Events Over Time")
                df_filtered['Hour'] = df_filtered['Timestamp'].dt.hour
                hourly_counts = df_filtered.groupby(['Hour', 'Event']).size().unstack(fill_value=0)

                fig2, ax2 = plt.subplots(figsize=(8, 4))
                hourly_counts.plot(kind='bar', stacked=True, ax=ax2)
                plt.xlabel("Hour of Day")
                plt.ylabel("Event Count")
                plt.title("Events per Hour")
                st.pyplot(fig2)

                # ----- Detailed History Table -----
                st.markdown("### 📜 Filtered Detection History")
                st.dataframe(
                    df_filtered.sort_values(by='Timestamp', ascending=False),
                    use_container_width=True
                )
            else:
                st.info("No data matches your filter selection.")

        else:
            st.info("No data available for analytics yet.")

    except Exception as e:
        st.error(f"Error loading analytics: {e}")
else:
    st.info("No log file found yet. Start detection to generate analytics data.")
