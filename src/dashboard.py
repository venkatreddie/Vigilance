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
import pandas as pd
import plotly.express as px

# ================= SOUND ALERT SETUP =================
try:
    import winsound

    def continuous_beep(freq=2000):
        def loop():
            while getattr(threading.current_thread(), "do_run", True):
                try:
                    winsound.Beep(freq, 500)
                except Exception:
                    pass
                time.sleep(0.1)
        t = threading.Thread(target=loop)
        t.do_run = True
        t.daemon = True
        t.start()
        return t

except Exception:
    def continuous_beep(freq=2000):
        return None


def stop_beep_thread(thread):
    try:
        if thread and hasattr(thread, "do_run"):
            thread.do_run = False
    except:
        pass


# ================= CONFIG =================
LOG_FILE = "detection_log.csv"
ALERT_DELAY = 5.0
EAR_THRESHOLD = 0.25
YAWN_RATIO_THRESHOLD = 0.45
DISTRACTION_YAW = 20.0
DISTRACTION_PITCH = 15.0

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

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

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["Timestamp","Event","Yaw_deg","Pitch_deg","EAR","MouthRatio"])


# ================= FUNCTIONS =================
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
    event = event.strip().title()  # standardize event name
    if event == "Yawning Detected": event = "Yawning"
    if event == "Drowsiness Detected": event = "Drowsiness"
    if event == "Distraction Detected": event = "Distraction"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([ts, event, round(yaw,2), round(pitch,2), round(ear,3), round(mr,3)])
    return {"Timestamp": ts, "Event": event, "Yaw": round(yaw,2), "Pitch": round(pitch,2), "EAR": round(ear,3), "MouthRatio": round(mr,3)}


# ================= STREAMLIT UI =================
st.set_page_config(page_title="Driver Vigilance Dashboard", layout="wide")
st.markdown("<h1 style='text-align:center;color:#ff4b4b;'>🚗 Driver Vigilance Detection Dashboard</h1>", unsafe_allow_html=True)
st.write("")

col1, col2 = st.columns([2.2, 1])

with col2:
    st.markdown("### 🎮 Controls")
    start = st.button("▶ Start Detection", use_container_width=True)
    stop = st.button("⏹ Stop Detection", use_container_width=True)

    st.markdown("### 🔍 Filter Logs")
    event_filter = st.selectbox("Event Type", ["All", "Drowsiness", "Yawning", "Distraction", "Mobile Usage"])

    st.markdown("### 🕒 Event Log (Recent 10)")
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        df["Event"] = df["Event"].replace({
            "Yawning Detected": "Yawning",
            "Drowsiness Detected": "Drowsiness",
            "Distraction Detected": "Distraction"
        })
        if event_filter != "All":
            df = df[df["Event"] == event_filter]
        st.dataframe(df.tail(10), use_container_width=True)
    else:
        st.info("No logs recorded yet.")

    # ========== SESSION ANALYTICS ==========
    st.markdown("### 📈 Session Analytics")
    if os.path.exists(LOG_FILE):
        df_all = pd.read_csv(LOG_FILE)
        if not df_all.empty:
            df_all["Event"] = df_all["Event"].replace({
                "Yawning Detected": "Yawning",
                "Drowsiness Detected": "Drowsiness",
                "Distraction Detected": "Distraction"
            })
            event_counts = df_all["Event"].value_counts().reset_index()
            event_counts.columns = ["Event", "Count"]
            st.plotly_chart(px.pie(event_counts, names="Event", values="Count", title="Event Distribution"), use_container_width=True)
            st.plotly_chart(px.bar(event_counts, x="Event", y="Count", color="Event", title="Event Counts"), use_container_width=True)
            df_all["Timestamp"] = pd.to_datetime(df_all["Timestamp"], errors="coerce")
            df_time = df_all.groupby(pd.Grouper(key="Timestamp", freq="1min")).size().reset_index(name="Detections")
            st.plotly_chart(px.line(df_time, x="Timestamp", y="Detections", title="Detection Trend Over Time"), use_container_width=True)
        else:
            st.info("No analytics data yet.")
    else:
        st.info("No analytics data yet.")


with col1:
    frame_slot = st.empty()
    status_slot = st.empty()
    st.markdown("### 📊 Real-time EAR & Mouth Ratio")
    ear_chart = st.line_chart([], use_container_width=True)
    mouth_chart = st.line_chart([], use_container_width=True)

# ================= SESSION STATE =================
if "running" not in st.session_state:
    st.session_state.running = False
if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False
if "beep_thread" not in st.session_state:
    st.session_state.beep_thread = None

ear_values, mouth_values = [], []

# ================= MAIN DETECTION LOOP =================
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Camera not accessible.")
        st.session_state.running = False
    else:
        with mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
             mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

            drowsy_start = yawn_start = dist_start = mobile_start = None
            drowsy_active = yawn_active = dist_active = mobile_active = False

            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_results = face_mesh.process(rgb)
                hand_results = hands.process(rgb)

                ear, mr, yaw_deg, pitch_deg = 0, 0, 0, 0
                face_present = False
                if face_results.multi_face_landmarks:
                    face_present = True
                    lm = face_results.multi_face_landmarks[0].landmark
                    lm_pts = [(int(p.x * w), int(p.y * h)) for p in lm]
                    ear = eye_aspect_ratio(lm_pts, LEFT_EYE_IDX, RIGHT_EYE_IDX)
                    mr = mouth_open_ratio(lm_pts)
                    try:
                        img_pts = np.array([lm_pts[i] for i in LMKS_IDX], dtype=np.float64)
                        focal_length = w
                        center = (w / 2, h / 2)
                        cam_mtx = np.array([[focal_length, 0, center[0]],
                                            [0, focal_length, center[1]],
                                            [0, 0, 1]], dtype=np.float64)
                        success, rvec, tvec = cv2.solvePnP(MODEL_POINTS, img_pts, cam_mtx, np.zeros((4, 1)))
                        R, _ = cv2.Rodrigues(rvec)
                        roll_deg, pitch_deg, yaw_deg = rotationMatrixToEulerAngles(R)
                    except:
                        pass

                now = time.time()
                # Drowsiness
                if face_present and ear < EAR_THRESHOLD:
                    if drowsy_start is None: drowsy_start = now
                    if now - drowsy_start >= ALERT_DELAY:
                        if not drowsy_active: log_event("Drowsiness", yaw_deg, pitch_deg, ear, mr)
                        drowsy_active = True
                else:
                    drowsy_start = None; drowsy_active = False

                # Yawning
                if face_present and mr > YAWN_RATIO_THRESHOLD:
                    if yawn_start is None: yawn_start = now
                    if now - yawn_start >= ALERT_DELAY:
                        if not yawn_active: log_event("Yawning", yaw_deg, pitch_deg, ear, mr)
                        yawn_active = True
                else:
                    yawn_start = None; yawn_active = False

                # Distraction
                if (not face_present) or abs(yaw_deg) > DISTRACTION_YAW or pitch_deg > DISTRACTION_PITCH:
                    if dist_start is None: dist_start = now
                    if now - dist_start >= ALERT_DELAY:
                        if not dist_active: log_event("Distraction", yaw_deg, pitch_deg, ear, mr)
                        dist_active = True
                else:
                    dist_start = None; dist_active = False

                # Mobile Usage
                mobile_detected = False
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        for lm in hand_landmarks.landmark:
                            x, y = int(lm.x * w), int(lm.y * h)
                            if h * 0.2 < y < h * 0.8:
                                mobile_detected = True
                                break
                        if mobile_detected:
                            break

                if mobile_detected:
                    if mobile_start is None: mobile_start = now
                    if now - mobile_start >= ALERT_DELAY:
                        if not mobile_active: log_event("Mobile Usage", yaw_deg, pitch_deg, ear, mr)
                        mobile_active = True
                else:
                    mobile_start = None; mobile_active = False

                # ===== Accident Probability Calculation =====
                risk = 0
                if drowsy_active: risk += 50
                if yawn_active: risk += 30
                if dist_active: risk += 40
                if mobile_active: risk += 20
                risk = min(100, risk)

                if risk <= 30:
                    color = "🟢 Low Risk"
                elif risk <= 60:
                    color = "🟡 Medium Risk"
                else:
                    color = "🔴 HIGH RISK"

                # Sound alert
                if drowsy_active or yawn_active or dist_active or mobile_active:
                    if st.session_state.beep_thread is None or not st.session_state.beep_thread.is_alive():
                        st.session_state.beep_thread = continuous_beep(1500)
                else:
                    stop_beep_thread(st.session_state.beep_thread)
                    st.session_state.beep_thread = None

                # ----- Display -----
                disp = frame.copy()
                if drowsy_active: cv2.putText(disp,"⚠ DROWSINESS DETECTED",(30,80),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,0,255),3)
                if yawn_active: cv2.putText(disp,"⚠ YAWNING DETECTED",(30,130),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,0,255),3)
                if dist_active: cv2.putText(disp,"⚠ DISTRACTION DETECTED",(30,180),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,0,255),3)
                if mobile_active: cv2.putText(disp,"⚠ MOBILE USAGE DETECTED",(30,230),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,0,255),3)
                cv2.putText(disp, f"🚦 Accident Risk: {risk:.1f}% ({color})", (30,280), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,165,255), 3)

                frame_slot.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), use_container_width=True)

                status_slot.markdown(f"""
                    **EAR:** {ear:.2f} | **Mouth Ratio:** {mr:.2f}  
                    **Yaw:** {yaw_deg:.2f}° | **Pitch:** {pitch_deg:.2f}°  
                    **🚦 Accident Risk Probability:** {risk:.1f}% ({color})
                """)

                ear_values.append(ear)
                mouth_values.append(mr)
                if len(ear_values) > 50:
                    ear_values.pop(0)
                    mouth_values.pop(0)
                ear_chart.add_rows([ear])
                mouth_chart.add_rows([mr])

                time.sleep(0.03)

            stop_beep_thread(st.session_state.beep_thread)
            st.session_state.beep_thread = None
            cap.release()
            cv2.destroyAllWindows()
else:
    frame_slot.text("📷 Camera not running. Click ▶ Start Detection to begin.")

    # ----- 📱 Mobile Usage Trend Chart -----
    if os.path.exists(LOG_FILE):
        df_all = pd.read_csv(LOG_FILE)
        mobile_df = df_all[df_all["Event"].str.contains("Mobile", case=False, na=False)]
        if not mobile_df.empty:
            mobile_df["Timestamp"] = pd.to_datetime(mobile_df["Timestamp"], errors="coerce")
            mobile_time = mobile_df.groupby(pd.Grouper(key="Timestamp", freq="1min")).size().reset_index(name="MobileDetections")
            st.plotly_chart(
                px.line(
                    mobile_time,
                    x="Timestamp",
                    y="MobileDetections",
                    title="📱 Mobile Usage Trends Over Time",
                    markers=True,
                    line_shape="linear"
                ),
                use_container_width=True
            )
        else:
            st.info("No Mobile Usage data recorded yet.")
