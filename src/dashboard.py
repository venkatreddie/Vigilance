import streamlit as st
import pandas as pd
import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import os
import datetime
import matplotlib.pyplot as plt

# ==========================================================
# CONFIGURATION
# ==========================================================
LOG_FILE = "detection_log.csv"
EVENT_TYPES = ["Drowsiness", "Yawning", "Distraction"]
BEEP_FREQ = 2000
BEEP_DUR = 200

# ==========================================================
# PAGE SETUP
# ==========================================================
st.set_page_config(page_title="Driver Vigilance Dashboard", layout="wide")
st.title("🧠 Driver Vigilance Monitoring System")

tabs = st.tabs(["🚗 Live Detection", "📊 Analytics Dashboard"])

# ==========================================================
# TAB 1 — LIVE DETECTION
# ==========================================================
with tabs[0]:
    st.header("🚦 Real-Time Driver Monitoring")

    start_button = st.button("Start Detection")
    stop_button = st.button("Stop Detection")

    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    if start_button:
        st.session_state['run_camera'] = True

    if stop_button:
        st.session_state['run_camera'] = False

    if 'run_camera' not in st.session_state:
        st.session_state['run_camera'] = False

    if st.session_state['run_camera']:
        cap = cv2.VideoCapture(0)
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

        drowsy_start = yawn_start = distract_start = None
        alert_active = {"drowsy": False, "yawn": False, "distract": False}

        def beep():
            try:
                winsound.Beep(BEEP_FREQ, BEEP_DUR)
            except:
                pass

        while st.session_state['run_camera']:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not detected!")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            h, w = frame.shape[:2]
            now = time.time()

            # Detection Logic (Simplified Simulation for UI)
            if results.multi_face_landmarks:
                # Simulated yawning detection (open mouth)
                landmarks = results.multi_face_landmarks[0].landmark
                mouth_top = landmarks[13].y * h
                mouth_bottom = landmarks[14].y * h
                mar = mouth_bottom - mouth_top

                if mar > 25:
                    if yawn_start is None:
                        yawn_start = now
                    elif now - yawn_start > 5 and not alert_active["yawn"]:
                        alert_active["yawn"] = True
                        beep()
                        status_placeholder.error("😮 Yawning Detected!")
                        with open(LOG_FILE, "a") as f:
                            f.write(f"{datetime.datetime.now()},Yawning\n")
                else:
                    yawn_start = None
                    alert_active["yawn"] = False

                # Simulated drowsiness detection (eyes closed)
                left_eye = landmarks[159].y - landmarks[145].y
                if left_eye < 0.002:
                    if drowsy_start is None:
                        drowsy_start = now
                    elif now - drowsy_start > 5 and not alert_active["drowsy"]:
                        alert_active["drowsy"] = True
                        beep()
                        status_placeholder.error("😴 Drowsiness Detected!")
                        with open(LOG_FILE, "a") as f:
                            f.write(f"{datetime.datetime.now()},Drowsiness\n")
                else:
                    drowsy_start = None
                    alert_active["drowsy"] = False

                # Simulated distraction detection (face turned)
                nose = landmarks[1].x
                if nose < 0.35 or nose > 0.65:
                    if distract_start is None:
                        distract_start = now
                    elif now - distract_start > 5 and not alert_active["distract"]:
                        alert_active["distract"] = True
                        beep()
                        status_placeholder.error("⚠ Distraction Detected!")
                        with open(LOG_FILE, "a") as f:
                            f.write(f"{datetime.datetime.now()},Distraction\n")
                else:
                    distract_start = None
                    alert_active["distract"] = False

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB", width=700)

        cap.release()
        st.success("Camera stopped successfully.")
        face_mesh.close()

# ==========================================================
# TAB 2 — ANALYTICS DASHBOARD
# ==========================================================
with tabs[1]:
    st.header("📈 Detection Analytics & History")

    if not os.path.exists(LOG_FILE) or os.stat(LOG_FILE).st_size == 0:
        st.warning("No detection logs found yet. Run detection first.")
    else:
        df = pd.read_csv(LOG_FILE, names=["Timestamp", "Event"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

        st.subheader("🗓️ Event Timeline")
        st.line_chart(df.groupby(df["Timestamp"].dt.minute)["Event"].count())

        st.subheader("📊 Event Count Summary")
        counts = df["Event"].value_counts()
        st.bar_chart(counts)

        st.subheader("🥧 Event Distribution")
        fig, ax = plt.subplots()
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
        st.pyplot(fig)

        st.subheader("🕒 Recent Logs")
        st.dataframe(df.tail(10), use_container_width=True)
