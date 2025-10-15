import cv2
import numpy as np
import streamlit as st
import time
import pygame
from datetime import datetime

# Initialize pygame for sound
pygame.mixer.init()

# Sound alert file (use any short beep sound)
ALERT_SOUND = "alert.wav"

# Load Haar cascades for face, eyes, and mouth
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# Initialize detection timers
drowsy_start = None
yawn_start = None
distraction_start = None

def play_alert():
    pygame.mixer.music.load(ALERT_SOUND)
    pygame.mixer.music.play()

def detect_drowsiness(eyes):
    """If no eyes detected, assume possible drowsiness."""
    return len(eyes) == 0

def detect_yawning(mouth):
    """If mouth region large enough, assume yawning."""
    for (x, y, w, h) in mouth:
        if h > 30:  # adjust based on camera distance
            return True
    return False

def detect_distraction(face_detected):
    """If no face detected for a while -> distraction."""
    return not face_detected

def log_event(event_type, logs):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logs.append(f"{timestamp} - {event_type}")
    return logs

def main():
    st.set_page_config(page_title="Driver Vigilance Detection", layout="wide")
    st.title("🚗 Driver Vigilance Monitoring System")

    st.sidebar.header("Settings")
    detection_mode = st.sidebar.radio("Select Detection Mode", ["All", "Drowsiness", "Yawning", "Distraction"])
    st.sidebar.info("Alerts will trigger after 5 seconds of continuous detection.")

    start_button = st.sidebar.button("Start Camera")
    stop_button = st.sidebar.button("Stop Camera")

    if "run" not in st.session_state:
        st.session_state.run = False
    if "logs" not in st.session_state:
        st.session_state.logs = []

    if start_button:
        st.session_state.run = True
    if stop_button:
        st.session_state.run = False

    frame_window = st.empty()
    status_placeholder = st.empty()

    if st.session_state.run:
        cap = cv2.VideoCapture(0)

        global drowsy_start, yawn_start, distraction_start
        drowsy_start = yawn_start = distraction_start = None

        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Camera not accessible!")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            drowsy = yawn = distracted = False

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = frame[y:y+h, x:x+w]

                    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
                    mouth = mouth_cascade.detectMultiScale(roi_gray, 1.5, 15)

                    # Detection logic
                    if detection_mode in ["All", "Drowsiness"]:
                        if detect_drowsiness(eyes):
                            if drowsy_start is None:
                                drowsy_start = time.time()
                            elif time.time() - drowsy_start > 5:
                                drowsy = True
                        else:
                            drowsy_start = None

                    if detection_mode in ["All", "Yawning"]:
                        if detect_yawning(mouth):
                            if yawn_start is None:
                                yawn_start = time.time()
                            elif time.time() - yawn_start > 5:
                                yawn = True
                        else:
                            yawn_start = None

                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    for (mx, my, mw, mh) in mouth:
                        cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (255, 0, 0), 2)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

            else:
                if detection_mode in ["All", "Distraction"]:
                    if distraction_start is None:
                        distraction_start = time.time()
                    elif time.time() - distraction_start > 5:
                        distracted = True
                else:
                    distraction_start = None

            # Display status
            status = "✅ Normal"
            color = (0, 255, 0)
            if drowsy:
                status = "⚠️ Drowsiness Detected! Please Stay Alert!"
                color = (0, 0, 255)
                play_alert()
                st.session_state.logs = log_event("Drowsiness Detected", st.session_state.logs)

            elif yawn:
                status = "😮 Yawning Detected! Take a Break!"
                color = (0, 0, 255)
                play_alert()
                st.session_state.logs = log_event("Yawning Detected", st.session_state.logs)

            elif distracted:
                status = "🚫 Distraction Detected! Focus on the Road!"
                color = (0, 0, 255)
                play_alert()
                st.session_state.logs = log_event("Distraction Detected", st.session_state.logs)

            cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame, channels="RGB")

            status_placeholder.markdown(f"### Status: {status}")

            time.sleep(0.05)

        cap.release()

    st.sidebar.header("Detection Logs")
    st.sidebar.write("\n".join(st.session_state.logs[-10:]))

if __name__ == "__main__":
    main()
