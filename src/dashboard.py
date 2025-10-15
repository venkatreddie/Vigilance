import cv2
import streamlit as st
import time
import pygame
from datetime import datetime

# Initialize pygame for sound
pygame.mixer.init()
ALERT_SOUND = "alert.wav"

def play_alert():
    try:
        pygame.mixer.music.load(ALERT_SOUND)
        pygame.mixer.music.play()
    except Exception:
        pass

# Load Haar cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

def log_event(event_type, logs):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logs.append(f"{timestamp} - {event_type}")
    return logs

def main():
    st.set_page_config(page_title="Driver Vigilance Detection", layout="wide")
    st.title("🚗 Driver Vigilance Detection Dashboard")

    st.sidebar.header("Settings")
    detection_mode = st.sidebar.radio("Select Detection Mode", ["All", "Drowsiness", "Yawning", "Distraction"])
    st.sidebar.info("Alerts trigger after 5 seconds of continuous detection.")

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

    cap = None
    drowsy_start = None
    yawn_start = None
    distraction_start = None

    if st.session_state.run:
        cap = cv2.VideoCapture(0)
        st.success("Camera started successfully! Please wait...")

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

                    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(30, 30))
                    mouths = mouth_cascade.detectMultiScale(roi_gray, 1.5, 11)

                    # Drowsiness detection (no eyes)
                    if detection_mode in ["All", "Drowsiness"]:
                        if len(eyes) == 0:
                            if drowsy_start is None:
                                drowsy_start = time.time()
                            elif time.time() - drowsy_start > 5:
                                drowsy = True
                        else:
                            drowsy_start = None

                    # Yawning detection (mouth detected)
                    if detection_mode in ["All", "Yawning"]:
                        if len(mouths) > 0:
                            # filter false detections by position
                            for (mx, my, mw, mh) in mouths:
                                if y + my > y + h/2:  # ensure it's lower half
                                    if yawn_start is None:
                                        yawn_start = time.time()
                                    elif time.time() - yawn_start > 5:
                                        yawn = True
                                    break
                        else:
                            yawn_start = None

                    # Draw detections
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    for (mx, my, mw, mh) in mouths:
                        cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (255, 0, 0), 2)

            else:
                # Distraction detection (no face for 5 sec)
                if detection_mode in ["All", "Distraction"]:
                    if distraction_start is None:
                        distraction_start = time.time()
                    elif time.time() - distraction_start > 5:
                        distracted = True
                else:
                    distraction_start = None

            # Alerts and UI updates
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
