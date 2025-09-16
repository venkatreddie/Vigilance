import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
import winsound
from datetime import datetime

# ---------------- Mediapipe Setup ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# EAR & MAR landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 81, 311, 291, 402, 14, 178, 17]

# ---------------- Utility Functions ----------------
def euclidean_dist(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def calculate_EAR(landmarks, eye_indices, image_w, image_h):
    coords = [(int(landmarks[i].x * image_w), int(landmarks[i].y * image_h)) for i in eye_indices]
    A = euclidean_dist(coords[1], coords[5])
    B = euclidean_dist(coords[2], coords[4])
    C = euclidean_dist(coords[0], coords[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_MAR(landmarks, mouth_indices, image_w, image_h):
    coords = [(int(landmarks[i].x * image_w), int(landmarks[i].y * image_h)) for i in mouth_indices]
    A = euclidean_dist(coords[1], coords[5])
    B = euclidean_dist(coords[2], coords[4])
    C = euclidean_dist(coords[0], coords[3])
    mar = (A + B) / (2.0 * C)
    return mar

# ---------------- Calibration ----------------
def calibrate(cap, duration=5):
    print("Calibration started... Keep your face relaxed & eyes open.")
    start = time.time()
    ear_vals, mar_vals = [], []

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            leftEAR = calculate_EAR(lm, LEFT_EYE, w, h)
            rightEAR = calculate_EAR(lm, RIGHT_EYE, w, h)
            mar = calculate_MAR(lm, MOUTH, w, h)
            ear_vals.append((leftEAR + rightEAR) / 2.0)
            mar_vals.append(mar)
            cv2.putText(frame, "Calibrating...", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyWindow("Calibration")
    ear_thr = np.mean(ear_vals) * 0.85
    mar_thr = np.mean(mar_vals) * 1.3
    print(f"Calibration complete → EAR_THRESHOLD={ear_thr:.3f}, MAR_THRESHOLD={mar_thr:.3f}")
    return ear_thr, mar_thr

# ---------------- CSV Logger ----------------
def log_alert(alert_type):
    log_file = "log.csv"
    exists = os.path.isfile(log_file)
    with open(log_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["Timestamp", "Alert"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), alert_type])

# ---------------- Main ----------------
def main():
    cap = cv2.VideoCapture(0)
    EAR_THRESHOLD, MAR_THRESHOLD = calibrate(cap)

    last_alert = 0
    strong_alert_start = None
    COOLDOWN = 5  # seconds
    STRONG_ALERT_TIME = 15  # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            leftEAR = calculate_EAR(lm, LEFT_EYE, w, h)
            rightEAR = calculate_EAR(lm, RIGHT_EYE, w, h)
            ear = (leftEAR + rightEAR) / 2.0
            mar = calculate_MAR(lm, MOUTH, w, h)

            # Display EAR & MAR values live
            cv2.putText(frame, f"EAR: {ear:.3f} (Thr={EAR_THRESHOLD:.3f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.3f} (Thr={MAR_THRESHOLD:.3f})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            alert = None
            if ear < EAR_THRESHOLD:
                alert = "DROWSINESS DETECTED"
            elif mar > MAR_THRESHOLD:
                alert = "YAWNING DETECTED"

            if alert:
                now = time.time()
                if now - last_alert > COOLDOWN:
                    cv2.putText(frame, alert, (50, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    winsound.Beep(2500, 800)
                    log_alert(alert)
                    last_alert = now
                    strong_alert_start = strong_alert_start or now

                # Stronger alert after 15s continuous detection
                if strong_alert_start and now - strong_alert_start > STRONG_ALERT_TIME:
                    cv2.putText(frame, "STRONG ALERT! WAKE UP!", (50, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    winsound.Beep(3000, 1200)
                    log_alert("STRONG ALERT")
                    strong_alert_start = None
            else:
                strong_alert_start = None

        cv2.imshow("Driver Vigilance - Alerts", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
