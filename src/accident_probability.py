import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import csv
import os

# ===============================
# CONFIG
# ===============================
EAR_THRESHOLD = 0.20   # Adjust after testing EAR live values
MAR_THRESHOLD = 0.60   # Adjust after testing MAR live values
CLOSED_CONSEC_FRAMES = 15  # Number of frames for drowsiness detection

LOG_FILE = "log.csv"

# ===============================
# MEDIAPIPE
# ===============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH = [78, 308, 13, 14, 82, 312]

# ===============================
# FUNCTIONS
# ===============================
def euclidean_distance(pt1, pt2):
    return np.linalg.norm(pt1 - pt2)

def calculate_EAR(landmarks, eye_indices):
    eye = np.array([[landmarks[p].x, landmarks[p].y] for p in eye_indices])
    A = euclidean_distance(eye[1], eye[5])
    B = euclidean_distance(eye[2], eye[4])
    C = euclidean_distance(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def calculate_MAR(landmarks, mouth_indices):
    mouth = np.array([[landmarks[p].x, landmarks[p].y] for p in mouth_indices])
    A = euclidean_distance(mouth[1], mouth[5])
    B = euclidean_distance(mouth[0], mouth[3])
    return A / B

def log_event(event_type, probability):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Event", "Probability"])
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), event_type, probability])

# ===============================
# MAIN LOOP
# ===============================
cap = cv2.VideoCapture(0)

closed_frames = 0
probability = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lm = face_landmarks.landmark

            # EAR for both eyes
            left_EAR = calculate_EAR(lm, LEFT_EYE)
            right_EAR = calculate_EAR(lm, RIGHT_EYE)
            EAR = (left_EAR + right_EAR) / 2.0

            # MAR for mouth
            MAR = calculate_MAR(lm, MOUTH)

            # =======================
            # DISPLAY LIVE VALUES
            # =======================
            cv2.putText(frame, f"EAR: {EAR:.2f}", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"MAR: {MAR:.2f}", (50, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # =======================
            # DROWSINESS DETECTION
            # =======================
            if EAR < EAR_THRESHOLD:
                closed_frames += 1
                if closed_frames >= CLOSED_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS DETECTED!", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    winsound.Beep(1000, 500)
                    probability = min(probability + 10, 100)
                    log_event("Drowsiness", probability)
            else:
                closed_frames = 0

            # =======================
            # YAWNING DETECTION
            # =======================
            if MAR > MAR_THRESHOLD:
                cv2.putText(frame, "YAWNING DETECTED!", (50, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                winsound.Beep(1500, 500)
                probability = min(probability + 5, 100)
                log_event("Yawning", probability)

            # =======================
            # PROBABILITY BAR
            # =======================
            cv2.rectangle(frame, (50, 30), (350, 60), (255, 255, 255), -1)
            cv2.rectangle(frame, (50, 30), (50 + int(3 * probability), 60), (0, 0, 255), -1)
            cv2.putText(frame, f"Accident Risk: {probability}%", (360, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow("Driver Vigilance Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
