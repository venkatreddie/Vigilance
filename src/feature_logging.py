import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import winsound  # For beep sound

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Eye & Mouth indices for Mediapipe
LEFT_EYE = [33, 160, 158, 133, 153, 144]   # left eye landmarks
RIGHT_EYE = [362, 385, 387, 263, 373, 380] # right eye landmarks
MOUTH = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
         308, 415, 310, 311, 312, 13]       # mouth landmarks

# EAR calculation
def calculate_EAR(landmarks, eye_indices):
    p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    p2 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    p5 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    p6 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])

    vertical1 = np.linalg.norm(p2 - p5)
    vertical2 = np.linalg.norm(p3 - p4)
    horizontal = np.linalg.norm(p1 - p6)


    EAR = (vertical1 + vertical2) / (2.0 * horizontal)
    return EAR

# MAR calculation
def calculate_MAR(landmarks):
    top_lip = np.array([landmarks[13].x, landmarks[13].y])
    bottom_lip = np.array([landmarks[14].x, landmarks[14].y])
    left_corner = np.array([landmarks[78].x, landmarks[78].y])
    right_corner = np.array([landmarks[308].x, landmarks[308].y])

    vertical = np.linalg.norm(top_lip - bottom_lip)
    horizontal = np.linalg.norm(left_corner - right_corner)

    MAR = vertical / horizontal
    return MAR

# Beep alert
def play_alert():
    frequency = 1000  # 1kHz
    duration = 300    # 300 ms
    winsound.Beep(frequency, duration)

# CSV logging setup
csv_file = "../data/feature_log.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "EAR", "MAR"])

# Thresholds
EAR_THRESHOLD = 0.21  # Adjusted for Mediapipe normalized coords
MAR_THRESHOLD = 0.6
CONSEC_FRAMES = 10

ear_counter = 0
mar_counter = 0

# Video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Calculate EAR
            leftEAR = calculate_EAR(face_landmarks.landmark, LEFT_EYE)
            rightEAR = calculate_EAR(face_landmarks.landmark, RIGHT_EYE)
            EAR = (leftEAR + rightEAR) / 2.0

            # Calculate MAR
            MAR = calculate_MAR(face_landmarks.landmark)

            # Log into CSV
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([time.time(), EAR, MAR])

            # Drowsiness detection
            if EAR < EAR_THRESHOLD:
                ear_counter += 1
                if ear_counter >= CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS DETECTED!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    play_alert()
            else:
                ear_counter = 0

            # Yawning detection
            if MAR > MAR_THRESHOLD:
                mar_counter += 1
                if mar_counter >= CONSEC_FRAMES:
                    cv2.putText(frame, "YAWNING DETECTED!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    play_alert()
            else:
                mar_counter = 0

    cv2.imshow("Driver Vigilance Monitoring", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
