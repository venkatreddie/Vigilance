import cv2
import mediapipe as mp
import numpy as np
import winsound
from collections import deque

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                  refine_landmarks=True, min_detection_confidence=0.5)

# Eye & Mouth landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144] 
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 81, 91, 146, 181, 311, 321, 375, 405, 78, 82, 87, 13, 14, 17]

# Calibration settings
CALIBRATION_FRAMES = 50
ear_values = []
mar_values = []
calibrated = False
EAR_THRESHOLD = 0.18
MAR_THRESHOLD = 0.6

# Frame counters
EAR_CONSEC_FRAMES = 15
COUNTER = 0

# For smoothing EAR
EAR_queue = deque(maxlen=5)

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def calculate_EAR(landmarks, eye_indices):
    eye = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])
    A = euclidean_distance(eye[1], eye[5])
    B = euclidean_distance(eye[2], eye[4])
    C = euclidean_distance(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_MAR(landmarks, mouth_indices):
    mouth = np.array([(landmarks[i].x, landmarks[i].y) for i in mouth_indices])
    A = euclidean_distance(mouth[13], mouth[14])  # vertical
    B = euclidean_distance(mouth[0], mouth[6])    # horizontal
    mar = A / B
    return mar

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            leftEAR = calculate_EAR(landmarks, LEFT_EYE)
            rightEAR = calculate_EAR(landmarks, RIGHT_EYE)
            EAR = (leftEAR + rightEAR) / 2.0

            MAR = calculate_MAR(landmarks, MOUTH)

            # Calibration phase
            if not calibrated:
                ear_values.append(EAR)
                mar_values.append(MAR)

                cv2.putText(frame, "Calibrating... Keep eyes open, mouth closed",
                            (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if len(ear_values) >= CALIBRATION_FRAMES:
                    avg_EAR = np.mean(ear_values)
                    avg_MAR = np.mean(mar_values)

                    EAR_THRESHOLD = avg_EAR * 0.75  # 25% lower than normal open EAR
                    MAR_THRESHOLD = avg_MAR * 1.8   # 80% higher than normal MAR

                    calibrated = True
                    print(f"Calibration complete → EAR_THRESHOLD={EAR_THRESHOLD:.3f}, MAR_THRESHOLD={MAR_THRESHOLD:.3f}")
            else:
                # Apply smoothing to EAR
                EAR_queue.append(EAR)
                EAR_avg = sum(EAR_queue) / len(EAR_queue)

                # Drowsiness detection
                if EAR_avg < EAR_THRESHOLD:
                    COUNTER += 1
                    if COUNTER >= EAR_CONSEC_FRAMES:
                        cv2.putText(frame, "DROWSINESS DETECTED", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        winsound.Beep(2500, 500)
                else:
                    COUNTER = 0

                # Yawning detection
                if MAR > MAR_THRESHOLD:
                    cv2.putText(frame, "YAWNING DETECTED", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    winsound.Beep(2000, 500)

    cv2.imshow('Driver Vigilance Monitoring', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

