import cv2
import mediapipe as mp
import numpy as np
import winsound
import csv
from datetime import datetime

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)

# EAR / MAR landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

# Thresholds
EAR_THRESHOLD = 0.25   # Eye Aspect Ratio
MAR_THRESHOLD = 0.6    # Mouth Aspect Ratio

# Accident probability
probability = 0

# --- CSV Logging ---
log_file = "log.csv"
with open(log_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Time", "Alert_Type", "Accident_Probability"])

def log_event(alert_type, probability):
    """Log events with timestamp into CSV"""
    with open(log_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), alert_type, probability])

# --- Utility Functions ---
def euclidean_dist(a, b):
    return np.linalg.norm(a - b)

def calculate_EAR(landmarks, eye_indices):
    left = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    right = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    top1 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    top2 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    bottom1 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    bottom2 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
    vertical = (euclidean_dist(top1, bottom1) + euclidean_dist(top2, bottom2)) / 2.0
    horizontal = euclidean_dist(left, right)
    return vertical / horizontal

def calculate_MAR(landmarks):
    top = np.array([landmarks[MOUTH_TOP].x, landmarks[MOUTH_TOP].y])
    bottom = np.array([landmarks[MOUTH_BOTTOM].x, landmarks[MOUTH_BOTTOM].y])
    left = np.array([landmarks[MOUTH_LEFT].x, landmarks[MOUTH_LEFT].y])
    right = np.array([landmarks[MOUTH_RIGHT].x, landmarks[MOUTH_RIGHT].y])
    vertical = euclidean_dist(top, bottom)
    horizontal = euclidean_dist(left, right)
    return vertical / horizontal

# Video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process face and hands
    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    phone_usage = False
    EAR, MAR = None, None

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # EAR calculation
            EAR_left = calculate_EAR(landmarks, LEFT_EYE)
            EAR_right = calculate_EAR(landmarks, RIGHT_EYE)
            EAR = (EAR_left + EAR_right) / 2.0

            # MAR calculation
            MAR = calculate_MAR(landmarks)

            # Face bounding box
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            face_xmin, face_ymin = int(min(xs) * w), int(min(ys) * h)
            face_xmax, face_ymax = int(max(xs) * w), int(max(ys) * h)
            cv2.rectangle(frame, (face_xmin, face_ymin), (face_xmax, face_ymax), (255, 255, 255), 2)

            # --- Drowsiness Detection ---
            if EAR < EAR_THRESHOLD:
                cv2.putText(frame, "DROWSINESS DETECTED", (50, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                winsound.Beep(1000, 500)
                probability = min(probability + 5, 100)
                log_event("Drowsiness", probability)

            # --- Yawning Detection ---
            if MAR > MAR_THRESHOLD:
                cv2.putText(frame, "YAWNING DETECTED", (50, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                winsound.Beep(800, 500)
                probability = min(probability + 5, 100)
                log_event("Yawning", probability)

            # --- Phone Usage Detection ---
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        if face_xmin < x < face_xmax and face_ymin < y < face_ymax:
                            phone_usage = True

            if phone_usage:
                cv2.putText(frame, "PHONE USAGE DETECTED", (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                winsound.Beep(1200, 500)
                probability = min(probability + 10, 100)
                log_event("Phone Usage", probability)

    # Decrease probability if no issues
    if EAR is not None and MAR is not None and not phone_usage:
        probability = max(probability - 2, 0)

    # Probability bar (TOP SIDE of frame)
    bar_x, bar_y = 50, 40
    bar_width, bar_height = 400, 30
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + int(bar_width * (probability / 100)), bar_y + bar_height),
                  (0, 0, 255), -1)
    cv2.putText(frame, f"Accident Probability: {probability}%",
                (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Accident Probability Monitor", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
