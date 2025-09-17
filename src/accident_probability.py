import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import winsound

EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.65
RISK_HIGH = 0.7
RISK_MEDIUM = 0.4

LOG_FILE = "log.csv"
with open(LOG_FILE, mode="a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Event", "Risk Score"])

def play_beep():
    winsound.Beep(1000, 500)

def log_event(event_type, risk_score):
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), event_type, f"{risk_score:.2f}"])

def euclidean_dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def calculate_EAR(landmarks, eye_indices):
    p1, p2, p3, p4, p5, p6 = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_indices]
    vertical1 = euclidean_dist(p2, p6)
    vertical2 = euclidean_dist(p3, p5)
    horizontal = euclidean_dist(p1, p4)
    return (vertical1 + vertical2) / (2.0 * horizontal)

def calculate_MAR(landmarks, mouth_indices):
    top = np.array([landmarks[mouth_indices[13]].x, landmarks[mouth_indices[13]].y])
    bottom = np.array([landmarks[mouth_indices[14]].x, landmarks[mouth_indices[14]].y])
    left = np.array([landmarks[mouth_indices[78]].x, landmarks[mouth_indices[78]].y])
    right = np.array([landmarks[mouth_indices[308]].x, landmarks[mouth_indices[308]].y])
    vertical = euclidean_dist(top, bottom)
    horizontal = euclidean_dist(left, right)
    return vertical / horizontal

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
hands = mp_hands.Hands()

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH = [13, 14, 78, 308]

cap = cv2.VideoCapture(0)
last_alert_time = 0
cooldown = 5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    EAR, MAR = None, None
    drowsy, yawning, phone_use = False, False, False

    if face_results.multi_face_landmarks:
        for landmarks in face_results.multi_face_landmarks:
            lm = landmarks.landmark
            EAR = (calculate_EAR(lm, LEFT_EYE) + calculate_EAR(lm, RIGHT_EYE)) / 2.0
            MAR = calculate_MAR(lm, MOUTH)
            if EAR < EAR_THRESHOLD: drowsy = True
            if MAR > MAR_THRESHOLD: yawning = True

    if hand_results.multi_hand_landmarks and face_results.multi_face_landmarks:
        phone_use = True

    risk_score = 0
    if drowsy: risk_score += 0.5
    if yawning: risk_score += 0.3
    if phone_use: risk_score += 0.4
    risk_score = min(risk_score, 1.0)

    risk_percent = int(risk_score * 100)

    current_time = time.time()
    if risk_score >= RISK_HIGH and current_time - last_alert_time > cooldown:
        cv2.putText(frame, "🚨 HIGH RISK!", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        play_beep()
        log_event("HIGH_RISK", risk_score)
        last_alert_time = current_time
    elif risk_score >= RISK_MEDIUM:
        cv2.putText(frame, "⚠ Medium Risk", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        log_event("MEDIUM_RISK", risk_score)
    else:
        cv2.putText(frame, "✅ Low Risk", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # -------------------
    # Draw probability bar
    # -------------------
    bar_x, bar_y = 50, h - 100
    bar_w, bar_h = 400, 30

    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), -1)

    # Fill color depends on risk
    if risk_score >= RISK_HIGH:
        color = (0, 0, 255)  # red
    elif risk_score >= RISK_MEDIUM:
        color = (0, 255, 255)  # yellow
    else:
        color = (0, 255, 0)  # green

    fill_w = int(bar_w * risk_score)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)

    # Outline
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 0, 0), 2)

    # Text
    cv2.putText(frame, f"Risk: {risk_percent}%", (bar_x, bar_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Accident Probability Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
