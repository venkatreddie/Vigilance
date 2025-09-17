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
CLOSED_CONSEC_FRAMES = 15  # Frames for drowsiness detection
LOG_FILE = "log.csv"
CALIBRATION_TIME = 5  # Seconds for auto-calibration

# ===============================
# MEDIAPIPE
# ===============================
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
hands = mp_hands.Hands(max_num_hands=1)

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

def log_event(event_type, probability, suggestion=""):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Event", "Probability", "Suggestion"])
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), event_type, probability, suggestion])

# ===============================
# MAIN LOOP
# ===============================
cap = cv2.VideoCapture(0)

closed_frames = 0
probability = 0
live_suggestion = "Drive Safe!"  # Default suggestion

# Calibration variables
ear_values = []
mar_values = []
EAR_THRESHOLD, MAR_THRESHOLD = None, None
calibration_start = time.time()

print("🔧 Calibration started... Keep eyes open & mouth closed...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = face_mesh.process(rgb_frame)
    results_hands = hands.process(rgb_frame)

    alert_triggered = False  

    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            lm = face_landmarks.landmark

            left_EAR = calculate_EAR(lm, LEFT_EYE)
            right_EAR = calculate_EAR(lm, RIGHT_EYE)
            EAR = (left_EAR + right_EAR) / 2.0
            MAR = calculate_MAR(lm, MOUTH)

            if time.time() - calibration_start < CALIBRATION_TIME:
                ear_values.append(EAR)
                mar_values.append(MAR)
                cv2.putText(frame, "Calibrating... Keep eyes open, mouth closed", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                if EAR_THRESHOLD is None and MAR_THRESHOLD is None:
                    EAR_THRESHOLD = np.mean(ear_values) * 0.75
                    MAR_THRESHOLD = np.mean(mar_values) * 1.5
                    print(f"✅ Calibration complete. EAR_THRESHOLD={EAR_THRESHOLD:.2f}, MAR_THRESHOLD={MAR_THRESHOLD:.2f}")

                cv2.putText(frame, f"EAR: {EAR:.2f}", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"MAR: {MAR:.2f}", (50, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Drowsiness detection
                if EAR < EAR_THRESHOLD:
                    closed_frames += 1
                    if closed_frames >= CLOSED_CONSEC_FRAMES:
                        cv2.putText(frame, "DROWSINESS DETECTED!", (50, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        winsound.Beep(1000, 500)
                        probability = min(probability + 10, 100)
                        live_suggestion = "⚠ Take a Break!"
                        log_event("Drowsiness", probability, live_suggestion)
                        alert_triggered = True
                else:
                    closed_frames = 0

                # Yawning detection
                if MAR > MAR_THRESHOLD:
                    cv2.putText(frame, "YAWNING DETECTED!", (50, 190),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    winsound.Beep(1500, 500)
                    probability = min(probability + 5, 100)
                    live_suggestion = "😮 Stay Hydrated!"
                    log_event("Yawning", probability, live_suggestion)
                    alert_triggered = True

    # Phone usage detection
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            index_finger = hand_landmarks.landmark[8]
            if results_face.multi_face_landmarks:
                nose = results_face.multi_face_landmarks[0].landmark[1]
                dist = euclidean_distance(np.array([index_finger.x, index_finger.y]),
                                          np.array([nose.x, nose.y]))
                if dist < 0.1:
                    cv2.putText(frame, "PHONE USAGE DETECTED!", (50, 230),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    winsound.Beep(2000, 500)
                    probability = min(probability + 15, 100)
                    live_suggestion = "📵 Avoid Phone Usage!"
                    log_event("Phone Usage", probability, live_suggestion)
                    alert_triggered = True

    # Probability decay
    if not alert_triggered and probability > 0:
        probability = max(probability - 1, 0)
        live_suggestion = "✅ Stay Focused!"

    # Probability bar
    cv2.rectangle(frame, (50, 30), (350, 60), (255, 255, 255), -1)
    cv2.rectangle(frame, (50, 30), (50 + int(3 * probability), 60), (0, 0, 255), -1)
    cv2.putText(frame, f"Accident Risk: {probability}%", (360, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Live suggestion below bar
    cv2.putText(frame, live_suggestion, (50, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if "✅" in live_suggestion else (0, 0, 255), 2)

    # Show output
    cv2.imshow("Driver Vigilance Monitoring", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break
    elif key == ord("c"):
        calibration_start = time.time()
        EAR_THRESHOLD, MAR_THRESHOLD = None, None
        ear_values, mar_values = [], []
        print("🔄 Re-calibration started...")

cap.release()
cv2.destroyAllWindows()
