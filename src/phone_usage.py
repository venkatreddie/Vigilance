import cv2
import mediapipe as mp
import winsound
import time
import csv
import os

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Log file path
log_file = "log.csv"

# Ensure log file exists
if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Alert_Type", "Message"])

# Function to log events
def log_event(alert_type, message):
    with open(log_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), alert_type, message])

# Beep sound
def play_beep():
    winsound.Beep(1000, 500)  # frequency=1000Hz, duration=500ms

# Start webcam
cap = cv2.VideoCapture(0)

cooldown = 5  # seconds
last_alert_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process detections
    result_hands = hands.process(rgb_frame)
    result_face = face_mesh.process(rgb_frame)

    phone_usage_detected = False

    if result_hands.multi_hand_landmarks and result_face.multi_face_landmarks:
        # Get face center (nose tip)
        for face_landmarks in result_face.multi_face_landmarks:
            nose = face_landmarks.landmark[1]  # Nose tip
            nose_x, nose_y = int(nose.x * frame.shape[1]), int(nose.y * frame.shape[0])

        # Check each hand
        for hand_landmarks in result_hands.multi_hand_landmarks:
            # Take index finger tip
            finger_tip = hand_landmarks.landmark[8]
            finger_x, finger_y = int(finger_tip.x * frame.shape[1]), int(finger_tip.y * frame.shape[0])

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Distance between finger and nose
            distance = ((finger_x - nose_x) ** 2 + (finger_y - nose_y) ** 2) ** 0.5

            if distance < 100:  # threshold in pixels (adjust if needed)
                phone_usage_detected = True
                break

    # Trigger alerts only if phone usage detected
    if phone_usage_detected:
        current_time = time.time()
        if current_time - last_alert_time > cooldown:
            play_beep()
            log_event("PHONE_USAGE", "Driver may be using phone (hand near face)")
            last_alert_time = current_time

        # Show red alert text
        cv2.putText(frame, "WARNING: PHONE USAGE DETECTED!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

    # Show window
    cv2.imshow("Driver Monitoring - Phone Usage", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
