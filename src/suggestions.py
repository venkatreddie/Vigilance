import cv2
import mediapipe as mp
import time
import csv
import math

# Initialize mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Open video
cap = cv2.VideoCapture(0)

# CSV log file
def log_event(alert_type, suggestion):
    with open("suggestions_log.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), alert_type, suggestion])

# Function to calculate distance
def euclidean_dist(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

# Mouth aspect ratio (MAR) for yawning detection
def mouth_aspect_ratio(landmarks):
    top = landmarks[13]   # upper lip
    bottom = landmarks[14] # lower lip
    left = landmarks[78]   # left mouth corner
    right = landmarks[308] # right mouth corner

    vertical = euclidean_dist(top, bottom)
    horizontal = euclidean_dist(left, right)

    return vertical / horizontal

# Eye aspect ratio (EAR) for drowsiness detection
def eye_aspect_ratio(landmarks):
    left_eye_top = landmarks[159]
    left_eye_bottom = landmarks[145]
    left_eye_left = landmarks[133]
    left_eye_right = landmarks[33]

    vertical = euclidean_dist(left_eye_top, left_eye_bottom)
    horizontal = euclidean_dist(left_eye_left, left_eye_right)

    return vertical / horizontal

# Thresholds
MAR_THRESH = 0.6   # yawning
EAR_THRESH = 0.2   # drowsiness
FRAME_CHECK = 20

counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Calculate EAR and MAR
            mar = mouth_aspect_ratio(landmarks)
            ear = eye_aspect_ratio(landmarks)

            suggestion = None

            # Drowsiness detection
            if ear < EAR_THRESH:
                counter += 1
                if counter > FRAME_CHECK:
                    suggestion = "⚠️ You look drowsy! Please take a break."
            else:
                counter = 0

            # Yawning detection
            if mar > MAR_THRESH:
                suggestion = "⚠️ You are yawning! Please stay alert."

            # Show suggestion live
            if suggestion:
                cv2.putText(frame, suggestion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2, cv2.LINE_AA)
                log_event("SUGGESTION", suggestion)

    # Display video
    cv2.imshow("Driver Vigilance - Live Suggestions", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
