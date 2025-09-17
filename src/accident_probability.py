import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Indices for EAR and MAR
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]  # top, bottom, left, right lips

# Thresholds
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6

# Utility functions
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
    # Top and bottom lips
    top = np.array([landmarks[13].x, landmarks[13].y])
    bottom = np.array([landmarks[14].x, landmarks[14].y])
    # Left and right corners of mouth
    left = np.array([landmarks[78].x, landmarks[78].y])
    right = np.array([landmarks[308].x, landmarks[308].y])

    vertical = euclidean_dist(top, bottom)
    horizontal = euclidean_dist(left, right)
    return vertical / horizontal

# Accident probability tracker
probability = 0
last_update = time.time()

# Capture video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            EAR_left = calculate_EAR(landmarks, LEFT_EYE)
            EAR_right = calculate_EAR(landmarks, RIGHT_EYE)
            EAR = (EAR_left + EAR_right) / 2.0
            MAR = calculate_MAR(landmarks)

            # Update probability
            if EAR < EAR_THRESHOLD or MAR > MAR_THRESHOLD:
                probability = min(probability + 5, 100)
            else:
                probability = max(probability - 2, 0)

            # Draw alerts
            if EAR < EAR_THRESHOLD:
                cv2.putText(frame, "DROWSINESS DETECTED", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            if MAR > MAR_THRESHOLD:
                cv2.putText(frame, "YAWNING DETECTED", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # Draw probability bar
            bar_x, bar_y = 50, 250
            bar_width, bar_height = 400, 40
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + int(bar_width * (probability / 100)), bar_y + bar_height),
                          (0, 0, 255), -1)
            cv2.putText(frame, f"Accident Probability: {probability}%",
                        (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Accident Probability Monitor", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
