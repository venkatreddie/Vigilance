import cv2
import mediapipe as mp
import numpy as np
import time
import winsound

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Mouth indices
MOUTH = {
    "left": 61,
    "right": 291,
    "top_outer": 81,
    "bottom_outer": 178,
    "top_inner": 13,
    "bottom_inner": 14,
}

# Eye indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Distance helper
def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

# EAR calculation
def calculate_EAR(landmarks, eye_indices):
    A = euclidean_dist(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    B = euclidean_dist(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    C = euclidean_dist(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    return (A + B) / (2.0 * C)

# MAR calculation (fixed)
def calculate_MAR(landmarks):
    A = euclidean_dist(landmarks[MOUTH["top_inner"]], landmarks[MOUTH["bottom_inner"]])
    B = euclidean_dist(landmarks[MOUTH["top_outer"]], landmarks[MOUTH["bottom_outer"]])
    C = euclidean_dist(landmarks[MOUTH["left"]], landmarks[MOUTH["right"]])
    return (A + B) / (2.0 * C)

# Calibration
def calibrate(cap, duration=5):
    print("Calibration started... Look straight, keep eyes open, lips closed.")
    start = time.time()
    ear_vals, mar_vals = [], []
    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            ear = (calculate_EAR(lm, LEFT_EYE) + calculate_EAR(lm, RIGHT_EYE)) / 2.0
            mar = calculate_MAR(lm)
            ear_vals.append(ear)
            mar_vals.append(mar)
    ear_thresh = np.mean(ear_vals) * 0.8   # 80% of open eye EAR
    mar_base = np.mean(mar_vals)           # baseline closed-mouth MAR
    mar_thresh = max(mar_base * 1.8, 0.5)  # yawning only if much larger
    print(f"Calibration complete → EAR_THRESHOLD={ear_thresh:.3f}, MAR_THRESHOLD={mar_thresh:.3f}")
    return ear_thresh, mar_thresh

# Main
cap = cv2.VideoCapture(0)
EAR_THRESHOLD, MAR_THRESHOLD = calibrate(cap)

FRAME_CHECK = 20
ear_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lm = face_landmarks.landmark

            # EAR
            ear_left = calculate_EAR(lm, LEFT_EYE)
            ear_right = calculate_EAR(lm, RIGHT_EYE)
            ear_avg = (ear_left + ear_right) / 2.0

            # MAR
            mar = calculate_MAR(lm)

            # Drowsiness detection
            if ear_avg < EAR_THRESHOLD:
                ear_counter += 1
                if ear_counter >= FRAME_CHECK:
                    cv2.putText(frame, "DROWSINESS DETECTED!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    winsound.Beep(1000, 800)
            else:
                ear_counter = 0

            # Yawning detection
            if mar > MAR_THRESHOLD:
                cv2.putText(frame, "YAWNING DETECTED!", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                winsound.Beep(1500, 800)

            # Display ratios
            cv2.putText(frame, f"EAR: {ear_avg:.3f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.3f}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Driver Vigilance System", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
