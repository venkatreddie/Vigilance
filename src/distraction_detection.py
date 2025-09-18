import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import csv
import os

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# CSV log file
log_file = "log.csv"
if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Alert_Type", "Details"])

def log_event(alert_type, details):
    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), alert_type, details])

# Camera setup
cap = cv2.VideoCapture(0)

# 3D model points for head pose estimation
model_points = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),   # Right eye right corner
    (-150.0, -150.0, -125.0), # Left Mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
], dtype=np.float64)

# For reducing false triggers
yaw_buffer, pitch_buffer = [], []
buffer_size = 5  

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    alert_triggered = False
    details = ""

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        # Get 2D points
        image_points = np.array([
            (landmarks[1].x * w, landmarks[1].y * h),     # Nose tip
            (landmarks[152].x * w, landmarks[152].y * h), # Chin
            (landmarks[33].x * w, landmarks[33].y * h),   # Left eye left corner
            (landmarks[263].x * w, landmarks[263].y * h), # Right eye right corner
            (landmarks[61].x * w, landmarks[61].y * h),   # Left Mouth corner
            (landmarks[291].x * w, landmarks[291].y * h)  # Right mouth corner
        ], dtype=np.float64)

        # Camera intrinsics
        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4,1))

        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        yaw, pitch, roll = [a * 180 for a in angles]

        # Smooth values with buffer
        yaw_buffer.append(yaw)
        pitch_buffer.append(pitch)
        if len(yaw_buffer) > buffer_size:
            yaw_buffer.pop(0)
            pitch_buffer.pop(0)

        avg_yaw = np.mean(yaw_buffer)
        avg_pitch = np.mean(pitch_buffer)

        # Show angles
        cv2.putText(frame, f"Yaw: {avg_yaw:.1f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Pitch: {avg_pitch:.1f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Distraction conditions
        if abs(avg_yaw) > 20:
            alert_triggered = True
            details = "Looking Left" if avg_yaw < 0 else "Looking Right"
        elif avg_pitch < -10:  # head down threshold fixed
            alert_triggered = True
            details = "Looking Down"

    # Show alert continuously
    if alert_triggered:
        # Draw a red sign board
        cv2.rectangle(frame, (80, 40), (560, 120), (0, 0, 255), -1)
        cv2.putText(frame, "DISTRACTION DETECTED!", (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Play sound
        winsound.Beep(1000, 300)

        # Log continuously
        log_event("DISTRACTION", details)

    cv2.imshow("Driver Vigilance - Distraction Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
