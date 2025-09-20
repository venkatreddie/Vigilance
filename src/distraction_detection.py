import cv2
import mediapipe as mp
import time
import csv
import os
import winsound
import numpy as np
import math

# ====== CONFIG ======
LOG_FILE = "distraction_log.csv"

# Pose thresholds
DISTRACTION_YAW = 20.0      # side look
DISTRACTION_PITCH = 15.0    # looking down
STRAIGHT_YAW = 12.0
STRAIGHT_PITCH = 8.0

# Timing
DISTRACTION_DURATION = 10.0   # must be distracted for 10s before alert
ALERT_REPEAT_INTERVAL = 2.0   # repeat beep every 2s if still distracted

# Beep settings
BEEP_FREQ = 2000
BEEP_DUR_MS = 200

# ====== SETUP ======
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Yaw_deg", "Pitch_deg", "Event"])

def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])   # roll
        y = math.atan2(-R[2,0], sy)      # pitch
        z = math.atan2(R[1,0], R[0,0])   # yaw
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.degrees([x, y, z])  # roll, pitch, yaw

def beep():
    try:
        winsound.Beep(BEEP_FREQ, BEEP_DUR_MS)
    except Exception:
        pass

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)

LMKS_IDX = [1, 199, 33, 263, 61, 291]

alert_active = False
last_beep_time = 0.0
distraction_start_time = None
distraction_logged = False  # to prevent duplicate logs per event

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            try:
                image_points = np.array([
                    (lm[LMKS_IDX[0]].x * w, lm[LMKS_IDX[0]].y * h),
                    (lm[LMKS_IDX[1]].x * w, lm[LMKS_IDX[1]].y * h),
                    (lm[LMKS_IDX[2]].x * w, lm[LMKS_IDX[2]].y * h),
                    (lm[LMKS_IDX[3]].x * w, lm[LMKS_IDX[3]].y * h),
                    (lm[LMKS_IDX[4]].x * w, lm[LMKS_IDX[4]].y * h),
                    (lm[LMKS_IDX[5]].x * w, lm[LMKS_IDX[5]].y * h)
                ], dtype=np.float64)

                focal_length = w
                center = (w/2.0, h/2.0)
                camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype=np.float64)
                dist_coeffs = np.zeros((4,1))

                success, rotation_vector, translation_vector = cv2.solvePnP(
                    MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )

                if success:
                    R_mat, _ = cv2.Rodrigues(rotation_vector)
                    roll_deg, pitch_deg, yaw_deg = rotationMatrixToEulerAngles(R_mat)

                    cv2.putText(frame, f"Yaw:{yaw_deg:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    cv2.putText(frame, f"Pitch:{pitch_deg:.2f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                    now = time.time()
                    is_distracted = (abs(yaw_deg) > DISTRACTION_YAW) or (pitch_deg > DISTRACTION_PITCH)
                    is_straight = (abs(yaw_deg) <= STRAIGHT_YAW) and (abs(pitch_deg) <= STRAIGHT_PITCH)

                    if is_distracted:
                        if distraction_start_time is None:
                            distraction_start_time = now
                            distraction_logged = False  # reset logging
                        elif (now - distraction_start_time) >= DISTRACTION_DURATION:
                            if not alert_active:
                                alert_active = True
                                beep()
                                if not distraction_logged:
                                    with open(LOG_FILE, "a", newline="") as f:
                                        writer = csv.writer(f)
                                        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                                                         round(yaw_deg,2), round(pitch_deg,2), "Distraction Started"])
                                    distraction_logged = True
                                last_beep_time = now
                            elif (now - last_beep_time) >= ALERT_REPEAT_INTERVAL:
                                beep()
                                last_beep_time = now
                    elif is_straight:
                        if alert_active and distraction_logged:
                            with open(LOG_FILE, "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                                                 round(yaw_deg,2), round(pitch_deg,2), "Distraction Ended"])
                        alert_active = False
                        distraction_start_time = None
                        distraction_logged = False

            except Exception:
                cv2.putText(frame, "Pose error", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        else:
            alert_active = False
            distraction_start_time = None
            distraction_logged = False
            cv2.putText(frame, "No face detected", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Draw warning only when alert is active
        if alert_active:
            cv2.rectangle(frame, (80, 40), (w-80, 130), (0, 0, 255), -1)
            cv2.putText(frame, "⚠ DISTRACTION DETECTED ⚠", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 3)

        cv2.imshow("Distraction Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
