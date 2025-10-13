import cv2
import mediapipe as mp
import time
import csv
import os
import winsound
import numpy as np
import math

# ===== CONFIG =====
LOG_FILE = "distraction_log.csv"

# Head-pose thresholds
DISTRACTION_YAW = 20.0
DISTRACTION_PITCH = 15.0
STRAIGHT_YAW = 12.0
STRAIGHT_PITCH = 8.0
DISTRACTION_DURATION = 10.0
ALERT_REPEAT_INTERVAL = 2.0

# EAR/MAR thresholds
EAR_THRESHOLD = 0.22  # eyes closed
EAR_DURATION = 3.0    # seconds
MAR_THRESHOLD = 0.6   # yawning
MAR_DURATION = 2.0    # seconds

# Beep settings
BEEP_FREQ = 2000
BEEP_DUR_MS = 200

# ===== SETUP =====
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Yaw_deg", "Pitch_deg", "EAR", "MAR", "Event"])

# ===== FUNCTIONS =====
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

def eye_aspect_ratio(eye_landmarks):
    # EAR approximation using eye bounding rectangle
    if len(eye_landmarks) == 0:
        return 0
    xs = [p[0] for p in eye_landmarks]
    ys = [p[1] for p in eye_landmarks]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    return h / w if w > 0 else 0

def mouth_aspect_ratio(mouth_landmarks):
    # MAR approximation using mouth bounding rectangle
    if len(mouth_landmarks) == 0:
        return 0
    xs = [p[0] for p in mouth_landmarks]
    ys = [p[1] for p in mouth_landmarks]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    return h / w if w > 0 else 0

# Face landmarks for head-pose (PnP)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)

LMKS_IDX = [1, 199, 33, 263, 61, 291]

# ===== STATE VARIABLES =====
alert_head = False
alert_drowsy = False
alert_yawn = False

head_start_time = None
drowsy_start_time = None
yawn_start_time = None

last_head_beep = 0
last_drowsy_beep = 0
last_yawn_beep = 0

logged_head = False
logged_drowsy = False
logged_yawn = False

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

            # ===== Head-pose =====
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

                    # Draw angles
                    cv2.putText(frame, f"Yaw:{yaw_deg:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    cv2.putText(frame, f"Pitch:{pitch_deg:.2f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                    now = time.time()
                    is_distracted = (abs(yaw_deg) > DISTRACTION_YAW) or (pitch_deg > DISTRACTION_PITCH)
                    is_straight = (abs(yaw_deg) <= STRAIGHT_YAW) and (abs(pitch_deg) <= STRAIGHT_PITCH)

                    # ===== Head-pose alert =====
                    if is_distracted:
                        if head_start_time is None:
                            head_start_time = now
                            logged_head = False
                        elif (now - head_start_time) >= DISTRACTION_DURATION:
                            if not alert_head:
                                alert_head = True
                                beep()
                                if not logged_head:
                                    with open(LOG_FILE, "a", newline="") as f:
                                        writer = csv.writer(f)
                                        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                                                         round(yaw_deg,2), round(pitch_deg,2), "N/A", "N/A", "Distraction Started"])
                                    logged_head = True
                                last_head_beep = now
                            elif (now - last_head_beep) >= ALERT_REPEAT_INTERVAL:
                                beep()
                                last_head_beep = now
                    elif is_straight:
                        if alert_head and logged_head:
                            with open(LOG_FILE, "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                                                 round(yaw_deg,2), round(pitch_deg,2), "N/A", "N/A", "Distraction Ended"])
                        alert_head = False
                        head_start_time = None
                        logged_head = False

            except Exception:
                cv2.putText(frame, "Head-pose error", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # ===== Eyes & Mouth (Drowsiness & Yawning) =====
            # Simplified: Use bounding box of eyes and mouth
            h_lm = [(lm[i].x * w, lm[i].y * h) for i in range(len(lm))]
            # Example landmarks (left eye: 33-133, right eye: 362-263, mouth: 61-291)
            left_eye = h_lm[33:133]
            right_eye = h_lm[362:263] if 362 < 263 else h_lm[263:362]
            mouth = h_lm[61:291]

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
            mar = mouth_aspect_ratio(mouth)

            # ===== Drowsiness alert =====
            if ear < EAR_THRESHOLD:
                if drowsy_start_time is None:
                    drowsy_start_time = now
                    logged_drowsy = False
                elif now - drowsy_start_time >= EAR_DURATION:
                    if not alert_drowsy:
                        alert_drowsy = True
                        beep()
                        if not logged_drowsy:
                            with open(LOG_FILE, "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                                                 "N/A","N/A", round(ear,2), "N/A", "Drowsiness"])
                            logged_drowsy = True
                        last_drowsy_beep = now
                    elif now - last_drowsy_beep >= ALERT_REPEAT_INTERVAL:
                        beep()
                        last_drowsy_beep = now
            else:
                alert_drowsy = False
                drowsy_start_time = None
                logged_drowsy = False

            # ===== Yawning alert =====
            if mar > MAR_THRESHOLD:
                if yawn_start_time is None:
                    yawn_start_time = now
                    logged_yawn = False
                elif now - yawn_start_time >= MAR_DURATION:
                    if not alert_yawn:
                        alert_yawn = True
                        beep()
                        if not logged_yawn:
                            with open(LOG_FILE, "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                                                 "N/A","N/A","N/A", round(mar,2), "Yawning"])
                            logged_yawn = True
                        last_yawn_beep = now
                    elif now - last_yawn_beep >= ALERT_REPEAT_INTERVAL:
                        beep()
                        last_yawn_beep = now
            else:
                alert_yawn = False
                yawn_start_time = None
                logged_yawn = False

        else:
            cv2.putText(frame, "No face detected", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            alert_head = alert_drowsy = alert_yawn = False
            head_start_time = drowsy_start_time = yawn_start_time = None
            logged_head = logged_drowsy = logged_yawn = False

        # ===== Draw warnings =====
        if alert_head or alert_drowsy or alert_yawn:
            cv2.rectangle(frame, (80, 40), (w-80, 130), (0, 0, 255), -1)
            cv2.putText(frame, "⚠ ALERT DETECTED ⚠", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 3)

        cv2.imshow("Driver Vigilance System", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
