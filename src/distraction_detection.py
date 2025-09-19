import cv2
import mediapipe as mp
import time
import csv
import os
import winsound
import numpy as np
import math
from collections import deque

# ====== CONFIG ======
LOG_FILE = "distraction_log.csv"

# Pose thresholds (enter distraction / exit distraction = hysteresis)
DISTRACTION_YAW = 25.0      # degrees -> enter distraction if abs(yaw) > this
DISTRACTION_PITCH = 15.0    # degrees -> enter distraction if pitch > this (looking down)

STRAIGHT_YAW = 15.0         # degrees -> considered straight if abs(yaw) <= this
STRAIGHT_PITCH = 8.0        # degrees -> considered straight if pitch <= this

# Timing
DISTRACTION_HOLD = 10.0     # seconds of continuous distracted pose before alert triggers
STRAIGHT_COOLDOWN = 2.0     # seconds of comfortable straight pose required to clear alert
ALERT_REPEAT_INTERVAL = 3.0 # seconds between repeated beeps while distracted

# Beep settings (Windows)
BEEP_FREQ = 2000            # Hz
BEEP_DUR_MS = 200          # ms (short beep so frames continue updating)

# Smoothing
SMOOTH_WINDOW = 6          # number of recent angle samples to average

# ====== SETUP ======
# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Prepare log file (single persistent CSV)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Yaw_deg", "Pitch_deg", "Event"])

# Utility: compute Euler angles (roll, pitch, yaw) from rotation matrix
def rotationMatrixToEulerAngles(R):
    # R is 3x3 rotation matrix
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
    # convert to degrees
    return np.degrees([x, y, z])  # roll, pitch, yaw

# Beep (non-blocking feel via short beep repeated)
def beep():
    try:
        winsound.Beep(BEEP_FREQ, BEEP_DUR_MS)
    except Exception:
        pass

# Head-pose estimation using selected MediaPipe FaceMesh indexes
# Model points (3D) approximations (same across examples)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),        # nose tip
    (0.0, -330.0, -65.0),   # chin
    (-225.0, 170.0, -135.0),# left eye left corner
    (225.0, 170.0, -135.0), # right eye right corner
    (-150.0, -150.0, -125.0),# left mouth corner
    (150.0, -150.0, -125.0)  # right mouth corner
], dtype=np.float64)

# Indices in MediaPipe FaceMesh to use for image_points:
# [nose_tip, chin, left_eye_outer, right_eye_outer, mouth_left, mouth_right]
# MediaPipe FaceMesh common indices: 1 (nose tip), 199 (chin), 33 (left eye outer),
# 263 (right eye outer), 61 (mouth left), 291 (mouth right)
LMKS_IDX = [1, 199, 33, 263, 61, 291]

# smoothing buffers
yaw_buf = deque(maxlen=SMOOTH_WINDOW)
pitch_buf = deque(maxlen=SMOOTH_WINDOW)

# state vars
distraction_start = None
straight_start = None
alert_active = False
logged_this_event = False
last_beep_time = 0.0

# Video capture
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

        # default display values
        display_yaw = 0.0
        display_pitch = 0.0

        face_detected = False

        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            face_detected = True
            fml = results.multi_face_landmarks[0]
            lm = fml.landmark

            # build 2D image points from selected landmarks
            try:
                image_points = np.array([
                    (lm[LMKS_IDX[0]].x * w, lm[LMKS_IDX[0]].y * h),  # nose tip
                    (lm[LMKS_IDX[1]].x * w, lm[LMKS_IDX[1]].y * h),  # chin
                    (lm[LMKS_IDX[2]].x * w, lm[LMKS_IDX[2]].y * h),  # left eye
                    (lm[LMKS_IDX[3]].x * w, lm[LMKS_IDX[3]].y * h),  # right eye
                    (lm[LMKS_IDX[4]].x * w, lm[LMKS_IDX[4]].y * h),  # mouth left
                    (lm[LMKS_IDX[5]].x * w, lm[LMKS_IDX[5]].y * h)   # mouth right
                ], dtype=np.float64)

                # camera internals
                focal_length = w
                center = (w/2.0, h/2.0)
                camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype=np.float64)
                dist_coeffs = np.zeros((4,1))  # assume no lens distortion

                success, rotation_vector, translation_vector = cv2.solvePnP(
                    MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )

                if success:
                    R_mat, _ = cv2.Rodrigues(rotation_vector)
                    roll_deg, pitch_deg, yaw_deg = rotationMatrixToEulerAngles(R_mat)
                    # mapping: roll (x), pitch (y), yaw (z)
                    display_yaw = yaw_deg
                    display_pitch = pitch_deg

                    # smoothing (reduces jitter)
                    yaw_buf.append(display_yaw)
                    pitch_buf.append(display_pitch)
                    smooth_yaw = float(np.mean(yaw_buf))
                    smooth_pitch = float(np.mean(pitch_buf))

                    # draw smoothed numeric debug on frame
                    cv2.putText(frame, f"yaw:{smooth_yaw:6.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    cv2.putText(frame, f"pitch:{smooth_pitch:6.2f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                    # --- distraction decision with hysteresis ---
                    now = time.time()
                    # check if current pose qualifies as distracted (enter thresholds)
                    is_distracted_now = (abs(smooth_yaw) > DISTRACTION_YAW) or (smooth_pitch > DISTRACTION_PITCH)

                    # check if current pose qualifies as comfortably straight (exit thresholds)
                    is_straight_now = (abs(smooth_yaw) <= STRAIGHT_YAW) and (abs(smooth_pitch) <= STRAIGHT_PITCH)

                    if is_distracted_now:
                        straight_start = None  # reset straight counter
                        if distraction_start is None:
                            distraction_start = now
                            logged_this_event = False
                        else:
                            # enough continuous duration?
                            if (now - distraction_start) >= DISTRACTION_HOLD:
                                # Activate alert
                                if not alert_active:
                                    alert_active = True
                                    # immediate beep + log (once)
                                    beep_time = now
                                    try:
                                        winsound.Beep(BEEP_FREQ, BEEP_DUR_MS)
                                    except Exception:
                                        pass
                                    # log once for the event
                                    if not logged_this_event:
                                        with open(LOG_FILE, "a", newline="") as f:
                                            writer = csv.writer(f)
                                            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                                                             round(smooth_yaw,2), round(smooth_pitch,2), "Distraction Detected"])
                                        logged_this_event = True
                                    last_beep = now
                                # while alert_active, repeat beeps every ALERT_REPEAT_INTERVAL
                                if alert_active:
                                    # use a persistent last_beep_time variable (nonlocal)
                                    try:
                                        if (now - last_beep_time) >= ALERT_REPEAT_INTERVAL:
                                            beep()
                                            last_beep_time = now
                                    except NameError:
                                        last_beep_time = now
                    else:
                        # not distracted now -> start straight timer or reset
                        distraction_start = None
                        # begin straight cooldown only if currently alert_active
                        if alert_active:
                            if straight_start is None:
                                straight_start = now
                            else:
                                if (now - straight_start) >= STRAIGHT_COOLDOWN:
                                    # fully cleared
                                    alert_active = False
                                    logged_this_event = False
                                    straight_start = None
                                    # reset buffers to avoid stale values
                                    yaw_buf.clear()
                                    pitch_buf.clear()
                                    distraction_start = None
                else:
                    # solvePnP failed: don't change state, but avoid spurious alerts
                    cv2.putText(frame, "pose solvePnP fail", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            except Exception as ex:
                # landmark indexing or solvePnP error: safe fallback
                cv2.putText(frame, f"pose error", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        # end for face landmarks
        # draw signboard if currently alert_active
        if alert_active:
            cv2.rectangle(frame, (80, 40), (w-80, 130), (0, 0, 255), -1)
            cv2.putText(frame, "⚠ DISTRACTION DETECTED ⚠", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 3)
    else:
        # no face detected -> treat as not distracted, reset timers
        distraction_start = None
        straight_start = None
        alert_active = False
        logged_this_event = False
        yaw_buf.clear()
        pitch_buf.clear()
        cv2.putText(frame, "No face detected", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # show frame
    cv2.imshow("Distraction Detection (Day11)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

finally:
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
