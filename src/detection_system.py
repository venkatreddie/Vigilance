import cv2
import mediapipe as mp
import numpy as np
import math
import time
import csv
import os
import winsound

# =========================
# CONFIG
# =========================
LOG_FILE = "distraction_log.csv"

EAR_THRESHOLD = 0.25        # Eyes closed threshold
EAR_CONSEC_FRAMES = 15      # Frames eyes must be below threshold to trigger drowsiness

MAR_THRESHOLD = 0.7         # Mouth open threshold for yawning

DISTRACTION_YAW = 20.0      # Head yaw threshold
DISTRACTION_PITCH = 15.0    # Head pitch threshold
STRAIGHT_YAW = 12.0
STRAIGHT_PITCH = 8.0
DISTRACTION_DURATION = 10.0 # Seconds before distraction alert

BEEP_FREQ = 2000
BEEP_DUR_MS = 200

# =========================
# SETUP
# =========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# CSV log setup
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp","Yaw_deg","Pitch_deg","EAR","MAR","Event"])

# Beep function
def beep():
    try:
        winsound.Beep(BEEP_FREQ, BEEP_DUR_MS)
    except:
        pass

# Logging function
def log_event(event, yaw=0, pitch=0, ear=0, mar=0):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE,"a",newline="") as f:
        csv.writer(f).writerow([timestamp, round(yaw,2), round(pitch,2), round(ear,2), round(mar,2), event])

# Compute EAR
def eye_aspect_ratio(landmarks, left_idx, right_idx):
    left = np.array([landmarks[i] for i in left_idx])
    right = np.array([landmarks[i] for i in right_idx])
    A = np.linalg.norm(left[1]-left[5])
    B = np.linalg.norm(left[2]-left[4])
    C = np.linalg.norm(left[0]-left[3])
    ear_left = (A+B)/(2.0*C)
    A = np.linalg.norm(right[1]-right[5])
    B = np.linalg.norm(right[2]-right[4])
    C = np.linalg.norm(right[0]-right[3])
    ear_right = (A+B)/(2.0*C)
    return (ear_left + ear_right)/2.0

# Compute MAR
def mouth_aspect_ratio(landmarks, mouth_idx):
    mouth = np.array([landmarks[i] for i in mouth_idx])
    A = np.linalg.norm(mouth[13]-mouth[19])
    B = np.linalg.norm(mouth[14]-mouth[18])
    C = np.linalg.norm(mouth[15]-mouth[17])
    D = np.linalg.norm(mouth[0]-mouth[6])
    mar = (A+B+C)/(3.0*D)
    return mar

# Head Pose Utilities
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -330.0, -65.0),      # Chin
    (-225.0, 170.0, -135.0),   # Left eye corner
    (225.0, 170.0, -135.0),    # Right eye corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0)    # Right mouth corner
], dtype=np.float64)

LMKS_IDX = [1, 199, 33, 263, 61, 291]

def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.degrees([x,y,z])

# MediaPipe indices
LEFT_EYE_IDX = [33,160,158,133,153,144]
RIGHT_EYE_IDX = [362,385,387,263,373,380]
MOUTH_IDX = [61,81,13,311,308,402,317,14,87,178,88,95,78,191,80,81,82,13,312,311] 

# =========================
# VIDEO CAPTURE
# =========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

ear_counter = 0
distraction_start_time = None
alert_active = False
drowsy_alerted = False
yawn_alerted = False
distraction_logged = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h,w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm_raw = results.multi_face_landmarks[0].landmark
            lm = [(int(p.x*w), int(p.y*h)) for p in lm_raw]

            # Compute EAR & MAR
            ear = eye_aspect_ratio(lm, LEFT_EYE_IDX, RIGHT_EYE_IDX)
            mar = mouth_aspect_ratio(lm, MOUTH_IDX)

            # Head pose
            image_points = np.array([lm[i] for i in LMKS_IDX], dtype=np.float64)
            focal_length = w
            center = (w/2.0,h/2.0)
            camera_matrix = np.array([[focal_length,0,center[0]],[0,focal_length,center[1]],[0,0,1]],dtype=np.float64)
            dist_coeffs = np.zeros((4,1))
            success, rotation_vector, _ = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix, dist_coeffs)
            R,_ = cv2.Rodrigues(rotation_vector)
            roll,pitch,yaw = rotationMatrixToEulerAngles(R)

            now = time.time()

            # -----------------------
            # DROWSINESS
            # -----------------------
            if ear < EAR_THRESHOLD and mar < MAR_THRESHOLD: # avoid conflict with yawning
                ear_counter += 1
                if ear_counter >= EAR_CONSEC_FRAMES and not drowsy_alerted:
                    beep()
                    log_event("Drowsiness", yaw, pitch, ear, mar)
                    drowsy_alerted = True
            else:
                ear_counter = 0
                drowsy_alerted = False

            # -----------------------
            # YAWNING
            # -----------------------
            if mar > MAR_THRESHOLD and not yawn_alerted:
                beep()
                log_event("Yawning", yaw, pitch, ear, mar)
                yawn_alerted = True
            elif mar <= MAR_THRESHOLD:
                yawn_alerted = False

            # -----------------------
            # DISTRACTION
            # -----------------------
            is_distracted = abs(yaw) > DISTRACTION_YAW or pitch > DISTRACTION_PITCH
            is_straight = abs(yaw) <= STRAIGHT_YAW and abs(pitch) <= STRAIGHT_PITCH

            if is_distracted:
                if distraction_start_time is None:
                    distraction_start_time = now
                    distraction_logged = False
                elif now - distraction_start_time >= DISTRACTION_DURATION and not alert_active:
                    beep()
                    if not distraction_logged:
                        log_event("Distraction Started", yaw, pitch, ear, mar)
                        distraction_logged = True
                    alert_active = True
            elif is_straight:
                if alert_active and distraction_logged:
                    log_event("Distraction Ended", yaw, pitch, ear, mar)
                alert_active = False
                distraction_start_time = None
                distraction_logged = False

            # -----------------------
            # Display
            # -----------------------
            cv2.putText(frame,f"EAR:{ear:.2f}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
            cv2.putText(frame,f"MAR:{mar:.2f}",(10,55),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
            cv2.putText(frame,f"Yaw:{yaw:.2f}",(10,85),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
            cv2.putText(frame,f"Pitch:{pitch:.2f}",(10,115),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

            # Show alert messages on screen
            if drowsy_alerted:
                cv2.putText(frame,"⚠ DROWSINESS DETECTED ⚠",(150,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            if yawn_alerted:
                cv2.putText(frame,"⚠ YAWNING DETECTED ⚠",(150,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            if alert_active:
                cv2.putText(frame,"⚠ DISTRACTION DETECTED ⚠",(150,250),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

        else:
            cv2.putText(frame,"No face detected",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            ear_counter = 0
            drowsy_alerted = False
            yawn_alerted = False
            alert_active = False
            distraction_start_time = None
            distraction_logged = False

        cv2.imshow("Driver Vigilance Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key==27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

