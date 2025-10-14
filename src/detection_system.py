import cv2
import mediapipe as mp
import numpy as np
import math
import time
import csv
import os
import winsound

# -----------------------
# CONFIG
# -----------------------
LOG_FILE = "detection_log.csv"

EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.7
DISTRACTION_YAW = 20.0
DISTRACTION_PITCH = 15.0
ALERT_DELAY = 5.0  # wait 5 seconds before alerting

BEEP_FREQ = 2000
BEEP_DUR_MS = 200

# -----------------------
# SETUP
# -----------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["Timestamp","Event","Yaw","Pitch","EAR","MAR"])

# -----------------------
# UTILITY FUNCTIONS
# -----------------------
def beep():
    try:
        winsound.Beep(BEEP_FREQ, BEEP_DUR_MS)
    except:
        pass

def log_event(event, yaw=0, pitch=0, ear=0, mar=0):
    with open(LOG_FILE,"a",newline="") as f:
        csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"), event, round(yaw,2), round(pitch,2), round(ear,2), round(mar,2)])

# -----------------------
# ASPECT RATIO CALCULATIONS
# -----------------------
def eye_aspect_ratio(lm, left_idx, right_idx):
    def ear_calc(eye):
        A = np.linalg.norm(eye[1]-eye[5])
        B = np.linalg.norm(eye[2]-eye[4])
        C = np.linalg.norm(eye[0]-eye[3])
        return (A+B)/(2.0*C)
    left = np.array([lm[i] for i in left_idx])
    right = np.array([lm[i] for i in right_idx])
    return (ear_calc(left)+ear_calc(right))/2.0

def mouth_aspect_ratio(lm, mouth_idx):
    mouth = np.array([lm[i] for i in mouth_idx])
    A = np.linalg.norm(mouth[13]-mouth[19])
    B = np.linalg.norm(mouth[14]-mouth[18])
    C = np.linalg.norm(mouth[15]-mouth[17])
    D = np.linalg.norm(mouth[0]-mouth[6])
    return (A+B+C)/(3.0*D)

# -----------------------
# HEAD POSE
# -----------------------
MODEL_POINTS = np.array([
    (0.0,0.0,0.0),
    (0.0,-330.0,-65.0),
    (-225.0,170.0,-135.0),
    (225.0,170.0,-135.0),
    (-150.0,-150.0,-125.0),
    (150.0,-150.0,-125.0)
],dtype=np.float64)

LMKS_IDX = [1,199,33,263,61,291]

def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
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

# -----------------------
# INDICES
# -----------------------
LEFT_EYE_IDX = [33,160,158,133,153,144]
RIGHT_EYE_IDX = [362,385,387,263,373,380]
MOUTH_IDX = [61,81,13,311,308,402,317,14,87,178,88,95,78,191,80,81,82,13,312,311]

# -----------------------
# VIDEO CAPTURE
# -----------------------
cap = cv2.VideoCapture(0)

# Alert timers
drowsy_start = None
yawn_start = None
distraction_start = None

drowsy_alerted = False
yawn_alerted = False
distraction_alerted = False

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

            # EAR & MAR
            ear = eye_aspect_ratio(lm, LEFT_EYE_IDX, RIGHT_EYE_IDX)
            mar = mouth_aspect_ratio(lm, MOUTH_IDX)

            # Head pose
            image_points = np.array([lm[i] for i in LMKS_IDX], dtype=np.float64)
            focal_length = w
            center = (w/2.0, h/2.0)
            camera_matrix = np.array([[focal_length,0,center[0]],[0,focal_length,center[1]],[0,0,1]],dtype=np.float64)
            dist_coeffs = np.zeros((4,1))
            success, rotation_vector, _ = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix, dist_coeffs)
            R,_ = cv2.Rodrigues(rotation_vector)
            roll,pitch,yaw = rotationMatrixToEulerAngles(R)
            now = time.time()

            # -----------------------
            # DROWSINESS
            # -----------------------
            if ear < EAR_THRESHOLD:
                if drowsy_start is None:
                    drowsy_start = now
                elif now - drowsy_start >= ALERT_DELAY and not drowsy_alerted:
                    beep()
                    log_event("Drowsiness", yaw, pitch, ear, mar)
                    drowsy_alerted = True
            else:
                drowsy_start = None
                drowsy_alerted = False

            # -----------------------
            # YAWNING
            # -----------------------
            if mar > MAR_THRESHOLD:
                if yawn_start is None:
                    yawn_start = now
                elif now - yawn_start >= ALERT_DELAY and not yawn_alerted:
                    beep()
                    log_event("Yawning", yaw, pitch, ear, mar)
                    yawn_alerted = True
            else:
                yawn_start = None
                yawn_alerted = False

            # -----------------------
            # DISTRACTION
            # -----------------------
            distracted = abs(yaw) > DISTRACTION_YAW or pitch > DISTRACTION_PITCH
            straight = abs(yaw) <= 12 and abs(pitch) <= 8
            if distracted:
                if distraction_start is None:
                    distraction_start = now
                elif now - distraction_start >= ALERT_DELAY and not distraction_alerted:
                    beep()
                    log_event("Distraction", yaw, pitch, ear, mar)
                    distraction_alerted = True
            elif straight:
                distraction_start = None
                distraction_alerted = False

            # -----------------------
            # DISPLAY
            # -----------------------
            cv2.putText(frame,f"EAR:{ear:.2f}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
            cv2.putText(frame,f"MAR:{mar:.2f}",(10,55),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
            cv2.putText(frame,f"Yaw:{yaw:.2f}",(10,85),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
            cv2.putText(frame,f"Pitch:{pitch:.2f}",(10,115),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

            if drowsy_alerted:
                cv2.putText(frame,"⚠ DROWSINESS DETECTED ⚠",(150,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            if yawn_alerted:
                cv2.putText(frame,"⚠ YAWNING DETECTED ⚠",(150,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            if distraction_alerted:
                cv2.putText(frame,"⚠ DISTRACTION DETECTED ⚠",(150,250),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

        else:
            cv2.putText(frame,"No face detected",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            drowsy_start = yawn_start = distraction_start = None
            drowsy_alerted = yawn_alerted = distraction_alerted = False

        cv2.imshow("Driver Vigilance Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
