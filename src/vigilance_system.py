import cv2
import dlib
import pygame
import time
from scipy.spatial import distance as dist

# ===============================
# Initialize pygame for alerts
# ===============================
pygame.mixer.init()

def play_alert(sound_file="alert.wav"):
    try:
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"[ERROR] Could not play alert: {e}")

# ===============================
# EAR & MAR Functions
# ===============================
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])   # 53, 57
    C = dist.euclidean(mouth[0], mouth[6])   # 49, 55
    return (A + B) / (2.0 * C)

# ===============================
# Thresholds & Constants
# ===============================
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20
MOUTH_AR_THRESH = 0.8
MOUTH_AR_CONSEC_FRAMES = 15

# ===============================
# Load Dlib Face Detector & Predictor
# ===============================
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = (42, 48)  # Left eye
(rStart, rEnd) = (36, 42)  # Right eye
(mStart, mEnd) = (48, 68)  # Mouth

# ===============================
# Start Video Stream
# ===============================
print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)

COUNTER_EYE = 0
COUNTER_MOUTH = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        coords = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        leftEye = coords[lStart:lEnd]
        rightEye = coords[rStart:rEnd]
        mouth = coords[mStart:mEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        mar = mouth_aspect_ratio(mouth)

        # Draw contours
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 0, 255), 1)

        # ===============================
        # Eye Closure Detection
        # ===============================
        if ear < EYE_AR_THRESH:
            COUNTER_EYE += 1
            if COUNTER_EYE >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                play_alert("alert.wav")
        else:
            COUNTER_EYE = 0

        # ===============================
        # Yawning Detection
        # ===============================
        if mar > MOUTH_AR_THRESH:
            COUNTER_MOUTH += 1
            if COUNTER_MOUTH >= MOUTH_AR_CONSEC_FRAMES:
                cv2.putText(frame, "YAWNING ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                play_alert("alert.wav")
        else:
            COUNTER_MOUTH = 0

    cv2.imshow("Driver Vigilance Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
