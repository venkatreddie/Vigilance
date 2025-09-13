import cv2
import dlib
import time
import numpy as np
from scipy.spatial import distance

# Load Haar cascades
face_cascade = cv2.CascadeClassifier("src/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("src/haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier("src/haarcascade_mcs_mouth.xml")

# Dlib predictor for landmarks (make sure .dat file is present)
predictor = dlib.shape_predictor("src/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

# EAR (Eye Aspect Ratio) for drowsiness
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 15
eye_counter = 0

# Yawn detection params
MOUTH_AR_THRESH = 0.6
MOUTH_AR_CONSEC_FRAMES = 15
mouth_counter = 0

# Head pose parameters
HEAD_TILT_THRESH = 20   # degrees
HEAD_TILT_FRAMES = 15
head_counter = 0

# 3D model points for head pose estimation
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        # Eye landmarks
        leftEye = shape[36:42]
        rightEye = shape[42:48]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        if ear < EYE_AR_THRESH:
            eye_counter += 1
            if eye_counter >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (50, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            eye_counter = 0

        # Mouth landmarks
        mouth = shape[48:68]
        mouth_width = distance.euclidean(mouth[0], mouth[6])
        mouth_height = distance.euclidean(mouth[3], mouth[9])
        mar = mouth_height / mouth_width

        if mar > MOUTH_AR_THRESH:
            mouth_counter += 1
            if mouth_counter >= MOUTH_AR_CONSEC_FRAMES:
                cv2.putText(frame, "YAWN ALERT!", (50, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        else:
            mouth_counter = 0

        # Head Pose Estimation
        image_points = np.array([
            (shape[30][0], shape[30][1]),     # Nose tip
            (shape[8][0], shape[8][1]),       # Chin
            (shape[36][0], shape[36][1]),     # Left eye left corner
            (shape[45][0], shape[45][1]),     # Right eye right corner
            (shape[48][0], shape[48][1]),     # Left Mouth corner
            (shape[54][0], shape[54][1])      # Right mouth corner
        ], dtype=np.float64)

        size = frame.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )

        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch, yaw, roll = [np.degrees(a) for a in angles]

        if abs(pitch) > HEAD_TILT_THRESH or abs(roll) > HEAD_TILT_THRESH:
            head_counter += 1
            if head_counter >= HEAD_TILT_FRAMES:
                cv2.putText(frame, "HEAD TILT ALERT!", (50, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        else:
            head_counter = 0

    cv2.imshow("Driver Vigilance Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
