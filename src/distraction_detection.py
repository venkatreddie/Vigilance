import cv2
import dlib
import time
import csv
import os
from imutils import face_utils
import numpy as np
import simpleaudio as sa

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# CSV logging setup
log_file = "distraction_log.csv"
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Yaw", "Pitch", "Alert"])

# Buzzer sound setup
def play_buzzer():
    try:
        wave_obj = sa.WaveObject.from_wave_file("buzzer.wav")
        play_obj = wave_obj.play()
    except:
        print("Buzzer file not found!")

# Helper: head pose estimation
def get_head_pose(shape):
    image_points = np.array([
        shape[30],  # Nose tip
        shape[8],   # Chin
        shape[36],  # Left eye left corner
        shape[45],  # Right eye right corner
        shape[48],  # Left Mouth corner
        shape[54]   # Right mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),             
        (0.0, -330.0, -65.0),       
        (-225.0, 170.0, -135.0),     
        (225.0, 170.0, -135.0),     
        (-150.0, -150.0, -125.0),    
        (150.0, -150.0, -125.0)     
    ])

    size = (640, 480)
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4,1)) 
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_mat, translation_vector))
    _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(pose_mat)
    yaw, pitch, roll = [angle[0] for angle in eulerAngles]
    return yaw, pitch

# Variables for distraction detection
distraction_start_time = None
alert_triggered = False
distraction_threshold = 10  # seconds

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        yaw, pitch = get_head_pose(shape)

        cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        # Detect distraction (yaw too far OR pitch down)
        if abs(yaw) > 20 or pitch < -15:
            if distraction_start_time is None:
                distraction_start_time = time.time()
            elif time.time() - distraction_start_time >= distraction_threshold:
                alert_triggered = True
        else:
            distraction_start_time = None
            alert_triggered = False

        # Show alert if triggered
        if alert_triggered:
            cv2.rectangle(frame, (80, 40), (560, 120), (0, 0, 255), -1)
            cv2.putText(frame, "DISTRACTION DETECTED!", (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

            # Play buzzer
            play_buzzer()

            # Log to CSV
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), yaw, pitch, "Distraction Detected"])

    cv2.imshow("Driver Vigilance - Distraction Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
