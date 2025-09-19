import cv2
import mediapipe as mp
import time
import csv
import os
import winsound
import numpy as np

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# CSV logging setup
log_file = "distraction_log.csv"
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Yaw", "Pitch", "Alert"])

# Buzzer function
def play_alert():
    winsound.Beep(2000, 500)  # frequency=2000Hz, duration=500ms

# Head pose estimation
def get_head_pose(landmarks, frame_w, frame_h):
    nose = (landmarks[1].x * frame_w, landmarks[1].y * frame_h)
    left_eye = (landmarks[33].x * frame_w, landmarks[33].y * frame_h)
    right_eye = (landmarks[263].x * frame_w, landmarks[263].y * frame_h)
    mouth_left = (landmarks[61].x * frame_w, landmarks[61].y * frame_h)
    mouth_right = (landmarks[291].x * frame_w, landmarks[291].y * frame_h)
    chin = (landmarks[199].x * frame_w, landmarks[199].y * frame_h)

    image_points = np.array([nose, chin, left_eye, right_eye, mouth_left, mouth_right], dtype="double")
    model_points = np.array([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
    ])

    focal_length = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_mat, translation_vector))
    _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(pose_mat)
    yaw, pitch, roll = [angle[0] for angle in eulerAngles]
    return yaw, pitch

# Variables
distraction_start_time = None
distraction_active = False
distraction_logged = False
last_alert_time = 0

# Settings
distraction_threshold = 10   # seconds before triggering alert
alert_repeat_interval = 5    # repeat alert every 5 sec while distracted
straight_cooldown = 2        # must be straight for 2 sec to reset

straight_start_time = None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            yaw, pitch = get_head_pose(face_landmarks.landmark, w, h)

            # Show yaw/pitch
            cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Distraction condition
            if abs(yaw) > 20 or pitch > 10:
                straight_start_time = None  # reset straight timer
                if distraction_start_time is None:
                    distraction_start_time = time.time()
                    distraction_logged = False

                if time.time() - distraction_start_time >= distraction_threshold:
                    distraction_active = True

                    # Show warning sign
                    cv2.rectangle(frame, (80, 40), (560, 120), (0, 0, 255), -1)
                    cv2.putText(frame, "DISTRACTION DETECTED!", (100, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

                    # Repeat alert every interval
                    if time.time() - last_alert_time >= alert_repeat_interval:
                        play_alert()
                        last_alert_time = time.time()

                        # Log only once per distraction event
                        if not distraction_logged:
                            with open(log_file, "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), yaw, pitch, "Distraction Detected"])
                            distraction_logged = True
            else:
                # looking straight
                if distraction_active:
                    if straight_start_time is None:
                        straight_start_time = time.time()
                    elif time.time() - straight_start_time >= straight_cooldown:
                        # Reset fully after cooldown
                        distraction_start_time = None
                        distraction_active = False
                        distraction_logged = False

    # Show video feed
    cv2.imshow("Driver Vigilance - Distraction Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
