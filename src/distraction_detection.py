import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
import winsound

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# CSV logging setup
log_file = "distraction_log.csv"
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Yaw", "Pitch", "Event"])

# Buzzer function
def play_buzzer():
    winsound.Beep(1000, 700)

# Head pose estimation
def get_head_pose(landmarks, frame_w, frame_h):
    nose = (landmarks[1].x * frame_w, landmarks[1].y * frame_h)
    chin = (landmarks[199].x * frame_w, landmarks[199].y * frame_h)
    left_eye = (landmarks[33].x * frame_w, landmarks[33].y * frame_h)
    right_eye = (landmarks[263].x * frame_w, landmarks[263].y * frame_h)
    mouth_left = (landmarks[61].x * frame_w, landmarks[61].y * frame_h)
    mouth_right = (landmarks[291].x * frame_w, landmarks[291].y * frame_h)

    image_points = np.array([nose, chin, left_eye, right_eye, mouth_left, mouth_right], dtype="double")
    model_points = np.array([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
    ])

    focal_length = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    rot_mat, _ = cv2.Rodrigues(rotation_vector)
    proj_mat = np.hstack((rot_mat, translation_vector))
    _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(proj_mat)

    yaw, pitch, roll = [float(angle) for angle in eulerAngles]
    return yaw, pitch

# Variables
distraction_start_time = None
distraction_active = False
distraction_logged = False
distraction_threshold = 10  # seconds

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            yaw, pitch = get_head_pose(face_landmarks.landmark, w, h)

            # Show values
            cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Check distraction (side or down)
            if abs(yaw) > 20 or pitch > 10:
                if distraction_start_time is None:
                    distraction_start_time = time.time()
                elif not distraction_active and (time.time() - distraction_start_time >= distraction_threshold):
                    distraction_active = True
                    distraction_logged = False  # reset logging for new event

                # If distraction active, trigger alert continuously
                if distraction_active:
                    cv2.rectangle(frame, (80, 40), (560, 120), (0, 0, 255), -1)
                    cv2.putText(frame, "DISTRACTION DETECTED!", (100, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                    play_buzzer()

                    if not distraction_logged:
                        with open(log_file, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                                             yaw, pitch, "Distraction"])
                        distraction_logged = True

            else:
                # Reset when looking straight
                distraction_start_time = None
                distraction_active = False

    cv2.imshow("Driver Distraction Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
