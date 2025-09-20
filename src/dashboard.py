import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
import winsound
import matplotlib.pyplot as plt

# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# CSV logging
log_file = "distraction_log.csv"
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Event", "Yaw", "Pitch", "Duration"])

# Alert function
def play_buzzer():
    winsound.Beep(1000, 800)

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
    dist_coeffs = np.zeros((4,1)) 

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_mat, translation_vector))
    _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(pose_mat)
    yaw, pitch, roll = [angle[0] for angle in eulerAngles]
    return yaw, pitch

# Function to draw matplotlib graph and return as OpenCV image
def draw_graph(values, width=400, height=200):
    plt.figure(figsize=(4,2))
    plt.plot(values, color="red", linewidth=2)
    plt.ylim(0, 100)
    plt.title("Distraction Probability", fontsize=10)
    plt.xlabel("Frames")
    plt.ylabel("%")
    plt.tight_layout()

    # Convert plot to image
    plt.savefig("graph.png")
    plt.close()
    graph_img = cv2.imread("graph.png")
    graph_img = cv2.resize(graph_img, (width, height))
    return graph_img

# State variables
distraction_start_time = None
distraction_active = False
total_distractions = 0
total_distraction_time = 0
longest_distraction = 0
probability = 0

prob_values = []  # For graph

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    h, w, _ = frame.shape
    current_time = time.time()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            yaw, pitch = get_head_pose(face_landmarks.landmark, w, h)

            distracted = abs(yaw) > 20 or pitch < -10

            if distracted:
                if distraction_start_time is None:
                    distraction_start_time = current_time
                elapsed = current_time - distraction_start_time

                probability = min(100, int((elapsed / 15) * 100))
                prob_values.append(probability)
                if len(prob_values) > 50:
                    prob_values.pop(0)

                if not distraction_active and elapsed >= 10:
                    distraction_active = True
                    total_distractions += 1
                    play_buzzer()

                    with open(log_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), "Distraction Start", yaw, pitch, ""])

                if distraction_active:
                    cv2.rectangle(frame, (80, 40), (560, 120), (0, 0, 255), -1)
                    cv2.putText(frame, "DISTRACTION DETECTED!", (100, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

            else:
                if distraction_active:
                    distraction_active = False
                    distraction_duration = current_time - distraction_start_time
                    total_distraction_time += distraction_duration
                    longest_distraction = max(longest_distraction, distraction_duration)

                    with open(log_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), "Distraction End", yaw, pitch, f"{distraction_duration:.1f}s"])

                distraction_start_time = None
                probability = 0
                prob_values.append(probability)
                if len(prob_values) > 50:
                    prob_values.pop(0)

            # Show yaw/pitch
            cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    # Probability bar
    cv2.rectangle(frame, (10, h-30), (210, h-10), (200, 200, 200), 2)
    cv2.rectangle(frame, (12, h-28), (12 + 2*probability, h-12), (0,0,255), -1)
    cv2.putText(frame, f"{probability}%", (220, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # Stats panel
    cv2.putText(frame, f"Total Distractions: {total_distractions}", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(frame, f"Total Time: {int(total_distraction_time)}s", (400, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(frame, f"Longest: {int(longest_distraction)}s", (400, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Add graph overlay
    graph_img = draw_graph(prob_values)
    gh, gw, _ = graph_img.shape
    frame[10:10+gh, w-gw-10:w-10] = graph_img

    # Show window
    cv2.imshow("Driver Vigilance - Dashboard", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
