import cv2
import mediapipe as mp
import time
import pygame
import csv
from datetime import datetime

# Initialize pygame mixer for sound alerts
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alerts/beep.wav")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

# Logging setup
log_file = "distraction_log.csv"
with open(log_file, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Yaw", "Pitch", "Event"])

# State variables
distraction_active = False
distraction_start = None
cooldown_active = False

def log_event(event, yaw, pitch):
    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), round(yaw, 2), round(pitch, 2), event])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    h, w, _ = frame.shape

    yaw, pitch = 0, 0  # Default

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get landmarks for nose (useful for rough head pose)
            nose_tip = face_landmarks.landmark[1]  # Nose tip index
            left_eye = face_landmarks.landmark[33]  # Left eye corner
            right_eye = face_landmarks.landmark[263]  # Right eye corner

            # Convert to pixel coordinates
            nx, ny = int(nose_tip.x * w), int(nose_tip.y * h)
            lx, ly = int(left_eye.x * w), int(left_eye.y * h)
            rx, ry = int(right_eye.x * w), int(right_eye.y * h)

            # Rough yaw (left/right)
            yaw = (lx - rx)

            # Rough pitch (up/down)
            pitch = (ny - ((ly + ry) // 2))

            # Scale values to look natural
            yaw = yaw * 0.3
            pitch = pitch * 0.3

            # Show debug values
            cv2.putText(frame, f"Yaw: {round(yaw,1)}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(frame, f"Pitch: {round(pitch,1)}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            # --- Distraction detection ---
            if abs(yaw) > 25 or pitch > 15:  # Sideways OR looking down
                if not distraction_active:
                    distraction_start = time.time()
                    distraction_active = True
                    cooldown_active = False
                else:
                    elapsed = time.time() - distraction_start
                    if elapsed > 10:  # Trigger only after 10s
                        if not cooldown_active:
                            log_event("Distraction Detected", yaw, pitch)
                            alert_sound.play()
                            cooldown_active = True

                        # Show alert on screen
                        cv2.rectangle(frame, (50, 80), (w-50, 160), (0,0,255), -1)
                        cv2.putText(frame, "DISTRACTION DETECTED!", (60, 140),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
            else:
                # Reset when looking straight
                distraction_active = False
                distraction_start = None
                cooldown_active = False

    cv2.imshow("Driver Distraction Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
