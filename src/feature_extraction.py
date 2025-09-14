import cv2
import mediapipe as mp
import numpy as np
import time

# Mediapipe FaceMesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# EAR calculation
def eye_aspect_ratio(landmarks, eye_indices):
    p1 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p2 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
    p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p4 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    p5 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    p6 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])

    A = np.linalg.norm(p2 - p4)
    B = np.linalg.norm(p3 - p5)
    C = np.linalg.norm(p1 - p6)

    ear = (A + B) / (2.0 * C)
    return ear

# MAR calculation
def mouth_aspect_ratio(landmarks, mouth_indices):
    top = np.array([landmarks[mouth_indices[13]].x, landmarks[mouth_indices[13]].y])
    bottom = np.array([landmarks[mouth_indices[14]].x, landmarks[mouth_indices[14]].y])
    left = np.array([landmarks[mouth_indices[78]].x, landmarks[mouth_indices[78]].y])
    right = np.array([landmarks[mouth_indices[308]].x, landmarks[mouth_indices[308]].y])

    A = np.linalg.norm(top - bottom)
    B = np.linalg.norm(left - right)

    mar = A / B
    return mar

# Webcam capture
cap = cv2.VideoCapture(0)

# Auto-stop settings
MAX_RUNTIME = 60       # stop after 60 seconds
MAX_FRAMES = 300       # stop after 300 frames
NO_FACE_LIMIT = 50     # stop if no face detected for 50 frames

start_time = time.time()
frame_count = 0
no_face_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        no_face_count = 0  # reset if face found
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            right_eye_indices = [33, 160, 158, 133, 153, 144]
            ear = eye_aspect_ratio(landmarks, right_eye_indices)

            mouth_indices = list(range(0, 468))
            mar = mouth_aspect_ratio(landmarks, mouth_indices)

            print(f"EAR: {ear:.3f}, MAR: {mar:.3f}")
    else:
        no_face_count += 1

    cv2.imshow("Driver Vigilance - Feature Extraction", frame)

    # Stop conditions
    elapsed_time = time.time() - start_time
    if elapsed_time > MAX_RUNTIME:
        print("⏳ Auto-stop: reached max runtime")
        break
    if frame_count > MAX_FRAMES:
        print("📸 Auto-stop: reached max frames")
        break
    if no_face_count > NO_FACE_LIMIT:
        print("🙈 Auto-stop: no face detected for too long")
        break

    # ESC key manual stop
    if cv2.waitKey(1) & 0xFF == 27:
        print("🛑 Manual stop (ESC key pressed)")
        break

cap.release()
cv2.destroyAllWindows()
