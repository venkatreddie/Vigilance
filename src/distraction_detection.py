import cv2
import time
import winsound
import numpy as np

# Haarcascade for face
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Thresholds
DISTRACTION_TIME = 10  # seconds to trigger alert
ALERT_FREQ = 2500  # sound frequency
ALERT_DUR = 1000   # sound duration (ms)

# State variables
distraction_start = None
alert_active = False

def play_alert():
    winsound.Beep(ALERT_FREQ, ALERT_DUR)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    distraction_detected = False

    for (x, y, w, h) in faces:
        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Define ROI
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Approximate head pose by comparing width/height ratio
        aspect_ratio = w / float(h)

        # Heuristic rules for pose
        if aspect_ratio < 0.7:  # Looking sideways
            distraction_detected = True
            cv2.putText(frame, "Looking Side!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        if h > w * 1.3:  # Looking downward (face elongated)
            distraction_detected = True
            cv2.putText(frame, "Looking Down!", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # If no face detected, also consider distracted
    if len(faces) == 0:
        distraction_detected = True
        cv2.putText(frame, "Face Not Visible!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

    # Logic for distraction timer
    if distraction_detected:
        if distraction_start is None:
            distraction_start = time.time()
        else:
            elapsed = time.time() - distraction_start
            if elapsed >= DISTRACTION_TIME:
                alert_active = True
    else:
        distraction_start = None
        alert_active = False

    # Trigger alerts if active
    if alert_active:
        cv2.putText(frame, "WARNING: STAY FOCUSED!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
        play_alert()

    cv2.imshow("Distraction Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
