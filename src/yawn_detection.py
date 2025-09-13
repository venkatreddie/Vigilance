import cv2
import os
import time

BASE_DIR = os.path.dirname(__file__)

# Load cascades
face_cascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml"))
mouth_cascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, "haarcascade_mcs_mouth.xml"))

cap = cv2.VideoCapture(0)

yawn_start_time = None
YAWN_THRESHOLD_RATIO = 0.6   # strict: must open wider
YAWN_HOLD_TIME = 0.8         # must keep open for 0.8s

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        mouths = mouth_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.7, minNeighbors=11, minSize=(30, 30)
        )

        for (mx, my, mw, mh) in mouths:
            # Ensure mouth is in lower half of face
            if my > h / 2:
                cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 255, 0), 2)

                mar = mh / float(mw)  # mouth aspect ratio

                if mar > YAWN_THRESHOLD_RATIO:
                    if yawn_start_time is None:
                        yawn_start_time = time.time()
                    elif time.time() - yawn_start_time >= YAWN_HOLD_TIME:
                        cv2.putText(frame, "Yawning... YAWN ALERT!", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                else:
                    yawn_start_time = None

    cv2.imshow("Driver Vigilance - Yawning Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
