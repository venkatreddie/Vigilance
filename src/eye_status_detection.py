import cv2
import time

# Load Haar cascades
face_cascade = cv2.CascadeClassifier('src/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('src/haarcascade_eye.xml')

# Start webcam
cap = cv2.VideoCapture(0)

closed_eyes_frame_count = 0  # Counter for closed eyes
drowsy_threshold = 15        # Number of frames before alert (adjust as needed)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) == 0:
            closed_eyes_frame_count += 1
            cv2.putText(frame, "Eyes Closed!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2)
        else:
            closed_eyes_frame_count = 0
            cv2.putText(frame, "Eyes Open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2)

        # If eyes are closed for too long → alert
        if closed_eyes_frame_count >= drowsy_threshold:
            cv2.putText(frame, "DROWSINESS ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.2, (0, 0, 255), 3)

    cv2.imshow("Driver Vigilance - Eye Status", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
