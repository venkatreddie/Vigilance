import cv2
import time

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
mouth_cascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")

cap = cv2.VideoCapture(0)

yawn_start_time = None

while True:
    success, frame = cap.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        mouths = mouth_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.3,
            minNeighbors=7,
            minSize=(30, 30)
        )

        for (mx, my, mw, mh) in mouths:
            # ✅ Only consider mouths in LOWER HALF of the face
            if my > h // 2:
                cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (255, 0, 0), 2)

                mouth_ratio = mh / float(mw)

                # ✅ Must be wide enough AND tall enough
                if mouth_ratio > 0.55 and mh > 25:
                    if yawn_start_time is None:
                        yawn_start_time = time.time()
                    else:
                        duration = time.time() - yawn_start_time
                        if duration > 0.8:  # 0.8 sec sustained
                            cv2.putText(frame, "Yawning Detected!", (50, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    yawn_start_time = None
                    cv2.putText(frame, "Normal", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Yawn Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
