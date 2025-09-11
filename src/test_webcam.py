import cv2

cap = cv2.VideoCapture(0)  # Open webcam (0 = default)

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to access webcam")
        break

    cv2.imshow("Webcam Test", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
