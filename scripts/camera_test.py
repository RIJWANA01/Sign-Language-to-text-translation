import cv2

cap = cv2.VideoCapture(0)

print("Camera opened:", cap.isOpened())

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Frame not received")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()