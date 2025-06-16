# save this as test_cam.py
import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    cv2.imwrite("test_frame.jpg", frame)
else:
    print("Failed to capture from camera")
cap.release()
