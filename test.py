import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

while True:
    capture = cv2.VideoCapture(0)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    _, frame = capture.read()

    cv2.imshow("Image", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
