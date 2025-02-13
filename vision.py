import cv2
import numpy as np

FEED_WAIT_DELAY_MS = 1

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    retval, image = capture.read()

    lower, upper = np.uint8([5,5,5]), np.uint8([0,0,0])
    mask = cv2.inRange(image, lower, upper)
    
    cv2.imshow("Mask", mask)
    cv2.imshow("Image", image)

    if cv2.waitKey(FEED_WAIT_DELAY_MS) & 0xff == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()