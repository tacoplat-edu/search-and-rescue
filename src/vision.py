import cv2
import numpy as np

from typing import Optional

from models.devices import DeviceConfiguration
from helpers.vision import get_dot_locations, draw_dots

FEED_WAIT_DELAY_MS = 1
NUM_DOTS = 3

class VisionProcessor:
    capture: cv2.VideoCapture
    devices: Optional[DeviceConfiguration]

    def __init__(
        self,
        devices: Optional[DeviceConfiguration],
        config_params: dict[int, float],
    ) -> None:
        self.capture = cv2.VideoCapture(0)
        if devices:
            self.devices = devices
        for k, v in config_params.items():
            self.capture.set(k, v)

    def get_path_mask(self, image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red1_lower, red1_upper = np.uint8([0, 100, 30]), np.uint8([10, 255, 255])
        red2_lower, red2_upper = np.uint8([160, 100, 30]), np.uint8([180, 255, 255])

        mask1 = cv2.inRange(hsv_image, red1_lower, red1_upper)
        mask2 = cv2.inRange(hsv_image, red2_lower, red2_upper)
        mask = cv2.bitwise_or(mask1, mask2)

        return cv2.bitwise_and(image, image, mask=mask)

    def get_path_curvature(self, image: cv2.typing.MatLike, mask: cv2.typing.MatLike):
        grayscale = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        contours, _ = cv2.findContours(grayscale, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        if contours:
            primary_contour = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(primary_contour)

            if w > 10 and h > 10:
                mask = np.zeros((h, w), dtype=np.uint8)
                adjusted_contour = primary_contour - [x, y]

                cv2.drawContours(mask, [adjusted_contour], -1, 255, thickness=1)

                locs = get_dot_locations(mask, (
                    self.capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                    self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                ))
                draw_dots(image, locs)

            return primary_contour

        return None

    def run(self):
        while True:
            _, image = self.capture.read() # BGR

            mask = self.get_path_mask(image)
            curvature = self.get_path_curvature(image, mask)

            locs = get_dot_locations(image, (
                self.capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            ))
            draw_dots(image, locs)

            if curvature is not None:
                cv2.drawContours(image, curvature, -1, (0,255,0), 2)

            cv2.imshow("Image", image)
            
            if cv2.waitKey(FEED_WAIT_DELAY_MS) & 0xFF == ord("q"):
                break

        self.capture.release()
        cv2.destroyAllWindows()

