import cv2
import numpy as np

from typing import Optional

from models.devices import DeviceConfiguration

FEED_WAIT_DELAY_MS = 1


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

    def get_path_mask(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        red1_lower, red1_upper = np.uint8([0, 50, 20]), np.uint8([10, 255, 255])
        red2_lower, red2_upper = np.uint8([170, 50, 20]), np.uint8([180, 255, 255])

        mask1 = cv2.inRange(image, red1_lower, red1_upper)
        mask2 = cv2.inRange(image, red2_lower, red2_upper)

        return cv2.bitwise_or(mask1, mask2)

    def get_path_curvature(mask: cv2.typing.MatLike):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        if len(contours) <= 0:
            return None

        primary_contour = max(contours)
        moments = cv2.moments(primary_contour)

        if moments["m00"] == 0:
            return None

        path_centre = {
            "x": int(moments["m10"] / moments["m00"]),
            "y": int(moments["m01"] / moments["m00"]),
        }

        # process path centroid

    def run(self):
        while True:
            _, image = self.capture.read()

            mask = self.get_path_mask(image)
            curvature = self.get_path_curvature(mask)

            # perform motor actions

            cv2.imshow("Mask", mask)
            cv2.imshow("Image", image)

            if cv2.waitKey(FEED_WAIT_DELAY_MS) & 0xFF == ord("q"):
                break

        self.capture.release()
        cv2.destroyAllWindows()
