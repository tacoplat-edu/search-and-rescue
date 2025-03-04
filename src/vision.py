import cv2
import numpy as np

from typing import Optional

from models.devices import DeviceConfiguration
from helpers.vision import get_dot_locations

FEED_WAIT_DELAY_MS = 1
NUM_DOTS = 3

class VisionProcessor:
    capture: cv2.VideoCapture
    devices: Optional[DeviceConfiguration]
    y_locs: list[int]

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
        
        self.y_locs = get_dot_locations(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

            dist_transform = cv2.distanceTransform(grayscale, cv2.DIST_L2, 5)
            cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)

            _, ridge = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
            ridge = ridge.astype(np.uint8)

            centreline = cv2.ximgproc.thinning(ridge)

            points = np.column_stack(np.where(centreline > 0))

            if len(points) < len(self.y_locs):
                return None

            points = sorted(points, key=lambda p: p[0])

            x_locs = []
            for i in range(len(self.y_locs)):
                norm_i = int(i * len(points) / len(self.y_locs))
                x_locs.append(points[norm_i][1])

            locs = list(zip(x_locs, self.y_locs))

            return primary_contour, locs

        return None

    def run(self):
        while True:
            _, image = self.capture.read() # BGR

            mask = self.get_path_mask(image)
            path, locs = self.get_path_curvature(image, mask)

            if path is not None:
                cv2.drawContours(image, path, -1, (0,255,0), 2)
                if locs is not None:
                    for loc in locs:
                        cv2.circle(image, loc, 6, (255,0,255))

            cv2.imshow("Image", image)
            
            if cv2.waitKey(FEED_WAIT_DELAY_MS) & 0xFF == ord("q"):
                break

        self.capture.release()
        cv2.destroyAllWindows()

