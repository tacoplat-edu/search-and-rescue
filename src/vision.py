import os
import math
import time

import cv2
import numpy as np

from motion import MotionController
from models.rescue import RescueState
from p_control import PController
from models.wheel import Wheel
from helpers.vision import get_dot_locations

FEED_WAIT_DELAY_MS = 1
FRAME_SAMPLE_DELAY_S = 0.1
PX_TO_CM = 13 / 640
CORRECTION_SCALE_FACTOR = 0.01
SHOW_IMAGES = os.environ.get("SHOW_IMAGE_WINDOW") == "true"

class VisionProcessor:
    running: bool
    capture: cv2.VideoCapture
    capture_config: dict[int, float]
    motion: MotionController
    rescue_state: RescueState
    reference_locs: list[int]
    p_controller: PController

    def __init__(
        self,
        motion: MotionController,
        config_params: dict[int, float],
    ) -> None:
        self.running = False
        self.capture = cv2.VideoCapture(0)
        self.rescue_state = RescueState()
        self.p_controller = PController(kp=2.5, scale_factor=CORRECTION_SCALE_FACTOR)
        self.motion = motion
        self.capture_config = config_params

        for k, v in self.capture_config.items():
            self.capture.set(k, v)

        width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        y_locs = get_dot_locations(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.reference_locs = [
            (int(width // 2), y_loc)
            for y_loc in y_locs
        ]

    def get_path_mask(self, image):
        if image is None:
            return None
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red1_lower, red1_upper = np.uint8([0, 100, 30]), np.uint8([10, 255, 255])
        red2_lower, red2_upper = np.uint8([160, 100, 30]), np.uint8([180, 255, 255])

        mask1 = cv2.inRange(hsv_image, red1_lower, red1_upper)
        mask2 = cv2.inRange(hsv_image, red2_lower, red2_upper)
        mask = cv2.bitwise_or(mask1, mask2)

        return cv2.bitwise_and(image, image, mask=mask)

    """
        Detect blue, to trigger pickup.
    """

    def get_danger_mask(self, image):
        if image is None:
            return None
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        blue_lower, blue_upper = np.uint8([100, 100, 30]), np.uint8([140, 255, 255])
        mask = cv2.inRange(hsv_image, blue_lower, blue_upper)

        return cv2.bitwise_and(image, image, mask=mask)

    """
        Detect green, to trigger drop-off.
    """

    def get_safe_mask(self, image):
        if image is None:
            return None
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        green_lower, green_upper = np.uint8([40, 40, 20]), np.uint8([95, 255, 255])
        mask = cv2.inRange(hsv_image, green_lower, green_upper)

        return cv2.bitwise_and(image, image, mask=mask)
    
    def detect_special_contours(self, mask, threshold: int = 50):
        if mask is None:
            return None
        
        grayscale = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(grayscale, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        if contours:
            primary = max(contours, key=cv2.contourArea)
            if cv2.contourArea(primary) > threshold:
                return primary
        
        return None

    """
        Returns the path contour and a list of coordinates of points (purple) on the 
        centreline with the same y-values as the reference coordinates (blue).
    """

    def get_path_data(
        self, mask
    ):
        if mask is None:
            return None, None

        grayscale = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        contours, _ = cv2.findContours(grayscale, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if contours:
            primary_contour = max(contours, key=cv2.contourArea)

            locs = []
            for i in range(len(self.reference_locs)):
                abs_y = self.reference_locs[i][1]

                contour_points = [pt[0] for pt in primary_contour if pt[0][1] == abs_y]
                if contour_points:
                    abs_x = int(np.mean([pt[0] for pt in contour_points]))
                    locs.append((abs_x, abs_y))

                if not locs:
                    return primary_contour, None
            return primary_contour, locs

        return None, None

    def calibrate(self):
        while True:
            _, image = self.capture.read()  # camera frame BGR

            path_mask = self.get_path_mask(image)
            path, path_locs = self.get_path_data(path_mask)

            deg_turned = 0
            if not path:
                self.motion.turn(30)
                deg_turned += 30
                if deg_turned % 360 == 0:
                    self.motion.move(0.1)
            else:
                if path_locs is not None:
                    dx = path_locs[-1][0] - self.reference_locs[-1][0]
                    dy = int(
                        image.shape[0]
                        - self.reference_locs[-1][0]
                    )

                    theta = math.atan(dx / dy)

                    self.motion.turn(theta)

                    return True

    def run(self):
        # Restart the stream if not already opened
        if not self.running:
            self.running = True
            self.capture.open(0)
            for k, v in self.capture_config.items():
                self.capture.set(k, v)
            self.rescue_state = RescueState()

        self.motion.start(self.motion.default_speed)

        while self.running:
            _, image = self.capture.read()  # camera frame, BGR

            path_mask = self.get_path_mask(image)
            path, path_locs = self.get_path_data(path_mask) # Purple dots along path centreline

            # Draw blue dots
            for loc in self.reference_locs:
                cv2.circle(image, loc, 6, (255, 0, 0))

            danger_mask = self.get_danger_mask(image)
            safe_mask = self.get_safe_mask(image)

            danger = self.detect_special_contours(danger_mask, 153600)
            safe = self.detect_special_contours(safe_mask, 153600)

            # Draw contours onto the frame
            if path is not None:
                cv2.drawContours(image, path, -1, (0, 0, 255), 2)
                if path_locs is not None:
                    for i in range(len(path_locs)):
                        cv2.circle(image, path_locs[i], 6, (255, 0, 255))
            if danger is not None:
                cv2.drawContours(image, danger, -1, (255, 0, 0), 2)
            if safe is not None:
                cv2.drawContours(image, safe, -1, (0, 255, 0), 2)

            if SHOW_IMAGES:
                cv2.imshow("Image", image)

            print(
                "motor state", 
                self.motion.devices.wheel_motors[Wheel.LEFT].value,
                self.motion.devices.wheel_motors[Wheel.RIGHT].value
            )

            # Look for blue only
            if (
                not self.rescue_state.is_rescue_complete
                and not self.rescue_state.is_figure_held
            ):
                if danger is not None:
                    print("blue detected")
                    self.motion.stop()
                    time.sleep(2.5)
                    res = self.motion.turn(180, 60)
                    if res:
                        self.motion.start(self.motion.default_speed)
                    self.rescue_state.is_figure_held = res
            # Look for green only
            elif (
                not self.rescue_state.is_rescue_complete
                and self.rescue_state.is_figure_held
            ):
                if safe is not None:
                    print('green detected')
                    self.motion.stop()
                    time.sleep(2.5)
                    self.motion.move(-45, 6)
                    self.rescue_state.is_rescue_complete = True

                    time.sleep(2)
                    break

            # Always look for red if not for the other two colours
            if (
                path is not None and 
                path_locs is not None
            ):
                error = (path_locs[-1][0] - self.reference_locs[-1][0]) * PX_TO_CM
                print("error", error)

                correction = self.p_controller.compute_correction(error)

                if abs(error) > 1.0 and abs(correction) < 0.05:
                    correction = 0.05 * (-1 if error < 0 else 1)
    
                left_speed = max(0.1, min(1.0, self.motion.default_speed + correction))
                right_speed = max(0.1, min(1.0, self.motion.default_speed - correction))
                
                print(f"Setting speeds: L={left_speed:.2f}, R={right_speed:.2f}")
                
                self.motion.set_forward_speed(left_speed, Wheel.LEFT)
                self.motion.set_forward_speed(right_speed, Wheel.RIGHT)

            else:
                # Need to run recalibration algorithm
                pass

            if cv2.waitKey(FEED_WAIT_DELAY_MS) & 0xFF == ord("q"):
                break

            time.sleep(FRAME_SAMPLE_DELAY_S)

        if self.capture is not None and self.capture.isOpened():
            self.capture.release()
        cv2.destroyAllWindows()
