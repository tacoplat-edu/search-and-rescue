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
MAX_CORRECITON = 0.1
MIN_SPEED = 0.18
MAX_SPEED = 0.40
TURN_SPEED = 0.65

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
        self.p_controller = PController(kp=0.75, scale_factor=CORRECTION_SCALE_FACTOR)
        self.motion = motion
        self.capture_config = config_params

        self.last_error = 0
        self.last_correction = 0
        self.blind_frames = 0
        self.max_blind_recovery = 20

        for k, v in self.capture_config.items():
            self.capture.set(k, v)

        width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        y_locs = get_dot_locations(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.reference_locs = [(int(width // 2), y_loc) for y_loc in y_locs]

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

    def get_danger_data(self, image):
        """Returns blue contour, center, and border data for better alignment"""
        if image is None:
            return None, None, None
        
        danger_mask = self.get_danger_mask(image)
        if danger_mask is None:
            return None, None, None
        
        grayscale = cv2.cvtColor(danger_mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(grayscale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, None
        
        blue_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(blue_contour) < 300:  
            return None, None, None
        
        M = cv2.moments(blue_contour)
        if M["m00"] == 0:
            return blue_contour, None, None
        
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        center = (center_x, center_y)
        
        frame_width = image.shape[1]
        frame_center_x = frame_width // 2
        
        leftmost = tuple(blue_contour[blue_contour[:, :, 0].argmin()][0])
        rightmost = tuple(blue_contour[blue_contour[:, :, 0].argmax()][0])
        
        border_margin = 5  
        touches_left = leftmost[0] <= border_margin
        touches_right = rightmost[0] >= frame_width - border_margin
        
        alignment_data = {
            'center': center,
            'leftmost': leftmost,
            'rightmost': rightmost,
            'width': rightmost[0] - leftmost[0],
            'x_offset': center_x - frame_center_x,
            'y_position': center_y / image.shape[0],
            'touches_left': touches_left,
            'touches_right': touches_right,
            'touches_border': touches_left or touches_right
        }
        
        return alignment_data
    def get_path_data(self, mask):
        if mask is None:
            return None, None

        grayscale = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        contours, _ = cv2.findContours(grayscale, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if contours:
            primary_contour = max(contours, key=cv2.contourArea)

            locs = []
            for i in range(len(self.reference_locs)):
                abs_y = self.reference_locs[i][1]

                tolerance = 2
                contour_points = [pt[0] for pt in primary_contour if abs(pt[0][1] - abs_y) <= tolerance]
                if contour_points:
                    abs_x = int(np.mean([pt[0] for pt in contour_points]))
                    locs.append((abs_x, abs_y))

            return primary_contour, locs if locs else None 

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
                    dy = int(image.shape[0] - self.reference_locs[-1][0])

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
            path, path_locs = self.get_path_data(
                path_mask
            )  # Purple dots along path centreline

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
                self.motion.devices.wheel_motors[Wheel.RIGHT].value,
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

                danger_data = self.get_danger_data(image)
                
                # if danger_data: 
                #     touches_left = danger_data['touches_left']
                #     touches_right = danger_data['touches_right']
                    
                #     if touches_left or touches_right:
                #         if touches_left and not touches_right:
                #             self.motion.turn(-15, 40)
                #             time.sleep(1.0)
                #             continue
                            
                #         elif touches_right and not touches_left:
                #             self.motion.turn(15, 40) 
                #             time.sleep(1.0)  
                #             continue
                            
                #         elif touches_left and touches_right:
                #             self.motion.move(-10, 25)  
                #             time.sleep(1.0)
                #             continue
                    
                #     # Align with center of the blue target
                #     x_offset = danger_data['x_offset']
                #     if abs(x_offset) > 40:
                #         print(f"Aligning with blue target, offset: {x_offset}px")
                        
                #         turn_angle = x_offset * 0.1
                #         self.motion.turn(turn_angle, 40)
                #         time.sleep(1.0)
                #         continue
                    
                    # Target is centered, move forward  
                    # print("Blue target centered - performing pickup")
                    # res = self.motion.move(20, 25)
                    # self.rescue_state.is_figure_held = res
            # Look for green only
            elif (
                not self.rescue_state.is_rescue_complete
                and self.rescue_state.is_figure_held
            ):
                if safe is not None:
                    print("green detected")
                    self.motion.stop()
                    time.sleep(2.5)
                    self.motion.move(-45, 6)
                    self.rescue_state.is_rescue_complete = True

                    time.sleep(2)
                    break

            # Always look for red if not for the other two colours
            if path is not None and path_locs is not None:
                if len(path_locs) >= 1:
                    top_index = len(path_locs) - 1 
                    error = (path_locs[top_index][0] - self.reference_locs[top_index][0]) * PX_TO_CM
                    self.blind_frames = 0 

                print("error", error)

                correction = self.p_controller.compute_correction(error)

                self.last_error = error
                self.last_correction = correction

                #correction = max(-MAX_CORRECITON, min(MAX_CORRECITON, correction))

                default_speed = self.motion.default_speed

                # # Try only adjusting one motor for p-control at a time when correction is large
                # if abs(correction) == MAX_CORRECITON:
                # When turning right (negative error/correction):
                if correction < 0:
                    left_speed = max(MIN_SPEED, default_speed + correction)  
                    right_speed = min(MAX_SPEED, default_speed - correction)  
                else:
                    left_speed = min(MAX_SPEED, default_speed + correction)  
                    right_speed = max(MIN_SPEED, default_speed - correction)  
                    # else:
                    #     if correction > 0:  
                    #         left_speed = min(MAX_SPEED, default_speed + correction)
                    #         right_speed = max(MIN_SPEED, default_speed )
                    #     else: 
                    #         left_speed = max(MIN_SPEED, default_speed)
                    #         right_speed = min(MAX_SPEED, default_speed - correction)
        

                    print(f"Setting speeds: L={left_speed:.2f}, R={right_speed:.2f}")

                self.motion.set_forward_speed(left_speed, Wheel.LEFT)
                self.motion.set_forward_speed(right_speed, Wheel.RIGHT)

            else:
                # Need to run recalibration algorithm
                self.blind_frames += 1
                if self.blind_frames < self.max_blind_recovery:
                    print(f"Cant see line, last error is {self.last_error}")

                    default_speed = self.motion.default_speed

                    if self.last_error < 0:
                        left_speed = MIN_SPEED
                        right_speed = min(MAX_SPEED, default_speed - self.last_correction)
                        print("Last seen line on left, moving right wheel")
                    else:
                        left_speed = min(MAX_SPEED, default_speed + self.last_correction)
                        right_speed = MIN_SPEED 
                        print("Last seen line on right, moving left wheel")
                    
                    print(f"Recovery speeds: L={left_speed:.2f}, R={right_speed:.2f}")
                    
                    self.motion.set_forward_speed(left_speed, Wheel.LEFT)
                    self.motion.set_forward_speed(right_speed, Wheel.RIGHT)
                    
                else:
                    print("Spinning to find line")
                    if self.last_error < 0:
                        self.motion.set_reverse_speed(TURN_SPEED, Wheel.LEFT)
                        self.motion.set_forward_speed(TURN_SPEED, Wheel.RIGHT)
                    else:
                        self.motion.set_forward_speed(TURN_SPEED, Wheel.LEFT)
                        self.motion.set_reverse_speed(TURN_SPEED, Wheel.RIGHT)

            if cv2.waitKey(FEED_WAIT_DELAY_MS) & 0xFF == ord("q"):
                break

            time.sleep(FRAME_SAMPLE_DELAY_S)

        if self.capture is not None and self.capture.isOpened():
            self.capture.release()
        cv2.destroyAllWindows()
