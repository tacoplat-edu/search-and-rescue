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
MAX_CORRECITON = 0.09
MIN_SPEED = 0.30
MAX_SPEED = 0.42


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
        self.p_controller = PController(kp=0.55, scale_factor=CORRECTION_SCALE_FACTOR)
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
        if image is None:
            return None
        
        danger_mask = self.get_danger_mask(image)
        if danger_mask is None:
            return None
        
        grayscale = cv2.cvtColor(danger_mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(grayscale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        blue_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(blue_contour) < 300:  
            return None
        
        M = cv2.moments(blue_contour)
        if M["m00"] == 0:
            return None
        
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
        kernel = np.ones((5,5), np.uint8)
        cleaned = cv2.morphologyEx(grayscale, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return None, None

        primary_contour = max(contours, key=cv2.contourArea)
        height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        y_values = []
        bottom_range = np.linspace(int(height * 0.9), int(height * 0.7), 4) 
        top_range = np.linspace(int(height * 0.65), int(height * 0.4), 3)    
        y_values = np.concatenate((bottom_range, top_range))
        
        locs = []
        for abs_y in y_values:
            abs_y = int(abs_y)
            tolerance = 3  
       
            contour_points = [pt[0] for pt in primary_contour if abs(pt[0][1] - abs_y) <= tolerance]
            
            if contour_points:
                leftmost = min(contour_points, key=lambda pt: pt[0])
                rightmost = max(contour_points, key=lambda pt: pt[0])
                abs_x = int((leftmost[0] + rightmost[0]) / 2)
                locs.append((abs_x, abs_y))
        
        if locs:
            self.reference_locs = [(int(width // 2), loc[1]) for loc in locs]
        
        return primary_contour, locs if locs else None 

    def detect_curve(self, path_locs):
        if not path_locs or len(path_locs) < 4:
            return False, 0, "straight"
            
        x_coords = [pt[0] for pt in path_locs]
        
        center_x = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) // 2
        
        left_count = sum(1 for x in x_coords if x < center_x - 20)
        right_count = sum(1 for x in x_coords if x > center_x + 20)
        
        x_diffs = np.diff(x_coords)
        
        is_curve = False
        curve_direction = "straight"
        curve_strength = 0
        
        if len(x_diffs) >= 3:
            consistent_right = all(diff > 0 for diff in x_diffs)
            consistent_left = all(diff < 0 for diff in x_diffs)
            
            if consistent_right or right_count > len(x_coords) * 0.6:
                is_curve = True
                curve_direction = "right"
                curve_strength = sum(x_diffs) / len(x_diffs)
            elif consistent_left or left_count > len(x_coords) * 0.6:
                is_curve = True
                curve_direction = "left"
                curve_strength = -sum(abs(diff) for diff in x_diffs) / len(x_diffs)
        
        return is_curve, curve_strength, curve_direction
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


    def calculate_adaptive_error(self, path_locs):
        if not path_locs or len(path_locs) < 2:
            return 0, 0
        
        is_curve, curve_strength, curve_direction = self.detect_curve(path_locs)
        
        errors = []
        for i, (path_loc, ref_loc) in enumerate(zip(path_locs, self.reference_locs[:len(path_locs)])):
            error = (path_loc[0] - ref_loc[0]) * PX_TO_CM
            errors.append(error)
        
        #top_error = errors[-1]
        
        if is_curve:
            if curve_direction == "left":
                weights = np.linspace(0.4, 1.0, len(errors))
            else: 
                weights = np.linspace(0.4, 1.0, len(errors))
                
            weighted_error = sum(error * weight for error, weight in zip(errors, weights)) / sum(weights)
        else:
            if len(errors) >= 3:
                weights = [0.6, 1.0, 0.8] + [0.7] * (len(errors) - 3)
                weighted_error = sum(error * weight for error, weight in zip(errors, weights)) / sum(weights)
            else:
                weighted_error = sum(errors) / len(errors)
        
        return weighted_error, is_curve, curve_direction, curve_strength
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
                
                if danger_data: 
                    touches_left = danger_data['touches_left']
                    touches_right = danger_data['touches_right']
                    
                    if touches_left or touches_right:
                        if touches_left and not touches_right:
                            self.motion.turn(-15, 40)
                            time.sleep(1.0)
                            continue
                            
                        elif touches_right and not touches_left:
                            self.motion.turn(15, 40) 
                            time.sleep(1.0)  
                            continue
                            
                        elif touches_left and touches_right:
                            self.motion.move(-10, 25)  
                            time.sleep(1.0)
                            continue
                    
                    # Align with center of the blue target
                    x_offset = danger_data['x_offset']
                    if abs(x_offset) > 40:
                        print(f"Aligning with blue target, offset: {x_offset}px")
                        
                        turn_angle = x_offset * 0.1
                        self.motion.turn(turn_angle, 40)
                        time.sleep(1.0)
                        continue
                    
                    # Target is centered, move forward  
                    print("Blue target centered - performing pickup")
                    res = self.motion.move(20, 25)
                    self.rescue_state.is_figure_held = res
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
                     weighted_error, is_curve, curve_direction, curve_strength = self.calculate_adaptive_error(path_locs)
        
                print(f"Path: {'CURVE '+curve_direction if is_curve else 'STRAIGHT'}, " +
                    f"Strength: {curve_strength:.1f}, Error: {weighted_error:.2f}cm")
                
                correction = self.p_controller.compute_correction(weighted_error)
            

                if is_curve:
                    curve_boost = min(0.04, abs(curve_strength) * 0.001)
                    
                    if curve_direction == "left":
                        correction -= curve_boost
                    else: 
                        correction += curve_boost
                        
                    if abs(weighted_error) > 3.0:
                        correction *= 1.2
                
                self.last_error = weighted_error
                self.last_correction = correction
                self.blind_frames = 0

                if abs(weighted_error) > 1.0 and abs(correction) < 0.05:
                    correction = 0.05 * (-1 if weighted_error < 0 else 1)

                correction = max(-MAX_CORRECITON, min(MAX_CORRECITON, correction))
                default_speed = self.motion.default_speed
        
                if is_curve:
                        curve_speed_factor = 0.9
                        adjusted_default = default_speed * curve_speed_factor
                        
                        if curve_direction == "left":
                            left_speed = max(MIN_SPEED, adjusted_default - abs(correction) * 1.2)
                            right_speed = adjusted_default
                        else: 
                            left_speed = adjusted_default
                            right_speed = max(MIN_SPEED, adjusted_default - abs(correction) * 1.2)
                else:
                    if abs(correction) == MAX_CORRECITON:
                        left_speed = max(MIN_SPEED, min(MAX_SPEED, default_speed + correction))
                        right_speed = max(MIN_SPEED, min(MAX_SPEED, default_speed - correction))
                    else:
                        if correction > 0:  
                            left_speed = min(MAX_SPEED, default_speed + correction)
                            right_speed = max(MIN_SPEED, default_speed)
                        else: 
                            left_speed = max(MIN_SPEED, default_speed)
                            right_speed = min(MAX_SPEED, default_speed - correction)

                print(f"Setting speeds: L={left_speed:.2f}, R={right_speed:.2f}")
                self.motion.set_forward_speed(left_speed, Wheel.LEFT)
                self.motion.set_forward_speed(right_speed, Wheel.RIGHT)

            else:
                # Need to run recalibration algorithm
                # self.blind_frames += 1
                # if self.blind_frames < self.max_blind_recovery:
                #     print(f"Cant see line, last error is {self.last_error}")

                #     default_speed = self.motion.default_speed

                #     if self.last_error < 0:
                #         left_speed = MIN_SPEED
                #         right_speed = min(MAX_SPEED, default_speed - self.last_correction)
                #         print("Last seen line on left, moving right wheel")
                #     else:
                #         left_speed = min(MAX_SPEED, default_speed + self.last_correction)
                #         right_speed = MIN_SPEED 
                #         print("Last seen line on right, moving left wheel")
                    
                #     print(f"Recovery speeds: L={left_speed:.2f}, R={right_speed:.2f}")
                    
                #     self.motion.set_forward_speed(left_speed, Wheel.LEFT)
                #     self.motion.set_forward_speed(right_speed, Wheel.RIGHT)
                    
                # else:
                #     print("Spinning to find line")
                #     if self.last_error < 0:
                #         self.motion.set_reverse_speed(0.40, Wheel.LEFT)
                #         self.motion.set_forward_speed(0.40, Wheel.RIGHT)
                #     else:
                #         self.motion.set_forward_speed(0.40, Wheel.LEFT)
                #         self.motion.set_reverse_speed(0.40, Wheel.RIGHT)
                pass

            if cv2.waitKey(FEED_WAIT_DELAY_MS) & 0xFF == ord("q"):
                break

            time.sleep(FRAME_SAMPLE_DELAY_S)

        if self.capture is not None and self.capture.isOpened():
            self.capture.release()
        cv2.destroyAllWindows()
