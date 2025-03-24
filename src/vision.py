import os
import math
import time

import cv2
import numpy as np

from motion import MotionController
from models.rescue import RescueState
from pid_control import PIDController
from models.wheel import Wheel
from helpers.vision import get_dot_locations

FEED_WAIT_DELAY_MS = 1
FRAME_SAMPLE_DELAY_S = 0.1
PX_TO_CM = 13 / 640
CORRECTION_SCALE_FACTOR = 0.01
SHOW_IMAGES = os.environ.get("SHOW_IMAGE_WINDOW") == "true"
#MIN_SPEED = 0.05
#MAX_SPEED = 0.35
#TURN_SPEED = 0.40

class VisionProcessor:
    running: bool
    capture: cv2.VideoCapture
    capture_config: dict[int, float]
    motion: MotionController
    rescue_state: RescueState
    reference_locs: list[int]
    pid_controller: PIDController

    def __init__(
        self,
        motion: MotionController,
        config_params: dict[int, float],
    ) -> None:
        self.running = False
        self.capture = cv2.VideoCapture(0, cv2.CAP_V4L2) # use v4l2 video capture for rpi
        self.rescue_state = RescueState()
        self.pid_controller = PIDController(kp=2.5, ki=0.02, kd= 0.3, scale_factor=CORRECTION_SCALE_FACTOR)
        self.motion = motion
        self.capture_config = config_params

        self.last_error = 0
        self.last_correction = 0
        self.blind_frames = 0
        self.max_blind_recovery = 20

        for k, v in self.capture_config.items():
            self.capture.set(k, v)

        width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # tune these for adjusting turn timing - higher = turn earlier
        self.lookahead_rows = [
            int(height * 0.85),  # Near
            int(height * 0.7),   # Mid
            int(height * 0.55),  # Far
            int(height * 0.4)    # Very far
        ]

        # Adjust weights to include the new point
        self.lookahead_weights = [0.4, 0.25, 0.2, 0.15]

        # Bird's eye view perspective transform
        self.setup_perspective_transform(width, height)
        self.warped_width = 640
        self.warped_height = 480
        self.reference_locs = self._create_reference_points(width, height)
        
    def setup_perspective_transform(self, width, height):
        """Set up the perspective transform for bird's eye view"""
        # Source points in the original image (adjust these based on your camera setup)
        self.src_points = np.float32([
            [width * 0.25, height * 0.9],   
            [width * 0.75, height * 0.9], 
            [width * 0.1, height * 0.5],    
            [width * 0.9, height * 0.5]    
        ])
        
        self.warped_width = 640
        self.warped_height = 480
        self.dst_points = np.float32([
            [0, self.warped_height],                 # Bottom left
            [self.warped_width, self.warped_height], # Bottom right
            [0, 0],                                 # Top left
            [self.warped_width, 0]                  # Top right
        ])
        
        self.warp_matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.unwarp_matrix = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

    def get_birds_eye_view(self, frame):
        """Transform the image to bird's eye view perspective"""
        if frame is None:
            return None
        
        warped = cv2.warpPerspective(
            frame, 
            self.warp_matrix, 
            (self.warped_width, self.warped_height), 
            flags=cv2.INTER_LINEAR
        )
        return warped
    
    def _create_reference_points(self, width, height):
        """Create reference points at the center of each look-ahead row"""
        print(f"Camera width: {width}, height: {height}")
        
        center_x = int(width // 2)
        reference_points = [(center_x, y) for y in self.lookahead_rows]
        
        print(f"Reference points calculated at: {reference_points}")
        
        return reference_points

    def get_path_mask(self, image):
        """Extract the red line from the image"""
        if image is None:
            return None
            
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        red1_lower, red1_upper = np.uint8([0, 70, 20]), np.uint8([15, 255, 255])
        red2_lower, red2_upper = np.uint8([155, 70, 20]), np.uint8([180, 255, 255])
        
        mask1 = cv2.inRange(hsv_image, red1_lower, red1_upper)
        mask2 = cv2.inRange(hsv_image, red2_lower, red2_upper)
    
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations to clean up the mask
        kernel_close = np.ones((7, 7), np.uint8) 
        kernel_open = np.ones((3, 3), np.uint8) 
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
        mask = cv2.dilate(mask, kernel_open, iterations=1)
        return mask, cv2.bitwise_and(image, image, mask=mask)

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
            'contour': blue_contour, 
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
    
    def get_path_points_with_lookahead(self, binary_mask):
        """Get line points at multiple look-ahead distances"""
        if binary_mask is None:
            return None, None
            
        # Get image dimensions
        height, width = binary_mask.shape
        center_x = width // 2
        
        # Find contours in binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return None, None
            
        # Filter contours by minimum area
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
        if not valid_contours:
            return None, None
            
        # Get the largest contour
        primary_contour = max(valid_contours, key=cv2.contourArea)
        
        # Initialize list to store detected points
        path_points = []
        
        # For each look-ahead row, find where the line intersects
        for row in self.lookahead_rows:
            # Extract all contour points at this row (with tolerance)
            tolerance = 3
            row_points = [pt[0] for pt in primary_contour if abs(pt[0][1] - row) <= tolerance]
            
            if row_points:
                # Calculate the average x-position at this row
                avg_x = int(np.mean([pt[0] for pt in row_points]))
                path_points.append((avg_x, row))
            else:
                # If no points found at this row, use None to mark missing data
                path_points.append(None)
        
        return primary_contour, path_points
    
    def calculate_weighted_error(self, path_points):
        """Calculate weighted error based on multiple look-ahead points"""
        if not path_points or all(pt is None for pt in path_points):
            return None, None, None, []
        
        # Filter out None points
        valid_points = [(i, pt) for i, pt in enumerate(path_points) if pt is not None]
        if not valid_points:
            return None, None, None, []
        
        # Calculate errors for each valid point
        errors = []
        for i, point in valid_points:
            ref_point = self.reference_locs[i]
            error = (point[0] - ref_point[0]) * PX_TO_CM
            errors.append((i, error))
        
        # If we don't have all points, adjust weights
        if len(errors) < len(self.lookahead_weights):
            # Create new weights normalized to sum to 1
            adjusted_weights = {}
            total_weight = 0
            
            # Only include weights for indices we have
            for i, _ in errors:
                if i < len(self.lookahead_weights):
                    adjusted_weights[i] = self.lookahead_weights[i]
                    total_weight += self.lookahead_weights[i]
            
            # Normalize weights
            if total_weight > 0:
                for i in adjusted_weights:
                    adjusted_weights[i] = adjusted_weights[i] / total_weight
            else:
                # Equal weights if we can't normalize
                even_weight = 1.0 / len(errors)
                for i, _ in errors:
                    adjusted_weights[i] = even_weight
        else:
            # Use the original weights as a dictionary
            adjusted_weights = {i: self.lookahead_weights[i] for i in range(len(self.lookahead_weights))}
        
        # Calculate weighted error using the dictionary
        weighted_sum = sum(error * adjusted_weights[i] for i, error in errors)
        
        # Return near and far errors for display, plus all errors
        near_error = errors[0][1] if errors and errors[0][0] == 0 else None
        far_error = errors[-1][1] if errors else None
        
        all_errors = [error for _, error in errors]
        
        return near_error, far_error, weighted_sum, all_errors
    
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
            self.pid_controller.reset()
        
        if SHOW_IMAGES:
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Bird's Eye View", cv2.WINDOW_NORMAL)

        self.motion.start(self.motion.default_speed)

        while self.running:
            _, image = self.capture.read()  # camera frame, BGR

            birds_eye = self.get_birds_eye_view(image)
            display = image.copy() if SHOW_IMAGES else None
            birds_eye_display = birds_eye.copy() if birds_eye is not None else None
            height, width = image.shape[:2]

            center_x = width // 2

            if birds_eye is not None:
                binary_mask, path_mask = self.get_path_mask(birds_eye)
                path_contour, path_points = self.get_path_points_with_lookahead(binary_mask)
            else:
                binary_mask, path_mask = self.get_path_mask(image)
                path_contour, path_points = self.get_path_points_with_lookahead(binary_mask)
            
            danger_mask = self.get_danger_mask(image)
            safe_mask = self.get_safe_mask(image)
            danger_data = self.get_danger_data(image)
            danger = self.detect_special_contours(danger_mask, 153600)
            safe = self.detect_special_contours(safe_mask, 153600)
            # Draw contours onto the frame
          
            # Draw visualization if showing images
            if SHOW_IMAGES:
                # Draw reference points
                for loc in self.reference_locs:
                    cv2.circle(display, loc, 6, (255, 0, 0), -1)  # Blue dots
                    if birds_eye_display is not None:
                        cv2.circle(birds_eye_display, loc, 6, (255, 0, 0), -1)
                
                # Draw center line
                cv2.line(display, (center_x, 0), (center_x, height), (0, 255, 0), 1)
                
                # Draw path contour and points
                if path_contour is not None:
                    cv2.drawContours(display, [path_contour], -1, (0, 0, 255), 2)  # Red contour
                    if birds_eye_display is not None:
                        cv2.drawContours(birds_eye_display, [path_contour], -1, (0, 0, 255), 2)
                
                if path_points and any(pt is not None for pt in path_points):
                    for point in path_points:
                        if point is not None:
                            cv2.circle(display, point, 6, (255, 0, 255), -1)  # Purple dots
                            if birds_eye_display is not None:
                                cv2.circle(birds_eye_display, point, 6, (255, 0, 255), -1)
                
                # Draw danger (blue) contour
                # Update these lines for drawing the danger (blue) contour
                if danger_data is not None and 'contour' in danger_data and 'center' in danger_data:
                    cv2.drawContours(display, [danger_data['contour']], -1, (255, 0, 0), 2)
                    cv2.circle(display, danger_data['center'], 8, (0, 255, 255), -1)
                
                # Draw safe (green) contour
                if safe is not None:
                    cv2.drawContours(display, [safe], -1, (0, 255, 0), 2)
               
                # Add tuning parameter display at the bottom of screen
                param_y = height - 120
                cv2.putText(display, f"PID: kp={self.pid_controller.kp:.2f}, ki={self.pid_controller.ki:.2f}, kd={self.pid_controller.kd:.2f}", 
                            (20, param_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display, f"Scale: {self.pid_controller.scale_factor:.4f}", 
                            (400, param_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show lookahead weights
                weight_text = "Weights: " + ", ".join([f"{w:.2f}" for w in self.lookahead_weights])
                cv2.putText(display, weight_text, (20, param_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add bird's eye view calibration visualization
                if birds_eye_display is not None:
                    # Draw horizontal lines at each lookahead row
                    for row in self.lookahead_rows:
                        cv2.line(birds_eye_display, (0, row), (width, row), (0, 255, 0), 1)
                    
                    # Draw vertical center line
                    cv2.line(birds_eye_display, (center_x, 0), (center_x, height), (0, 255, 0), 1)
                    
                    # Show the source points of perspective transform on original image
                    for point in self.src_points:
                        cv2.circle(display, (int(point[0]), int(point[1])), 4, (0, 165, 255), -1)
                        
                    # Outline the region being transformed to bird's eye view
                    src_points_int = np.array(self.src_points, dtype=np.int32)
                    cv2.polylines(display, [src_points_int], True, (0, 165, 255), 2)
        
            # Initialize motor control variables
            default_speed = self.motion.default_speed
            left_speed = default_speed
            right_speed = default_speed
            correction = 0
            
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

                pass
                
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
            
            # Handle line following with PID control
            if path_contour is not None and path_points and any(pt is not None for pt in path_points):
                # Calculate weighted error based on multiple look-ahead points
                near_error, far_error, weighted_error, all_errors = self.calculate_weighted_error(path_points)
                
                if weighted_error is not None:
                    correction = self.pid_controller.compute_correction(weighted_error)
                    
                    # Apply correction to motor speeds
                    left_speed = default_speed + correction
                    right_speed = default_speed - correction
                    
                    # Display visualization information if showing images
                    if SHOW_IMAGES:
                        direction = "RIGHT" if weighted_error > 0 else "LEFT" if weighted_error < 0 else "CENTER"
                        
                        if near_error is not None:
                            cv2.putText(display, f"Near Error: {near_error:.2f} cm", 
                                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        if far_error is not None:
                            cv2.putText(display, f"Far Error: {far_error:.2f} cm", 
                                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        cv2.putText(display, f"Weighted Error: {weighted_error:.2f} cm", 
                                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        cv2.putText(display, f"PID Correction: {correction:.3f}", 
                                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        
                        cv2.putText(display, f"Turn: {direction}", 
                                (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        
                        # Draw error lines
                        for i, error in enumerate(all_errors):
                            if path_points[i] is not None:
                                ref_point = self.reference_locs[i]
                                path_point = path_points[i]
                                cv2.line(display, 
                                        (ref_point[0], ref_point[1]), 
                                        (path_point[0], path_point[1]), 
                                        (0, 255, 255), 2)
                                if birds_eye_display is not None:
                                    cv2.line(birds_eye_display, 
                                            (ref_point[0], ref_point[1]), 
                                            (path_point[0], path_point[1]), 
                                            (0, 255, 255), 2)
                    
                    # Save the last successful error and correction
                    self.last_error = weighted_error
                    self.last_correction = correction
                    self.blind_frames = 0
                
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
                #         self.motion.set_reverse_speed(TURN_SPEED, Wheel.LEFT)
                #         self.motion.set_forward_speed(TURN_SPEED, Wheel.RIGHT)
                #     else:
                #         self.motion.set_forward_speed(TURN_SPEED, Wheel.LEFT)
                #         self.motion.set_reverse_speed(TURN_SPEED, Wheel.RIGHT)
                pass

            # Display images if enabled
            if SHOW_IMAGES:
                cv2.imshow("Image", display)
                if birds_eye_display is not None:
                    cv2.imshow("Bird's Eye View", birds_eye_display)
        
            if cv2.waitKey(FEED_WAIT_DELAY_MS) & 0xFF == ord("q"):
                break

            time.sleep(FRAME_SAMPLE_DELAY_S)

        if self.capture is not None and self.capture.isOpened():
            self.capture.release()
        cv2.destroyAllWindows()
