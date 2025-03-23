import cv2
import time
import numpy as np
from ..pid_control import PIDController 

class EnhancedVisionProcessor:
    def __init__(self, config_params=None):
        self.capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        self.config_params = config_params or {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
        }
        self.default_motor_speed = 0.3

        for k, v in self.config_params.items():
            self.capture.set(k, v)

        width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # Initiate PID controller for line following
        self.pid_controller = PIDController(kp=0.75, ki=0.02, kd=0.1, scale_factor=0.01)
        
        # Set up parameters for perspective transform (bird's eye view)
        self.setup_perspective_transform(width, height)
        
        # Define look-ahead points
        self.lookahead_rows = [
            int(height * 0.85),  # Near point (closest to robot)
            int(height * 0.7),   # Mid point
            int(height * 0.55)   # Far point (furthest from robot)
        ]
        
        # Create reference points at center of each row
        self.reference_locs = self._create_reference_points(width, height)
        self.PX_TO_CM = 13 / 640
        
        # Weights for look-ahead points (adjust these based on testing)
        self.lookahead_weights = [0.6, 0.3, 0.1]  # Near, mid, far
    
    def setup_perspective_transform(self, width, height):
        """Set up the perspective transform for bird's eye view"""
        # Source points in the original image (adjust these based on your camera setup)
        self.src_points = np.float32([
            [width * 0.25, height * 0.9],   # Bottom left
            [width * 0.75, height * 0.9],   # Bottom right
            [width * 0.1, height * 0.5],    # Top left
            [width * 0.9, height * 0.5]     # Top right
        ])
        
        # Destination points for transformed image
        self.warped_width = 640
        self.warped_height = 480
        self.dst_points = np.float32([
            [0, self.warped_height],                 # Bottom left
            [self.warped_width, self.warped_height], # Bottom right
            [0, 0],                                 # Top left
            [self.warped_width, 0]                  # Top right
        ])
        
        # Calculate perspective transform matrix
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
    
    def get_danger_mask(self, image):
        """Detect blue objects"""
        if image is None:
            return None
        
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        blue_lower, blue_upper = np.uint8([100, 100, 30]), np.uint8([140, 255, 255])
        mask = cv2.inRange(hsv_image, blue_lower, blue_upper)
        
        gray_mask = cv2.cvtColor(cv2.bitwise_and(image, image, mask=mask), cv2.COLOR_BGR2GRAY)
        
        return gray_mask     

    def get_danger_data(self, image):
        # [This method remains the same as in your original code]
        if image is None:
            return None, None, None
        
        mask = self.get_danger_mask(image)
        if mask is None or cv2.countNonZero(mask) < 100: 
            return None, None, None
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None
        
        blue_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(blue_contour) < 300: 
            return None, None, None
        
        frame_width = image.shape[1]
        frame_center_x = frame_width // 2
        
        border_margin = 5  
        
        leftmost = tuple(blue_contour[blue_contour[:, :, 0].argmin()][0])
        rightmost = tuple(blue_contour[blue_contour[:, :, 0].argmax()][0])
        
        touches_left = leftmost[0] <= border_margin
        touches_right = rightmost[0] >= frame_width - border_margin
        touches_border = touches_left or touches_right
        
        M = cv2.moments(blue_contour)
        if M["m00"] == 0:
            return blue_contour, None, None
        
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        center = (center_x, center_y)
        
        alignment_data = {
            'center': center,
            'leftmost': leftmost,
            'rightmost': rightmost,
            'width': rightmost[0] - leftmost[0],
            'x_offset': center_x - frame_center_x,  
            'y_position': center_y / image.shape[0],  
            'area': cv2.contourArea(blue_contour),
            'touches_left': touches_left,
            'touches_right': touches_right,
            'touches_border': touches_border
        }
        
        return blue_contour, center, alignment_data

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
            error = (point[0] - ref_point[0]) * self.PX_TO_CM
            errors.append((i, error))
        
        # If we don't have all points, adjust weights
        if len(errors) < len(self.lookahead_weights):
            # Create new weights normalized to sum to 1
            # The key fix - create a mapping from original indices to new weights
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
    
    def run(self):
        cv2.namedWindow("Line Detection", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Bird's Eye View", cv2.WINDOW_NORMAL)
        
        # PID controller is already initialized in __init__
        self.pid_controller.reset()  # Reset PID state before starting
        
        DEFAULT_SPEED = 0.3
        
        try:
            while True:
                ret, frame = self.capture.read()
                if not ret:
                    print("Failed to capture image")
                    break
                
                # Get bird's eye view
                birds_eye = self.get_birds_eye_view(frame)
                
                # Process the original frame for display
                display = frame.copy()
                birds_eye_display = birds_eye.copy() if birds_eye is not None else None
                
                height, width = display.shape[:2]
                center_x = width // 2
                cv2.line(display, (center_x, 0), (center_x, height), (0, 255, 0), 1)
                
                # Process path in bird's eye view
                if birds_eye is not None:
                    binary_mask, path_mask = self.get_path_mask(birds_eye)
                    path_contour, path_points = self.get_path_points_with_lookahead(binary_mask)
                else:
                    binary_mask, path_mask = self.get_path_mask(frame)
                    path_contour, path_points = self.get_path_points_with_lookahead(binary_mask)
                
                # Process danger data from original frame
                danger_contour, danger_center, danger_data = self.get_danger_data(frame)
                
                # Draw reference points on displays
                for loc in self.reference_locs:
                    cv2.circle(display, loc, 6, (255, 0, 0), -1)
                    if birds_eye_display is not None:
                        cv2.circle(birds_eye_display, (loc[0], loc[1]), 6, (255, 0, 0), -1)
                
                # Initialize driving parameters
                left_speed = DEFAULT_SPEED
                right_speed = DEFAULT_SPEED
                correction = 0
                direction = "CENTER"
                
                # Process line following if path is detected
                if path_contour is not None and path_points and any(pt is not None for pt in path_points):
                    # Draw contour on displays
                    cv2.drawContours(display, [path_contour], -1, (0, 0, 255), 2)
                    if birds_eye_display is not None:
                        cv2.drawContours(birds_eye_display, [path_contour], -1, (0, 0, 255), 2)
                    
                    # Draw path points on displays
                    for point in path_points:
                        if point is not None:
                            cv2.circle(display, point, 6, (255, 0, 255), -1)
                            if birds_eye_display is not None:
                                cv2.circle(birds_eye_display, point, 6, (255, 0, 255), -1)
                    
                    # Calculate weighted error from multiple look-ahead points
                    near_error, far_error, weighted_error, all_errors = self.calculate_weighted_error(path_points)
                    
                    # Compute PID correction based on weighted error
                    if weighted_error is not None:
                        correction = self.pid_controller.compute_correction(weighted_error)
                        
                        # Ensure minimum correction if error is significant
                        if abs(weighted_error) > 1.0 and abs(correction) < 0.05:
                            correction = 0.05 * (-1 if weighted_error < 0 else 1)
                        
                        # Apply correction to motor speeds
                        left_speed = DEFAULT_SPEED + correction
                        right_speed = DEFAULT_SPEED - correction
                        
                        # Determine direction of turn
                        direction = "RIGHT" if weighted_error > 0 else "LEFT" if weighted_error < 0 else "CENTER"
                        
                        # Display error information
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
                        
                        cv2.putText(display, f"LEFT Motor: {left_speed:.2f}", 
                                (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                        
                        cv2.putText(display, f"RIGHT Motor: {right_speed:.2f}", 
                                (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                        
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
                else:
                    cv2.putText(display, "NO PATH DETECTED", 
                            (width//2 - 150, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    if birds_eye_display is not None:
                        cv2.putText(birds_eye_display, "NO PATH DETECTED", 
                                (width//2 - 150, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                
                # Process danger detection (same as original code)
                if danger_contour is not None:
                    cv2.drawContours(display, [danger_contour], -1, (255, 0, 0), 2)
                    
                    if danger_center is not None:
                        cv2.circle(display, danger_center, 8, (0, 255, 255), -1)
                        
                        if danger_data:
                            if danger_data.get('touches_border', False):
                                border_text = "EDGE: "
                                if danger_data.get('touches_left', False) and danger_data.get('touches_right', False):
                                    border_text += "BOTH SIDES - MOVE BACK"
                                elif danger_data.get('touches_left', False):
                                    border_text += "LEFT SIDE - TURN RIGHT"
                                elif danger_data.get('touches_right', False):
                                    border_text += "RIGHT SIDE - TURN LEFT"
                                
                                cv2.putText(display, border_text, 
                                        (width - 400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            x_offset = danger_data['x_offset']
                            cv2.putText(display, f"Target X-Offset: {x_offset:.1f}px", 
                                    (width - 400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
                            if abs(x_offset) > 40:
                                turn_direction = "RIGHT" if x_offset > 0 else "LEFT"
                                turn_angle = abs(x_offset) * 0.1
                                cv2.putText(display, f"Turn {turn_direction}: {turn_angle:.1f} deg", 
                                        (width - 400, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                            else:
                                cv2.putText(display, "TARGET ALIGNED", 
                                        (width - 400, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw motor speed bars
                bar_x = 20
                bar_y = 240
                bar_height = 30
                max_bar_width = 200
                
                left_bar_width = int(max_bar_width * (left_speed / 1))
                right_bar_width = int(max_bar_width * (right_speed / 1))
                
                cv2.rectangle(display, (bar_x, bar_y), (bar_x + left_bar_width, bar_y + bar_height), (255, 165, 0), -1)
                cv2.rectangle(display, (bar_x, bar_y), (bar_x + max_bar_width, bar_y + bar_height), (255, 255, 255), 2)
                cv2.putText(display, "L", (bar_x - 15, bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                bar_y += bar_height + 10
                cv2.rectangle(display, (bar_x, bar_y), (bar_x + right_bar_width, bar_y + bar_height), (255, 165, 0), -1)
                cv2.rectangle(display, (bar_x, bar_y), (bar_x + max_bar_width, bar_y + bar_height), (255, 255, 255), 2)
                cv2.putText(display, "R", (bar_x - 15, bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw birds eye view calibration lines
                if birds_eye_display is not None:
                    for row in self.lookahead_rows:
                        cv2.line(birds_eye_display, (0, row), (width, row), (0, 255, 0), 1)
                    
                    # Draw the center line
                    cv2.line(birds_eye_display, (center_x, 0), (center_x, height), (0, 255, 0), 1)
                
                # Show the displays
                cv2.imshow("Line Detection", display)
                if birds_eye_display is not None:
                    cv2.imshow("Bird's Eye View", birds_eye_display)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.05)
            
        finally:
            self.capture.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    vision = EnhancedVisionProcessor()
    vision.run()