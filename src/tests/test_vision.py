import cv2
import time
import numpy as np
from p_control import PController 
class SimpleVisionProcessor:
    def __init__(self, config_params=None):
        self.capture = cv2.VideoCapture(0)
        
        self.config_params = config_params or {
            cv2.CAP_PROP_FRAME_WIDTH: 1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720,
        }
        self.default_motor_speed = 0.3

        for k, v in self.config_params.items():
            self.capture.set(k, v)

        width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        self.reference_locs = self._create_reference_points(width, height)
        self.PX_TO_CM = 13 / 1280
    
    def _create_reference_points(self, width, height):
        print(f"Camera width: {width}, height: {height}")
        
        center_x = int(width // 2)
        
        y_values = [int(height * 0.8), int(height * 0.65), int(height * 0.5)]
        
        reference_points = [(center_x, y) for y in y_values]
        
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
        if image is None:
            return None
            
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        red1_lower, red1_upper = np.uint8([0, 100, 30]), np.uint8([10, 255, 255])
        red2_lower, red2_upper = np.uint8([160, 100, 30]), np.uint8([180, 255, 255])
        
        mask1 = cv2.inRange(hsv_image, red1_lower, red1_upper)
        mask2 = cv2.inRange(hsv_image, red2_lower, red2_upper)
    
        mask = cv2.bitwise_or(mask1, mask2)
        
        return cv2.bitwise_and(image, image, mask=mask)
    
    def get_path_data(self, mask):
        if mask is None:
            return None, None
            
        grayscale = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(grayscale, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return None, None
            
        primary_contour = max(contours, key=cv2.contourArea)
        locs = []
        
        for ref_loc in self.reference_locs:
            abs_y = ref_loc[1]
            
            tolerance = 2
            contour_points = [pt[0] for pt in primary_contour if abs(pt[0][1] - abs_y) <= tolerance]
            
            if contour_points:
                abs_x = int(np.mean([pt[0] for pt in contour_points]))
                locs.append((abs_x, abs_y))

        return primary_contour, locs if locs else None

    def calculate_error(self, path_locs):
        if not path_locs or len(path_locs) == 0:
            return None, None, None
        
        top_dot_index = len(path_locs) - 1
        
        top_error = (path_locs[top_dot_index][0] - self.reference_locs[top_dot_index][0]) * self.PX_TO_CM
        bottom_error = (path_locs[0][0] - self.reference_locs[0][0]) * self.PX_TO_CM
        
        errors = [(loc[0] - ref[0]) * self.PX_TO_CM for loc, ref in zip(path_locs, self.reference_locs[:len(path_locs)])]
        avg_error = sum(errors) / len(errors)
        
        return top_error, bottom_error, avg_error
    
    def run(self):
        cv2.namedWindow("Line Detection", cv2.WINDOW_NORMAL)

        p_controller = PController(kp=0.75, scale_factor=0.01)
        
        DEFAULT_SPEED = 0.3
        MIN_SPEED = 0.1
        MAX_SPEED = 0.45
        MAX_CORRECTION = 0.1
        
        try:
            while True:
                ret, frame = self.capture.read()
                if not ret:
                    print("Failed to capture image")
                    break
                
                display = frame.copy()
                
                height, width = display.shape[:2]
                center_x = width // 2
                cv2.line(display, (center_x, 0), (center_x, height), (0, 255, 0), 1)
                
                path_mask = self.get_path_mask(frame)
                path, path_locs = self.get_path_data(path_mask)
                
                danger_contour, danger_center, danger_data = self.get_danger_data(frame)
                
                for loc in self.reference_locs:
                    cv2.circle(display, loc, 6, (255, 0, 0), -1)
                
                left_speed = DEFAULT_SPEED
                right_speed = DEFAULT_SPEED
                correction = 0
                direction = "CENTER"
                
                if path is not None:
                    cv2.drawContours(display, [path], -1, (0, 0, 255), 2)
                    
                    if path_locs is not None:
                        for loc in path_locs:
                            cv2.circle(display, loc, 6, (255, 0, 255), -1)
                        
                        top_error, bottom_error, avg_error = self.calculate_error(path_locs)
                        
                        if top_error is not None:
                            correction = p_controller.compute_correction(top_error)
                            
                            if abs(top_error) > 1.0 and abs(correction) < 0.05:
                                correction = 0.05 * (-1 if top_error < 0 else 1)
                            
                            correction = max(-MAX_CORRECTION, min(MAX_CORRECTION, correction))
                            
                            left_speed = max(MIN_SPEED, min(MAX_SPEED, DEFAULT_SPEED + correction))
                            right_speed = max(MIN_SPEED, min(MAX_SPEED, DEFAULT_SPEED - correction))
                            
                            direction = "RIGHT" if top_error > 0 else "LEFT" if top_error < 0 else "CENTER"
                        
                        cv2.putText(display, f"Top Error: {top_error:.2f} cm", 
                                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        cv2.putText(display, f"Bottom Error: {bottom_error:.2f} cm", 
                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        cv2.putText(display, f"Avg Error: {avg_error:.2f} cm", 
                                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        cv2.putText(display, f"Correction: {correction:.3f}", 
                                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        
                        cv2.putText(display, f"Turn: {direction}", 
                                (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        
                        cv2.putText(display, f"LEFT Motor: {left_speed:.2f}", 
                                (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                        
                        cv2.putText(display, f"RIGHT Motor: {right_speed:.2f}", 
                                (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                        
                        # Draw error lines
                        top_idx = min(2, len(path_locs)-1)
                        top_ref = self.reference_locs[top_idx]
                        top_path = path_locs[top_idx]
                        cv2.line(display, 
                                (top_ref[0], top_ref[1]), 
                                (top_path[0], top_path[1]), 
                                (0, 255, 255), 2)
                        
                        cv2.line(display, 
                                (self.reference_locs[0][0], self.reference_locs[0][1]), 
                                (path_locs[0][0], path_locs[0][1]), 
                                (255, 255, 0), 2)
                else:
                    cv2.putText(display, "NO PATH DETECTED", 
                            (width//2 - 150, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                
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
                
                bar_x = 20
                bar_y = 240
                bar_height = 30
                max_bar_width = 200
                
                left_bar_width = int(max_bar_width * (left_speed / MAX_SPEED))
                right_bar_width = int(max_bar_width * (right_speed / MAX_SPEED))
                
                cv2.rectangle(display, (bar_x, bar_y), (bar_x + left_bar_width, bar_y + bar_height), (255, 165, 0), -1)
                cv2.rectangle(display, (bar_x, bar_y), (bar_x + max_bar_width, bar_y + bar_height), (255, 255, 255), 2)
                cv2.putText(display, "L", (bar_x - 15, bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                bar_y += bar_height + 10
                cv2.rectangle(display, (bar_x, bar_y), (bar_x + right_bar_width, bar_y + bar_height), (255, 165, 0), -1)
                cv2.rectangle(display, (bar_x, bar_y), (bar_x + max_bar_width, bar_y + bar_height), (255, 255, 255), 2)
                cv2.putText(display, "R", (bar_x - 15, bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow("Line Detection", display)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.05)
            
        finally:
            self.capture.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    vision = SimpleVisionProcessor()
    vision.run()