import cv2
import time
import numpy as np
import os
from models.wheel import Wheel
from models.devices import DeviceConfiguration
from helpers.motion import MotionController

class LineFollowingRobot:
    def __init__(self, devices):
        # Initialize camera
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Use lower resolution for better performance
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        
        # Verify actual camera resolution
        ret, test_frame = self.capture.read()
        if ret:
            actual_height, actual_width = test_frame.shape[:2]
            print(f"Actual frame dimensions: {actual_width}x{actual_height}")
        
        # Create reference points
        width = actual_width if ret else self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = actual_height if ret else self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.center_x = int(width // 2)
        
        # Initialize motion controller
        self.motion = MotionController(devices)
        
        # Set speed thresholds
        self.DEFAULT_SPEED = 0.3
        self.TURN_SPEED = 0.4
        self.MIN_SPEED = 0.15
        
        # Set decision thresholds (based on actual frame width)
        self.LEFT_THRESHOLD = int(width * 0.6)    # Turn left if center_x > this value
        self.RIGHT_THRESHOLD = int(width * 0.4)   # Turn right if center_x < this value
        
    def get_red_mask(self, image):
        """Detect red line"""
        if image is None:
            return None
            
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Red in HSV is split across two ranges
        red1_lower, red1_upper = np.uint8([0, 100, 30]), np.uint8([10, 255, 255])
        red2_lower, red2_upper = np.uint8([160, 100, 30]), np.uint8([180, 255, 255])
        
        mask1 = cv2.inRange(hsv_image, red1_lower, red1_upper)
        mask2 = cv2.inRange(hsv_image, red2_lower, red2_upper)
        
        mask = cv2.bitwise_or(mask1, mask2)
        return mask
    
    def run(self):
        try:
            while True:
                ret, frame = self.capture.read()
                if not ret:
                    print("Failed to capture image")
                    break
                
                # Get red mask
                mask = self.get_red_mask(frame)
                
                # Find contours in the red mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw center line for reference
                cv2.line(frame, (self.center_x, 0), (self.center_x, frame.shape[0]), (0, 255, 0), 1)
                
                if contours:
                    # Find the largest contour (the red line)
                    c = max(contours, key=cv2.contourArea)
                    
                    # Calculate center of the line
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        
                        # Draw the center of the line
                        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
                        
                        # Calculate error (how far line is from center)
                        error = cx - self.center_x
                        
                        # Display error information
                        direction = "RIGHT" if error > 0 else "LEFT" if error < 0 else "CENTER"
                        cv2.putText(frame, f"Error: {error} ({direction})", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                        # Decision making based on line position
                        if cx >= self.LEFT_THRESHOLD:
                            print(f"Turn Left (CX: {cx}, Error: {error})")
                            
                            # Using your motion controller for left turn
                            left_speed = self.MIN_SPEED
                            right_speed = self.TURN_SPEED
                            
                            self.motion.set_forward_speed(left_speed, Wheel.LEFT)
                            self.motion.set_forward_speed(right_speed, Wheel.RIGHT)
                            
                        elif cx <= self.RIGHT_THRESHOLD:
                            print(f"Turn Right (CX: {cx}, Error: {error})")
                            
                            # Using your motion controller for right turn
                            left_speed = self.TURN_SPEED
                            right_speed = self.MIN_SPEED
                            
                            self.motion.set_forward_speed(left_speed, Wheel.LEFT)
                            self.motion.set_forward_speed(right_speed, Wheel.RIGHT)
                            
                        else:
                            print(f"On Track! (CX: {cx}, Error: {error})")
                            
                            # Go straight at default speed
                            self.motion.set_forward_speed(self.DEFAULT_SPEED)
                        
                        # Display motor speeds
                        cv2.putText(frame, f"LEFT Motor: {left_speed if 'left_speed' in locals() else self.DEFAULT_SPEED:.2f}", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
                        cv2.putText(frame, f"RIGHT Motor: {right_speed if 'right_speed' in locals() else self.DEFAULT_SPEED:.2f}", 
                                  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
                        
                        # Draw the contour
                        cv2.drawContours(frame, [c], -1, (0, 255, 0), 1)
                else:
                    print("I don't see the line")
                    self.motion.stop()
                    
                    cv2.putText(frame, "NO LINE DETECTED", 
                              (frame.shape[1]//2 - 80, frame.shape[0]//2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display the frame and mask
                cv2.imshow("Frame", frame)
                cv2.imshow("Mask", mask)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.05)  # Small delay to control loop speed
                
        finally:
            # Clean up
            self.motion.stop()
            self.capture.release()
            cv2.destroyAllWindows()

# Main entry point
if __name__ == "__main__":
    # Create device configuration
    devices = DeviceConfiguration()
    devices.setup()
    
    # Create and run the robot
    robot = LineFollowingRobot(devices)
    robot.run()