import os
import cv2
import time
import numpy as np
from signal import pause

# Set environment variables first
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
from dotenv import load_dotenv
load_dotenv()

# Import your existing classes
from motion import MotionController
from models.wheel import Wheel
from models.devices import devices

class LineFollowingRobot:
    def __init__(self, devices):
        # Initialize camera
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
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
        self.DEFAULT_SPEED = 0.3     # Normal forward speed
        self.MAX_SPEED = 0.4         # Maximum allowable speed
        self.TURN_SPEED = 0.35       # Speed for the faster wheel in turns
        self.MIN_SPEED = 0.15        # Speed for the slower wheel in turns
        
        # Set decision thresholds (based on actual frame width)
        self.LEFT_THRESHOLD = int(width * 0.6)    # Turn left if center_x > this value
        self.RIGHT_THRESHOLD = int(width * 0.4)   # Turn right if center_x < this value
        
        # Control flag
        self.running = False
        
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
        self.running = True
        print("Starting line following robot...")
        
        try:
            while self.running:
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
                            right_speed = min(self.TURN_SPEED, self.MAX_SPEED)  # Ensure we don't exceed MAX_SPEED
                            
                            self.motion.set_forward_speed(left_speed, Wheel.LEFT)
                            self.motion.set_forward_speed(right_speed, Wheel.RIGHT)
                            
                        elif cx <= self.RIGHT_THRESHOLD:
                            print(f"Turn Right (CX: {cx}, Error: {error})")
                            
                            # Using your motion controller for right turn
                            left_speed = min(self.TURN_SPEED, self.MAX_SPEED)  # Ensure we don't exceed MAX_SPEED
                            right_speed = self.MIN_SPEED
                            
                            self.motion.set_forward_speed(left_speed, Wheel.LEFT)
                            self.motion.set_forward_speed(right_speed, Wheel.RIGHT)
                            
                        else:
                            print(f"On Track! (CX: {cx}, Error: {error})")
                            
                            # Go straight at default speed, never exceeding MAX_SPEED
                            straight_speed = min(self.DEFAULT_SPEED, self.MAX_SPEED)
                            self.motion.set_forward_speed(straight_speed)
                            left_speed = right_speed = straight_speed
                        
                        # Display motor speeds and limits
                        cv2.putText(frame, f"MAX SPEED: {self.MAX_SPEED:.2f}", 
                                  (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(frame, f"LEFT Motor: {left_speed:.2f}", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
                        cv2.putText(frame, f"RIGHT Motor: {right_speed:.2f}", 
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
                
                # Add instruction to quit
                cv2.putText(frame, "Press 'q' to stop", 
                          (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Q key pressed - stopping motors")
                    self.motion.stop()  # Explicitly stop motors
                    self.running = False
                    break
                
                time.sleep(0.05)  # Small delay to control loop speed
                
        except KeyboardInterrupt:
            print("Keyboard interrupt - stopping robot")
            self.running = False
        finally:
            self.motion.stop()
            self.capture.release()
            cv2.destroyAllWindows()
            print("Line following robot stopped")

    def stop(self):
        self.running = False
        self.motion.stop()
        print("Line following robot stopped")

# Initialize the robot
lf = LineFollowingRobot(devices)

# Setup button handler
button = devices.action_button

def press_handler():
    print("Button pressed - Starting line follower")
    lf.run() 

button.when_pressed = press_handler

print("Press the button to start the line follower")
pause()