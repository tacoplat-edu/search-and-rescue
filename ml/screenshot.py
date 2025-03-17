import cv2
import os
import datetime
from pathlib import Path

def capture_screenshots():

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    screenshot_dir = Path("new_val")
    screenshot_dir.mkdir(exist_ok=True)
    
    print("Webcam initialized. Press 'q' to take a screenshot, 'ESC' to exit.")
    print(f"Screenshots will be saved to: {screenshot_dir.absolute()}")
    
    screenshot_count = 0
    
    while True:

        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        cv2.imshow('Webcam - Press Q for screenshot, ESC to quit', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}_{screenshot_count}.jpg"
            filepath = screenshot_dir / filename
            
            cv2.imwrite(str(filepath), frame)
            screenshot_count += 1
            print(f"Screenshot saved: {filepath}")
        
        elif key == 27:  
            print("Exiting...")
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_screenshots()