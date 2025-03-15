import cv2
from ultralytics import YOLO
import time

def run_webcam_detection():
    model = YOLO("/runs/detect/train5/weights/best.pt")  
    
    cap = cv2.VideoCapture(0) 
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Webcam: {width}x{height} @")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        results = model(frame, conf=0.4)
        
        annotated_frame = results[0].plot()
        cv2.imshow("LEGO Figure Detection", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_detection()