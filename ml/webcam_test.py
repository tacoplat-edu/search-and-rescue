from ultralytics import YOLO
import cv2
import time
import os

def run_webcam_detection():
    model_path = "../runs/detect/train/weights/best_ncnn_model"
    
    if not os.path.exists(model_path):
        print(f"Error: NCNN model directory not found at: {model_path}")
        return
    

    model = YOLO(model_path)
    print("Model loaded successfully!")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Failed to open webcam")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam: {width}x{height}")
    
    purple_color = (255, 0, 255)
    conf_threshold = 0.4
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        try:
            start_time = time.time()
            results = model(frame, conf=conf_threshold)
            inference_time = time.time() - start_time
            fps = 1.0 / inference_time
            
            result_frame = frame.copy()
            
            for r in results:
                boxes = r.boxes
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
 
                    conf = float(box.conf[0])

                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), purple_color, 2)
                    
                    conf_percentage = f"{int(conf * 100)}%"
                    text_size = cv2.getTextSize(conf_percentage, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    
                    cv2.rectangle(result_frame, 
                                (x1, y1 - text_size[1] - 10), 
                                (x1 + text_size[0], y1), 
                                purple_color, -1)
                    
                    cv2.putText(result_frame, 
                            conf_percentage, 
                            (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (255, 255, 255), 2)
            
            cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("NCNN Detection", result_frame)
            
        except Exception as e:
            print(f"Error during detection: {e}")
            import traceback
            traceback.print_exc()
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print(f"OpenCV version: {cv2.__version__}")
    run_webcam_detection()