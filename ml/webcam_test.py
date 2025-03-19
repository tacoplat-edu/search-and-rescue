import cv2
import numpy as np

def run_webcam_detection():
    model_xml = "../runs/detect/train/weights/best_openvino_model/model.xml"
    model_bin = "../runs/detect/train/weights/best_openvino_model/yolo11n_openvino_model/model.bin"
    net = cv2.dnn.readNet(model_xml, model_bin)
    
    print(f"OpenCV version: {cv2.__version__}")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam: {frame_width}x{frame_height}")
    
    input_size = 640 
    conf_threshold = 0.4
    purple_color = (255, 0, 255)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        blob = cv2.dnn.blobFromImage(
            frame, 
            scalefactor=1/255.0,  
            size=(input_size, input_size),  
            mean=(0, 0, 0),
            swapRB=True,  
            crop=False
        )
        
        net.setInput(blob)
        
        try:
            outputs = net.forward()
            
            result_frame = frame.copy()
            output = outputs[0]
            print(f"Output shape: {output.shape}")

            if output.shape[0] == 1 and (output.shape[1] == 5 or output.shape[2] == 5):
                output = np.transpose(output, (2, 1)) if output.shape[2] == 5 else output
                x_scale = frame_width / input_size
                y_scale = frame_height / input_size
                
                boxes = []
                scores = []
                
                for detection in output:
                    confidence = float(detection[4])
                    if confidence >= conf_threshold:
                        cx, cy, w, h = detection[0:4]
                        x = int((cx - w/2) * x_scale)
                        y = int((cy - h/2) * y_scale)
                        w_box = int(w * x_scale)
                        h_box = int(h * y_scale)
                        boxes.append([x, y, w_box, h_box])
                        scores.append(confidence)
                
                if boxes:
                    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, 0.5)
                    for i in indices:
            
                        if isinstance(i, (list, tuple, np.ndarray)):
                            i = i[0]
                        x, y, w_box, h_box = boxes[i]
                        x = max(0, x)
                        y = max(0, y)
                        w_box = min(w_box, frame_width - x)
                        h_box = min(h_box, frame_height - y)
                        cv2.rectangle(result_frame, (x, y), (x + w_box, y + h_box), purple_color, 2)
                        label = f"{int(scores[i] * 100)}%"
                        cv2.putText(result_frame, label, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, purple_color, 2)
            
            cv2.imshow("Detection", result_frame)
            
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
    run_webcam_detection()
