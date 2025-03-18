import cv2
import numpy as np

def run_webcam_detection():
    model_path = "../runs/detect/train/weights/best.onnx"
    net = cv2.dnn.readNetFromONNX(model_path)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam: {width}x{height}")
    
    input_size = 640  
    
    purple_color = (255, 0, 255)  
    
    conf_threshold = 0.4
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        blob = cv2.dnn.blobFromImage(
            frame, 
            1/255.0,  
            (input_size, input_size),  
            swapRB=True,  
            crop=False
        )
        
        net.setInput(blob)
        
        outputs = net.forward(net.getUnconnectedOutLayersNames())
        
        result_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        
        output = outputs[0]
        
        if len(output.shape) == 3:
            output = np.transpose(output, (0, 2, 1))
        
        x_scale = frame_width / input_size
        y_scale = frame_height / input_size
        
        rows = output.shape[1]
        boxes = []
        scores = []
        
        for i in range(rows):
            row = output[0][i]
            confidence = row[4]
            
            if confidence >= conf_threshold:
                cx, cy, w, h = row[0:4]
                
                x = int((cx - w/2) * x_scale)
                y = int((cy - h/2) * y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                
                box = np.array([x, y, width, height])
                boxes.append(box)
                scores.append(float(confidence))
        
        if boxes:  
            indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, 0.75)
            
            for i in indices:
                if isinstance(i, np.ndarray):
                    i = i[0]
                
                box = boxes[i]
                confidence = scores[i]
                x, y, w, h = box
                
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), purple_color, 2)
                
                conf_percentage = f"{int(confidence * 100)}%"
                
                text_size = cv2.getTextSize(conf_percentage, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
                cv2.rectangle(result_frame, 
                             (x, y - text_size[1] - 10), 
                             (x + text_size[0], y), 
                             purple_color, -1)
                
                cv2.putText(result_frame, 
                           conf_percentage, 
                           (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
        
        cv2.imshow("LEGO Figure Detection", result_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_detection()