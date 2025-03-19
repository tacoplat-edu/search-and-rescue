from ultralytics import YOLO

# Load your model
model = YOLO("../runs/detect/train/weights/best.pt")

# Export the model
model.export(format="ncnn",   
             imgsz = 640,  
             nms=False,
             half=False) 