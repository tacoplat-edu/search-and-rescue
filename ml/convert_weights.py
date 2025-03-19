from ultralytics import YOLO

# Load your model
model = YOLO("../runs/detect/train/weights/best.pt")

# Export the model
model.export(format="openvino",   
             imgsz = 640,
             simplify=False,     
             dynamic=False,
             nms=False,
             half=False) 