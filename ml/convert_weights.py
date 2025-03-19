from ultralytics import YOLO

# Load your model
model = YOLO("../runs/detect/train/weights/best.pt")

# Export the model
model.export(format="onnx",
             opset=12,        
             imgsz = 640,
             simplify=True,     
             dynamic=False,
             do_constant_folding=False,  
             nms=False,
             half=False) 