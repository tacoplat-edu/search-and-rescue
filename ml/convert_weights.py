from ultralytics import YOLO

# Load your model
model = YOLO("../runs/detect/train/weights/best.pt")

# Export the model
model.export(format="onnx",
             opset=11,        
             simplify=True,     
             dynamic=False,  
             half=False) 