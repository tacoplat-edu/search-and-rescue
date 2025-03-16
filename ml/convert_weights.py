from ultralytics import YOLO

# Load your model
model = YOLO("../runs/detect/train5/weights/best.pt")

# Export the model
model.export(format="onnx")