import os
import cv2

input_dir = "modified_images"
output_dir = "resized_modified_images"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (640, 480))
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, resized_image)
