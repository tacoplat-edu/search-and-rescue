import os
import cv2
import numpy as np

def create_label(image_path, output_folder, class_id=0, max_display_size=800):
    img = cv2.imread(image_path)
    original_height, original_width = img.shape[:2]
    
    display_scale = min(max_display_size / original_width, max_display_size / original_height)
    display_width = int(original_width * display_scale)
    display_height = int(original_height * display_scale)
    display_img = cv2.resize(img, (display_width, display_height))
    
    cv2.imshow('Draw bounding box', display_img)
    
    bbox_display = cv2.selectROI('Draw bounding box', display_img, False)
    cv2.destroyAllWindows()
    
    x_display, y_display, w_display, h_display = bbox_display
    x = int(x_display / display_scale)
    y = int(y_display / display_scale)
    w = int(w_display / display_scale)
    h = int(h_display / display_scale)
    
    x_center = (x + w/2) / original_width
    y_center = (y + h/2) / original_height
    norm_width = w / original_width
    norm_height = h / original_height
    
    image_name = os.path.basename(image_path).split('.')[0]
    label_path = os.path.join(output_folder, f"{image_name}.txt")
    
    with open(label_path, 'w') as f:
        f.write(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}")
    
    print(f"Label created for {image_path}: {class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
    
    return (x, y, w, h)

def batch_label_by_pattern(template_bbox, image_folder, output_folder, filename_pattern, class_id=0):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    x, y, w, h = template_bbox
    
    for filename in os.listdir(image_folder):
        if (filename.endswith('.jpg') or filename.endswith('.png')) and filename_pattern in filename:
            image_path = os.path.join(image_folder, filename)
            img = cv2.imread(image_path)
            height, width = img.shape[:2]
            
            x_center = (x + w/2) / width
            y_center = (y + h/2) / height
            norm_width = w / width
            norm_height = h / height
            
            image_name = os.path.basename(image_path).split('.')[0]
            label_path = os.path.join(output_folder, f"{image_name}.txt")
            
            with open(label_path, 'w') as f:
                f.write(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}")
            
            print(f"Label created for {image_path}")

if __name__ == "__main__":
    train_folder = "dataset/images/train"
    labels_folder = "dataset/labels/train"
    
    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)
    
    first_image = "dataset/images/train/000_1.jpg"  

    print("Draw bounding box for the standard variation (no _m in filename)")
    template_bbox1 = create_label(first_image, labels_folder, max_display_size=800)
    
    for filename in os.listdir(train_folder):
        if filename.endswith('.jpg') and "_m" not in filename:
            image_path = os.path.join(train_folder, filename)
            if image_path != first_image: 
                img = cv2.imread(image_path)
                height, width = img.shape[:2]
                
                x, y, w, h = template_bbox1
                x_center = (x + w/2) / width
                y_center = (y + h/2) / height
                norm_width = w / width
                norm_height = h / height
                
                image_name = os.path.basename(image_path).split('.')[0]
                label_path = os.path.join(labels_folder, f"{image_name}.txt")
                
                with open(label_path, 'w') as f:
                    f.write(f"0 {x_center} {y_center} {norm_width} {norm_height}")
                
                print(f"Label created for {image_path}")
    
    second_image = "dataset/images/train/000_1_m.jpg" 

    print("Draw bounding box for the modified variation (_m in filename)")
    template_bbox2 = create_label(second_image, labels_folder, max_display_size=800)
    
    for filename in os.listdir(train_folder):
        if filename.endswith('.jpg') and "_m" in filename:
            image_path = os.path.join(train_folder, filename)
            if image_path != second_image:  
                img = cv2.imread(image_path)
                height, width = img.shape[:2]
                
                x, y, w, h = template_bbox2
                x_center = (x + w/2) / width
                y_center = (y + h/2) / height
                norm_width = w / width
                norm_height = h / height
                
                image_name = os.path.basename(image_path).split('.')[0]
                label_path = os.path.join(labels_folder, f"{image_name}.txt")
                
                with open(label_path, 'w') as f:
                    f.write(f"0 {x_center} {y_center} {norm_width} {norm_height}")
                
                print(f"Label created for {image_path}")
    
    val_images_folder = "dataset/images/val" 
    val_labels_folder = "dataset/labels/val"
    
    if not os.path.exists(val_labels_folder):
        os.makedirs(val_labels_folder)
    
    print("Now labeling validation images one by one...")
    for filename in os.listdir(val_images_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(val_images_folder, filename)
            create_label(image_path, val_labels_folder, max_display_size=800)
    
    print("All images have been labeled!")