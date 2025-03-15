import cv2
import numpy as np
import os

def rgb_to_hsv(r, g, b):
    rgb = np.uint8([[[r, g, b]]])
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    return hsv[0][0]

def apply_background(foreground, background, lower_green, upper_green):
    hsv = cv2.cvtColor(foreground, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    mask_inv = cv2.bitwise_not(mask)
    
    fg = cv2.bitwise_and(foreground, foreground, mask=mask_inv)

    background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))
    
    bg = cv2.bitwise_and(background, background, mask=mask)
    
    combined = cv2.add(fg, bg)
    
    return combined

def process_images(foreground_folder, background_folder, output_folder, lower_green, upper_green):
    foreground_images = [f for f in os.listdir(foreground_folder) if f.endswith('.jpg')]
    
    background_images = [f for f in os.listdir(background_folder) if f.endswith('.jpg')]
    
    os.makedirs(output_folder, exist_ok=True)
    
    for fg_image_name in foreground_images:
        fg_image_path = os.path.join(foreground_folder, fg_image_name)
        foreground = cv2.imread(fg_image_path)
        
        for bg_image_name in background_images:
            bg_image_path = os.path.join(background_folder, bg_image_name)
            background = cv2.imread(bg_image_path)
            
            combined_image = apply_background(foreground, background, lower_green, upper_green)
            
            output_image_name = f"{os.path.splitext(fg_image_name)[0]}_{os.path.splitext(bg_image_name)[0]}.jpg"
            output_image_path = os.path.join(output_folder, output_image_name)
            cv2.imwrite(output_image_path, combined_image)

def print_top_left_pixel_color(image_folder):
    images = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    
    image_path = os.path.join(image_folder, images[0])
    image = cv2.imread(image_path)
    
    top_left_pixel = image[0, 0]
    print(f"Top-left pixel color (RGB): {top_left_pixel}")

if __name__ == "__main__":
    raw_images_folder = "raw_images"
    modified_images_folder = "modified_images"
    background_images_folder = "background_images"
    output_folder = "train"
    output_folder_modified = "train_modified"
    
    lower_green_raw = np.array([35, 40, 40])
    upper_green_raw = np.array([90, 255, 255])
    
    lower_green_rgb = (3, 244, 5)
    upper_green_rgb = (3, 244, 5)
    
    lower_green_modified = rgb_to_hsv(*lower_green_rgb) - np.array([10, 50, 50])
    upper_green_modified = rgb_to_hsv(*upper_green_rgb) + np.array([10, 50, 50])
    
    process_images(raw_images_folder, background_images_folder, output_folder, lower_green_raw, upper_green_raw)
    process_images(modified_images_folder, background_images_folder, output_folder_modified, lower_green_modified, upper_green_modified)
