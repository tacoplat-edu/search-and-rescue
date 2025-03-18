import cv2
import time
import numpy as np

class SimpleVisionProcessor:
    def __init__(self, config_params=None):
        self.capture = cv2.VideoCapture(0)
        
        self.config_params = config_params or {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
        }

        for k, v in self.config_params.items():
            self.capture.set(k, v)

        width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        self.reference_locs = self._create_reference_points(width, height)
        self.PX_TO_CM = 13 / 640
    
    def _create_reference_points(self, width, height):
        y_values = [int(height * 0.8), int(height * 0.65), int(height * 0.5)]
        return [(int(width // 2), y) for y in y_values]
        
    def get_path_mask(self, image):
        if image is None:
            return None
            
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        red1_lower, red1_upper = np.uint8([0, 100, 30]), np.uint8([10, 255, 255])
        red2_lower, red2_upper = np.uint8([160, 100, 30]), np.uint8([180, 255, 255])
        
        mask1 = cv2.inRange(hsv_image, red1_lower, red1_upper)
        mask2 = cv2.inRange(hsv_image, red2_lower, red2_upper)
    
        mask = cv2.bitwise_or(mask1, mask2)
        
        return cv2.bitwise_and(image, image, mask=mask)
    
    def get_path_data(self, mask):
        if mask is None:
            return None, None
            
        grayscale = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(grayscale, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return None, None
            
        primary_contour = max(contours, key=cv2.contourArea)
        locs = []
        
        for ref_loc in self.reference_locs:
            abs_y = ref_loc[1]
            
            tolerance = 2
            contour_points = [pt[0] for pt in primary_contour if abs(pt[0][1] - abs_y) <= tolerance]
            
            if contour_points:
                abs_x = int(np.mean([pt[0] for pt in contour_points]))
                locs.append((abs_x, abs_y))

        return primary_contour, locs if locs else None

    def calculate_error(self, path_locs):
        if not path_locs or len(path_locs) == 0:
            return None, None, None
        
        top_dot_index = len(path_locs) - 1
        
        top_error = (path_locs[top_dot_index][0] - self.reference_locs[top_dot_index][0]) * self.PX_TO_CM
        bottom_error = (path_locs[0][0] - self.reference_locs[0][0]) * self.PX_TO_CM
        
        errors = [(loc[0] - ref[0]) * self.PX_TO_CM for loc, ref in zip(path_locs, self.reference_locs[:len(path_locs)])]
        avg_error = sum(errors) / len(errors)
        
        return top_error, bottom_error, avg_error

    def run(self):
        cv2.namedWindow("Line Detection", cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = self.capture.read()
                if not ret:
                    print("Failed to capture image")
                    break
                
                display = frame.copy()
                
                path_mask = self.get_path_mask(frame)
                path, path_locs = self.get_path_data(path_mask)
                
                for loc in self.reference_locs:
                    cv2.circle(display, loc, 6, (255, 0, 0), -1)
            
                if path is not None:
                    cv2.drawContours(display, [path], -1, (0, 0, 255), 2)
                    
                    if path_locs is not None:
                        for loc in path_locs:
                            cv2.circle(display, loc, 6, (255, 0, 255), -1)
                        
                        top_error, bottom_error, avg_error = self.calculate_error(path_locs)
                        
                        cv2.putText(display, f"Top Error: {top_error:.2f} cm", 
                                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        cv2.putText(display, f"Bottom Error: {bottom_error:.2f} cm", 
                                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        cv2.putText(display, f"Avg Error: {avg_error:.2f} cm", 
                                   (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        top_idx = min(2, len(path_locs)-1)
                        top_ref = self.reference_locs[top_idx]
                        top_path = path_locs[top_idx]
                        cv2.line(display, 
                                (top_ref[0], top_ref[1]), 
                                (top_path[0], top_path[1]), 
                                (0, 255, 255), 2)
                        
                        cv2.line(display, 
                                (self.reference_locs[0][0], self.reference_locs[0][1]), 
                                (path_locs[0][0], path_locs[0][1]), 
                                (255, 255, 0), 2)
                
                if path_mask is not None:
                    small_mask = cv2.resize(path_mask, (160, 120))
                    h, w = small_mask.shape[:2]
                    display[10:10+h, display.shape[1]-10-w:display.shape[1]-10] = small_mask
                
                cv2.imshow("Line Detection", display)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.05)
                
        finally:
            self.capture.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    vision = SimpleVisionProcessor()
    vision.run()