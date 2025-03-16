import time

class PController:
    def __init__(self, kp: float, scale_factor: float):
        self.kp = kp
        self.scale_factor = scale_factor
        self.prev_time = time.time()

    def compute_correction(self, error):
        scaled_error = error * self.scale_factor
        correction = self.kp * scaled_error

        return correction
    
    def update_prev_time(self, new_time: float):
        self.prev_time = new_time