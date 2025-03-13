import time

class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, scale_factor: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.scale_factor = scale_factor

        self.prev_error = 0
        self.prev_time = time.time()
        self.integral = 0

    def compute_correction(self, error, dt):
        scaled_error = error * self.scale_factor

        p = self.kp * scaled_error
        
        self.integral += scaled_error * dt
        i = self.ki * self.integral

        derivative = (scaled_error - self.prev_error) / dt
        d = self.kd * derivative

        self.prev_error = scaled_error

        correction = p + i + d
        return correction
    
    def update_prev_time(self, new_time: float):
        self.prev_time = new_time