import time

class PIDController:
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.prev_error = 0
        self.prev_time = time.time()
        self.integral = 0

    def compute_correction(self, error, dt):
        p = self.kp * error
        
        self.integral += error * dt
        i = self.ki * self.integral

        derivative = (error - self.prev_error) / dt
        d = self.kd * derivative

        self.prev_error = error

        correction = p + i + d
        return correction
    
    def update_prev_time(self, new_time: float):
        self.prev_time = new_time