import time

class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, scale_factor: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.scale_factor = scale_factor

        self.prev_error = 0.0
        #self.integral =0.0
        self.last_time = time.time()

        self.integral_limit = 10.0

    def reset(self):
        self.prev_error = 0.0
       # self.integral = 0.0
        self.last_time = time.time()

    def compute_correction(self, error):
        curr_time = time.time()
        dt = curr_time - self.last_time

        if dt< 0.001:
            dt = 0.001

        scaled_error = error * self.scale_factor
        p = self.kp * scaled_error
        #self.integral += scaled_error * dt

        #elf.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        #i = self.ki * self.integral 

        d = self.kd * ((scaled_error - self.prev_error) / dt)

        self.prev_error = scaled_error
        self.last_time = curr_time

        correction = p + d
        return correction
    