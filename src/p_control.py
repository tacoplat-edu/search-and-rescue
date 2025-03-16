import time

class PController:
    def __init__(self, kp: float, scale_factor: float):
        self.kp = kp
        self.scale_factor = scale_factor

    def compute_correction(self, error):
        scaled_error = error * self.scale_factor
        correction = self.kp * scaled_error

        return correction
    