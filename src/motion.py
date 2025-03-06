import math
import time
import os

from models.devices import DeviceConfiguration
from models.wheel import Wheel


DEBUG = os.environ.get("DEBUG") == "true"
MAX_SPEED = 62.8 # [cm/s]

class MotionController:
    devices: DeviceConfiguration
    wheel_circumference: float
    default_speed_ratio: float

    def __init__(self, devices) -> None:
        self.devices = devices
        self.wheel_circumference = 2 * math.pi * 2  # 20 mm radius -> 4pi cm circumference
        self.default_speed_ratio = 0.45

    def set_forward_speed(self, speed: float, wheel: Wheel = Wheel.BOTH):
        assert speed >= 0 and speed <= 1

        if wheel == Wheel.BOTH:
            self.set_forward_speed(speed, Wheel.LEFT)
            self.set_forward_speed(speed, Wheel.RIGHT)
        elif wheel == Wheel.LEFT:
            self.devices.wheel_motors[Wheel.LEFT].forward(speed)
        else:
            self.devices.wheel_motors[Wheel.RIGHT].forward(speed)

    def set_reverse_speed(self, speed: float, wheel: Wheel = Wheel.BOTH):
        assert speed >= 0 and speed <= 1

        if wheel == Wheel.BOTH:
            self.set_reverse_speed(speed, Wheel.LEFT)
            self.set_reverse_speed(speed, Wheel.RIGHT)
        elif wheel == Wheel.LEFT:
            self.devices.wheel_motors[Wheel.LEFT].backward(speed)
        else:
            self.devices.wheel_motors[Wheel.RIGHT].backward(speed)

    def set_right_turn_speed(self, speed: float):
        assert speed >= 0 and speed <= 1
        self.devices.wheel_motors[Wheel.LEFT].forward(speed)
        self.devices.wheel_motors[Wheel.RIGHT].backward(speed)

    def set_left_turn_speed(self, speed: float):
        assert speed >= 0 and speed <= 11
        self.devices.wheel_motors[Wheel.LEFT].backward(speed)
        self.devices.wheel_motors[Wheel.RIGHT].forward(speed)

    def start(self, speed: float):
        assert speed >= 0 and speed <= 1
        self.set_forward_speed(speed)

    def stop(self):
        self.devices.wheel_motors[Wheel.LEFT].stop()
        self.devices.wheel_motors[Wheel.RIGHT].stop()

    def reverse(self, speed: float):
        self.set_reverse_speed(speed)


    """
        Distance in cm, Speed in cm/s
    """
    def move(self, distance: float, speed: float):
        rotations_needed = math.ceil(abs(distance) / self.wheel_circumference)
        current_rotations = self.devices.wheel_encoders[Wheel.LEFT].steps = 0

        normalized_speed = abs(speed) / MAX_SPEED
        assert normalized_speed > 0 and normalized_speed <= 1

        if distance >= 0:
            self.set_forward_speed(normalized_speed)
        else:
            self.set_reverse_speed(normalized_speed)

        print("lwv move", self.devices.wheel_motors[Wheel.LEFT].value)

        while current_rotations < rotations_needed:
            if DEBUG:
                time.sleep(abs(distance/speed) / rotations_needed)
                self.devices.wheel_encoders[Wheel.LEFT].steps += 1
            current_rotations = self.devices.wheel_encoders[Wheel.LEFT].steps
            
        self.stop()
        return True

    """
        Speed in deg/s
    """
    def turn(self, rotation_deg: float, angular_speed: float):
        normalized_speed = (angular_speed * (2*math.pi)/360 * 6) / MAX_SPEED
    
        execution_time = abs(rotation_deg) / angular_speed
        if rotation_deg > 0:
            self.set_right_turn_speed(normalized_speed)
        else:
            self.set_left_turn_speed(normalized_speed)

        print("lwv turn", self.devices.wheel_motors[Wheel.LEFT].value)

        time.sleep(execution_time)

        self.stop()
        return True
