import math
import time

from models.devices import DeviceConfiguration
from models.wheel import Wheel


class MotionController:
    devices: DeviceConfiguration

    def __init__(self, devices) -> None:
        self.devices = devices
        self.wheel_circumference = 2 * math.pi * 2  # 20 mm radius

    def set_motor_speed(self, speed: float, wheel: Wheel = Wheel.BOTH):
        if wheel == Wheel.BOTH:
            self.set_motor_speed(speed, Wheel.LEFT)
            self.set_motor_speed(speed, Wheel.RIGHT)
        elif wheel == Wheel.LEFT:
            self.devices.wheel_motors[Wheel.LEFT].forward(speed)
        else:
            self.devices.wheel_motors[Wheel.RIGHT].forward(speed)

    def set_reverse_speed(self, speed: float):
        self.devices.wheel_motors[Wheel.LEFT].backward(speed)
        self.devices.wheel_motors[Wheel.RIGHT].backward(speed)

    def set_right_turn_speed(self, speed: float):
        self.devices.wheel_motors[Wheel.LEFT].forward(speed)
        self.devices.wheel_motors[Wheel.RIGHT].backward(speed)

    def set_left_turn_speed(self, speed: float):
        self.devices.wheel_motors[Wheel.LEFT].backward(speed)
        self.devices.wheel_motors[Wheel.RIGHT].forward(speed)

    def start(self, speed: float):
        self.set_motor_speed(speed)

    def stop(self):
        self.devices.wheel_motors[Wheel.LEFT].stop()
        self.devices.wheel_motors[Wheel.RIGHT].stop()

    def reverse(self, speed: float):
        self.set_reverse_speed(speed)

    def move(self, distance: float):
        rotations_needed = distance / self.wheel_circumference
        current_rotations = 0
        # idk how to do this with gpiozero, maybe a motor encoder issue?
        while True:
            self.devices.wheel_motors[Wheel.LEFT].forward()
            self.devices.wheel_motors[Wheel.RIGHT].forward()
            if current_rotations >= rotations_needed:
                break
        self.stop()

    """
        Speed in deg/s
    """

    def turn(self, rotation_deg: float, speed: float):
        lo, ro = self.devices.wheel_motors[Wheel.LEFT].value, self.devices.wheel_motors[Wheel.RIGHT].value
        normalized_speed = speed * (2*math.pi)/360 * 6 / 62.8
    
        execution_time = abs(rotation_deg) / speed
        if rotation_deg > 0:
            self.set_right_turn_speed(normalized_speed)
        else:
            self.set_left_turn_speed(normalized_speed)
        time.sleep(execution_time)

        self.set_motor_speed(lo, Wheel.LEFT)
        self.set_motor_speed(ro, Wheel.RIGHT)

        return True
