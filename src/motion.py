import math 

from models.devices import devices
from models.wheel import Wheel

class MotionController():
    def __init__(self) -> None:
        self.left_motor = devices.wheel_motors[Wheel.LEFT]
        self.right_motor = devices.wheel_motors[Wheel.RIGHT]
        self.wheel_circumference = 2 * math.pi * 2 # 20 mm radius

    def set_motor_speed(self, speed: float):
        self.left_motor.forward(speed)
        self.right_motor.forward(speed)

    def set_reverse_speed(self, speed: float):
        self.left_motor.backward(speed)
        self.right_motor.backward(speed)

    def set_right_turn_speed(self, speed: float):
        self.left_motor.forward(speed)
        self.right_motor.backward(speed)

    def set_left_turn_speed(self, speed: float):
        self.left_motor.backward(speed)
        self.right_motor.forward(speed)
        
    def start(self, speed: float):
        self.set_motor_speed(speed)

    def stop(self):
        self.left_motor.stop()
        self.right_motor.stop()

    def reverse(self, speed: float):
        self.set_reverse_speed(speed)

    def move(self, distance: float):
        rotations_needed = distance / self.wheel_circumference
        current_rotations = 0
        # idk how to do this with gpiozero, maybe a motor encoder issue?
        while True:
            self.left_motor.forward()
            self.right_motor.forward()
            if current_rotations >= rotations_needed:
                break
        self.stop()

    def turn(self, rotation_deg: float, speed: float):
        if rotation_deg > 0:
            self.set_right_turn_speed(speed)
        else:
            self.set_left_turn_speed(speed)
