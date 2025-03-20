import math
import time
import os

from models.devices import DeviceConfiguration
from models.wheel import Wheel
from helpers.motion import assert_speed

DEBUG = os.environ.get("DEBUG") == "true"

MAX_SPEED = 109.96 # [cm/s]
PULSES_PER_REVOLUTION = 420
GEAR_RATIO = 30

LOOP_DELAY_S = 0.01

class MotionController:
    devices: DeviceConfiguration
    wheel_circumference: float # cm
    wheel_diameter: float # cm
    wheel_distance: float # cm
    default_speed: float # power ratio

    def __init__(self, devices) -> None:
        self.devices = devices
        self.wheel_diameter = 7
        self.wheel_distance = 18.25
        self.wheel_circumference = 2 * math.pi * self.wheel_diameter/2  
        self.default_speed = 0.32

    def reset_encoders(self):
        self.devices.wheel_encoders[Wheel.LEFT].steps = 0
        self.devices.wheel_encoders[Wheel.RIGHT].steps = 0

    @assert_speed
    def set_forward_speed(self, speed: float, wheel: Wheel = Wheel.BOTH):
        if wheel == Wheel.BOTH:
            self.set_forward_speed(speed, Wheel.LEFT)
            self.set_forward_speed(speed, Wheel.RIGHT)
        elif wheel == Wheel.LEFT:
            self.devices.wheel_motors[Wheel.LEFT].forward(speed)
        else:
            self.devices.wheel_motors[Wheel.RIGHT].forward(speed)

    @assert_speed
    def set_reverse_speed(self, speed: float, wheel: Wheel = Wheel.BOTH):
        if wheel == Wheel.BOTH:
            self.set_reverse_speed(speed, Wheel.LEFT)
            self.set_reverse_speed(speed, Wheel.RIGHT)
        elif wheel == Wheel.LEFT:
            self.devices.wheel_motors[Wheel.LEFT].backward(speed)
        else:
            self.devices.wheel_motors[Wheel.RIGHT].backward(speed)

    @assert_speed
    def set_right_turn_speed(self, speed: float):
        self.devices.wheel_motors[Wheel.LEFT].forward(speed)
        self.devices.wheel_motors[Wheel.RIGHT].backward(speed)

    @assert_speed
    def set_left_turn_speed(self, speed: float):
        self.devices.wheel_motors[Wheel.LEFT].backward(speed)
        self.devices.wheel_motors[Wheel.RIGHT].forward(speed)

    @assert_speed
    def start(self, speed: float):
        self.set_forward_speed(speed)

    def stop(self):
        self.devices.wheel_motors[Wheel.LEFT].stop()
        self.devices.wheel_motors[Wheel.RIGHT].stop()

    def reverse(self, speed: float):
        self.set_reverse_speed(speed)

    def wait_for_action(self, target_steps, ease_func):
        try:
            while True:
                current_steps = max(
                    abs(self.devices.wheel_encoders[Wheel.LEFT].steps),
                    abs(self.devices.wheel_encoders[Wheel.RIGHT].steps)
                )

                if current_steps >= target_steps:
                    break

                if current_steps > target_steps * 0.8:
                    remaining_factor = (target_steps - current_steps) / (target_steps * 0.2)
                    ease_func(remaining_factor)

                time.sleep(LOOP_DELAY_S)
        except KeyboardInterrupt:
            self.stop()
            return False
        self.stop()
        return True

    def move(self, distance: float, speed: float):
        """
        Parameters
        ----------
        distance : float
            The target distance in cm
        speed : float
            The desired speed in cm/s
        """
        self.reset_encoders()

        normalized_speed = abs(speed) / MAX_SPEED

        rotations_needed = math.ceil(abs(distance) / self.wheel_circumference)
        target_steps = int(rotations_needed * PULSES_PER_REVOLUTION * GEAR_RATIO)

        if distance > 0:
            self.set_forward_speed(normalized_speed)
        else:
            self.set_reverse_speed(normalized_speed)

        def ease(remaining_factor):
            # Proportional deceleration; slow down for last 20% of turn
            reduced_speed = normalized_speed * max(0.3, remaining_factor)
            
            if distance > 0:
                self.set_forward_speed(reduced_speed)
            else:
                self.set_reverse_speed(reduced_speed)

        res = self.wait_for_action(target_steps, ease)
        return res

    def turn(self, angle: float, angular_speed: float):
        """
        Parameters
        ----------
        angle : float
            The target rotation angle in deg
        angular_speed : float
            The desired rotation speed in deg/s
        """
        self.reset_encoders()

        normalized_speed = (angular_speed * (2*math.pi)/360.0 * self.wheel_distance) / MAX_SPEED

        turning_circumference = math.pi * self.wheel_distance
        travel_distance = (angle / 360.0) * turning_circumference

        rotations_needed = math.ceil(abs(travel_distance) / self.wheel_circumference)
        target_steps = abs(rotations_needed * PULSES_PER_REVOLUTION * GEAR_RATIO)

        if angle > 0:
            self.set_right_turn_speed(normalized_speed)
        else:
            self.set_left_turn_speed(normalized_speed)

        def ease(remaining_factor):
            # Proportional deceleration; slow down for last 20% of turn
                reduced_speed = normalized_speed * max(0.3, remaining_factor)
                
                if angle > 0:
                    self.set_right_turn_speed(reduced_speed)
                else:
                    self.set_left_turn_speed(reduced_speed)

        res = self.wait_for_action(target_steps, ease)
        return res
