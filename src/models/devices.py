# stdlib imports
import os
from dataclasses import dataclass

# external imports
from gpiozero import Device, Motor, Button, AngularServo, RotaryEncoder
from gpiozero.pins.mock import MockFactory, MockPWMPin

# local imports
from models.wheel import Wheel

if os.environ.get("DEBUG") == "true":
    Device.pin_factory = MockFactory(pin_class=MockPWMPin)

@dataclass
class DeviceConfiguration:
    wheel_motors: dict[Wheel : Motor]
    wheel_encoders: dict[Wheel: RotaryEncoder]
    servo_motor: Motor
    action_button: Button


devices = (
    DeviceConfiguration(
        wheel_motors={
            Wheel.LEFT: Motor(5,6,enable=19),
            Wheel.RIGHT: Motor(17,27,enable=12),
        },
        wheel_encoders={
            Wheel.LEFT: RotaryEncoder(26,25, max_steps=9000),
            Wheel.RIGHT: RotaryEncoder(23,24, max_steps=9000),
        },
        servo_motor=AngularServo(9, min_angle=-90, max_angle=90),
        action_button=Button(16, pull_up=True),
    )
)
