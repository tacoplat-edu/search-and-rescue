# stdlib imports
import os
from dataclasses import dataclass

# external imports
from gpiozero import Device, Motor, Button, Servo, RotaryEncoder
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
            Wheel.LEFT: Motor(6,5,enable=19),
            Wheel.RIGHT: Motor(27,17,enable=12),
        },
        wheel_encoders={
            Wheel.LEFT: RotaryEncoder(26,25, max_steps=9000),
            Wheel.RIGHT: RotaryEncoder(23,24, max_steps=9000),
        },
        servo_motor=Servo(13),
        action_button=Button(16, pull_up=True),
    )
)
