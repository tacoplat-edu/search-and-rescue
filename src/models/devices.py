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
            Wheel.RIGHT: Motor(27,17,enable=12),
        },
        wheel_encoders={
            Wheel.LEFT: RotaryEncoder(26,25, max_steps=25000),
            Wheel.RIGHT: RotaryEncoder(23,24, max_steps=25000),
        },
        servo_motor=AngularServo(13, min_angle=0, max_angle=180,min_pulse_width=0.0005,max_pulse_width=0.0025),
        action_button=Button(16, pull_up=True),
    )
)
