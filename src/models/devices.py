# stdlib imports
from dataclasses import dataclass

# external imports
from gpiozero import Device, Motor, Button, AngularServo, RotaryEncoder
from gpiozero.pins.mock import MockFactory, MockPWMPin

# local imports
from models.wheel import Wheel

DEBUG = True

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
            Wheel.LEFT: Motor(17,27,enable=12),
            Wheel.RIGHT: Motor(5,6,enable=19),
        },
        wheel_encoders={
            Wheel.LEFT: RotaryEncoder(23,24),
            Wheel.RIGHT: RotaryEncoder(25,26),
        },
        servo_motor=AngularServo(9, min_angle=-90, max_angle=90),
        action_button=Button(10),
    )
)
