# stdlib imports
from dataclasses import dataclass

# external imports
import gpiozero

# local imports
from models.wheel import Wheel

DEBUG = True


@dataclass
class DeviceConfiguration:
    wheel_motors: dict[Wheel : gpiozero.Motor]
    servo_motor: gpiozero.Motor
    action_button: gpiozero.Button


devices = (
    None
    if DEBUG
    else DeviceConfiguration(
        wheel_motors={
            Wheel.LEFT: gpiozero.Motor(23),
            Wheel.RIGHT: gpiozero.Motor(17),
        },
        servo_motor=gpiozero.AngularServo(12, min_angle = -90, max_angle = 90),
        action_button=gpiozero.Button(24),
    )
)