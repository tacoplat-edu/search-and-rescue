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
            Wheel.LEFT: gpiozero.Motor(),
            Wheel.RIGHT: gpiozero.Motor(),
        },
        servo_motor=gpiozero.Motor(),
        action_button=gpiozero.Button(),
    )
)
