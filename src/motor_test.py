from signal import pause

from models.devices import devices
from models.wheel import Wheel

devices.wheel_motors[Wheel.LEFT].forward(0.3)
devices.wheel_motors[Wheel.RIGHT].forward(0.3)

pause()