from models.devices import devices
from motion import MotionController

motion = MotionController(devices)

motion.move(10, 10)

