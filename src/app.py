from models.devices import devices
from servo import ServoController
from motion import MotionController

gripper = ServoController()
motion = MotionController()
button = devices.action_button

def test_run():
    motion.start(speed=0.5)
    gripper.grip()
    motion.stop()
    gripper.release_grip()
    motion.turn()
    


button.when_pressed = test_run

