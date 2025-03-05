import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

from vision import VisionProcessor
from servo import ServoController
from motion import MotionController

from models.devices import devices

vp = VisionProcessor(
    devices,
    {
        cv2.CAP_PROP_FRAME_WIDTH: 1280,
        cv2.CAP_PROP_FRAME_HEIGHT: 720,
    },
)

""" gripper = ServoController()
motion = MotionController()
button = devices.action_button

def test_run():
    motion.start(speed=0.5)
    gripper.grip()
    motion.stop()
    gripper.release_grip()
    motion.turn()

button.when_pressed = test_run """

# Uncomment when needed
vp.run()
