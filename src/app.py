import os
#import keyboard
from signal import pause

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from dotenv import load_dotenv
load_dotenv()

from vision import VisionProcessor
from servo import ServoController
from motion import MotionController

from models.devices import devices

motion = MotionController(devices)

vp = VisionProcessor(
    motion,
    {
        cv2.CAP_PROP_FRAME_WIDTH: 640,
        cv2.CAP_PROP_FRAME_HEIGHT: 480,
    },
)

button = devices.action_button
def press_handler():
   vp.run()
   vp.running = False
button.when_pressed = press_handler

#vp.run()
""" 
if os.environ.get("DEBUG") == "true":
    try:
        while True:
            state = button.pin.state
            if keyboard.read_key() == "f":
                if state:
                    button.pin.drive_low()
                else:
                    button.pin.drive_high()
    except KeyboardInterrupt:
        print("Stopped")
    finally:
        devices.servo_motor.close()
        devices.action_button.close()
else:
    pass """
pause()
