# import os
# #import keyboard
# from signal import pause

# os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
# os.environ["SHOW_IMAGE_WINDOW"] = "false"
# import cv2
# from dotenv import load_dotenv
# load_dotenv()

# from vision import VisionProcessor
# from servo import ServoController
# from motion import MotionController

# from models.devices import devices

# motion = MotionController(devices)
# servo = ServoController()

# vp = VisionProcessor(
#     motion,
#     servo,
#     {
#         cv2.CAP_PROP_FRAME_WIDTH: 640,
#         cv2.CAP_PROP_FRAME_HEIGHT: 480,
#     },
# )

# button = devices.action_button
# def press_handler():
#    print("what the sigma")
#    vp.run()
#    #vp.running = False
# button.when_pressed = press_handler

# #vp.run()
# """ 
# if os.environ.get("DEBUG") == "true":
#     try:
#         while True:
#             state = button.pin.state
#             if keyboard.read_key() == "f":
#                 if state:
#                     button.pin.drive_low()
#                 else:
#                     button.pin.drive_high()
#     except KeyboardInterrupt:
#         print("Stopped")
#     finally:
#         devices.servo_motor.close()
#         devices.action_button.close()
# else:
#     pass """
# pause()
from gpiozero import Servo, AngularServo
from gpiozero.pins.lgpio import LGPIOFactory

from time import sleep
factory = LGPIOFactory(chip=0)
#servo = Servo(18,pin_factory=factory)  # GPIO pin number (BCM numbering)
angularservo = AngularServo(18,pin_factory=factory)
try:
    while True:
        angularservo.angle = 20
        sleep(1)
        angularservo.angle = 45
        sleep(1)
        angularservo.angle = 70
        sleep(1)
        print("powering servo")
except KeyboardInterrupt:
    print("Exiting....")
