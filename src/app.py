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
# from gpiozero import Servo
# from time import sleep
# servo = Servo(13)  # GPIO pin number (BCM numbering)

# try:
#     while True:
#         servo.mid()
#         sleep(1)
#         servo.min()
#         sleep(1)
#         servo.max()
#         sleep(1)
#         print("powering servo")
# except KeyboardInterrupt:
#     print("Exiting....")
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(13, GPIO.OUT)

pwm = GPIO.PWM(13, 50)  # 50Hz for standard servo
pwm.start(7.5)  # Middle position (duty cycle ~7.5%)

try:
    while True:
        pwm.ChangeDutyCycle(7.5)  # Middle
        time.sleep(1)
        pwm.ChangeDutyCycle(5.0)  # Min
        time.sleep(1)
        pwm.ChangeDutyCycle(10.0) # Max
        time.sleep(1)
except KeyboardInterrupt:
    pwm.stop()
    GPIO.cleanup()
