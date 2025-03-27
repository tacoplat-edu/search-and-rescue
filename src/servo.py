import time

from gpiozero import AngularServo
from models.devices import devices
from models.custom_servo import CustomAngularServo

class ServoController:
    servo: CustomAngularServo

    last_servo_update: float
    servo_hold_interval: float

    def __init__(self) -> None:
        self.active_flag = False
        self.servo = devices.servo_motor
        self.servo.angle(0)  # Initialize the servo angle
        self.last_servo_update = 0
        self.servo_hold_interval = 0.25

    def set_servo_state(self, is_active: bool):
        self.active_flag = is_active

    def set_servo_angle(self, angle):
        now = time.time()
        if now - self.last_servo_update > self.servo_hold_interval:
            self.servo.angle(angle)
            self.last_servo_update = now
        #time.sleep(0.5)
        #self.servo.detach()

    def release_grip(self):
        if self.active_flag == True:
            self.set_servo_angle(0)
            self.set_servo_state(False)

    def grip(self):
        if self.active_flag == False:
            self.set_servo_angle(0.9)
            self.set_servo_state(True)
