import time
from models.devices import devices

class ServoController:
    def __init__(self) -> None:
        self.servo = devices.servo_motor

    def set_servo_state(self, is_active: bool):
        if is_active:
            self.servo.angle = 0
        else:
            self.servo.angle = 120
        time.sleep(0.5)
        self.servo.detach()
