import time
from models.devices import devices
from servo import ServoController

s = ServoController()

s.set_servo_state(False)
time.sleep(1)
s.set_servo_state(True)
time.sleep(1)

s.set_servo_state(False)
time.sleep(1)
s.set_servo_state(True)
time.sleep(1)