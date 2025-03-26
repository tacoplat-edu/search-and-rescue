import time
from gpiozero import AngularServo

s = AngularServo(13, min_angle=0, max_angle=180,
min_pulse_width=0.0005,
max_pulse_width=0.0025 )

try:
    while True:
        s.max()
        print("max")
        time.sleep(2)
        s.min()
        print("min")
        time.sleep(2)
except KeyboardInterrupt:
    print("exiting")