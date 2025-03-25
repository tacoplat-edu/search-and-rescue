import time
from gpiozero import AngularServo

s = AngularServo(13, min_angle=0, max_angle=180)

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