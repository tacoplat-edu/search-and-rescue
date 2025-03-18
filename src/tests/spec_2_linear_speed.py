import time

from motion import MotionController
from motion import MAX_SPEED

from models.devices import devices

motion = MotionController(devices)

ITERATION_LIMIT = 100000

# Fix time, measure distance
def run_variant_1(time_limit_s: float):
    time_start = time.time()
    motion.set_forward_speed(MAX_SPEED)

    i = 0
    while i < ITERATION_LIMIT:
        curr_time = time.time()
        diff = curr_time - time_start

        if diff >= time_limit_s:
            break
        
        i += 1
        time.sleep(0.01)

    motion.stop()

# Fix distance, measure time
def run_variant_2(distance_limit_cm: float):
    motion.move(distance_limit_cm, MAX_SPEED)

run_variant_1(5)