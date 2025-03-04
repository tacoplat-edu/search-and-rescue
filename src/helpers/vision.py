import numpy as np
import cv2

from typing import Optional

NUM_DOTS = 3

def get_dot_locations(frame_height: int):
    res = []
    for i in range(1, NUM_DOTS+1):
        res.append(int(frame_height * (i / (NUM_DOTS+1))))
    return res
