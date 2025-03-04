import numpy as np
import cv2

from typing import Optional

NUM_DOTS = 3

def get_dot_locations(body: cv2.typing.MatLike, frame_dims: tuple[int, int]):
    res = []
    for i in range(1, NUM_DOTS+1):
        coords = [
            int(frame_dims[0] // 2),
            int(frame_dims[1] * (i / (NUM_DOTS+1))),
        ]
        coords[1] = min(max(0, coords[1]), body.shape[0]-1)

        horizontal_slice = body[coords[1], :]
        intersections = np.where(horizontal_slice == 255)[0]

        if len(intersections) >= 2:
            coords[0] = int(np.mean(intersections))

        res.append(coords)
    return res
        

def draw_dots(body: cv2.typing.MatLike, locations: list[tuple[int, int]]):
    if not locations:
        return None
    for location in locations:
        cv2.circle(body, location, 6, (0,0,255), 2)