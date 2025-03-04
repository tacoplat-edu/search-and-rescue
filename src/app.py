import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

from vision import VisionProcessor
from models.devices import devices

vp = VisionProcessor(
    devices, {
        cv2.CAP_PROP_FRAME_WIDTH: 1280,
        cv2.CAP_PROP_FRAME_HEIGHT: 720,
    }
)

# Uncomment when needed
vp.run()
