"This script shows the left OR right infrared stream from a RealSense camera (one at a time, for USB 2 cables). The IR dot pattern projector is turned off to avoid interference with the infrared images."

import pyrealsense2 as rs
import numpy as np
import cv2

CAMERA = 1  # 1 = left, 2 = right

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.infrared, CAMERA, 640, 480, rs.format.y8, 30)
profile = pipe.start(cfg)

# Turn off the IR dot pattern projector
depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.emitter_enabled, 0)

try:
    while True:
        frames = pipe.wait_for_frames()
        img = np.asanyarray(frames.get_infrared_frame(CAMERA).get_data())
        cv2.imshow("Left IR" if CAMERA == 1 else "Right IR", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    pipe.stop()
    cv2.destroyAllWindows()
