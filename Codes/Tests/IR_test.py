"""Live view of one infrared camera of an Intel RealSense D455.

Works over USB 2 as well as USB 3: the script checks which IR modes the
camera actually offers and falls back to the closest supported one if the
requested mode (or the right IR camera) is not available.

Usage:
    python IR_test.py                     # left IR, 640x480 @ 30 fps, emitter off
    python IR_test.py --index 2           # right IR (USB 3 only)
    python IR_test.py --emitter           # turn the dot projector on

Keys: q or Esc to quit, e to toggle the emitter, s to save a PNG snapshot.
"""
import argparse
import time

import cv2
import numpy as np
import pyrealsense2 as rs


def supported_ir_modes(device):
    """Return {index: set of (width, height, fps)} for Y8 infrared streams."""
    modes = {}
    for p in device.first_depth_sensor().get_stream_profiles():
        if p.stream_type() != rs.stream.infrared or p.format() != rs.format.y8:
            continue
        v = p.as_video_stream_profile()
        modes.setdefault(p.stream_index(), set()).add((v.width(), v.height(), p.fps()))
    return modes


def pick_mode(available, width, height, fps):
    """Exact match if possible, else the largest mode with at least `fps`,
    else the fastest mode."""
    if (width, height, fps) in available:
        return width, height, fps
    fast_enough = [m for m in available if m[2] >= fps]
    if fast_enough:
        return max(fast_enough, key=lambda m: (m[0] * m[1], m[2]))
    return max(available, key=lambda m: (m[2], m[0] * m[1]))


def main():
    ap = argparse.ArgumentParser(description="Show one RealSense IR stream")
    ap.add_argument("--index", type=int, default=1, choices=[1, 2],
                    help="IR camera: 1 = left, 2 = right (default 1)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--emitter", action="store_true",
                    help="enable the IR dot projector (off by default)")
    args = ap.parse_args()

    devices = rs.context().query_devices()
    if len(devices) == 0:
        raise SystemExit("No RealSense camera found - check the cable.")
    device = devices[0]
    usb = device.get_info(rs.camera_info.usb_type_descriptor)
    print(f"{device.get_info(rs.camera_info.name)} on USB {usb}")

    modes = supported_ir_modes(device)
    index = args.index
    if index not in modes:
        print(f"IR camera {index} is not available on USB {usb}, using IR 1 instead")
        index = 1
    width, height, fps = pick_mode(modes[index], args.width, args.height, args.fps)
    if (width, height, fps) != (args.width, args.height, args.fps):
        print(f"{args.width}x{args.height} @ {args.fps} fps not supported, "
              f"using {width}x{height} @ {fps} fps")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device.get_info(rs.camera_info.serial_number))
    config.enable_stream(rs.stream.infrared, index, width, height, rs.format.y8, fps)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    emitter_on = args.emitter
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1 if emitter_on else 0)

    side = "left" if index == 1 else "right"
    window = f"RealSense IR {index} ({side}) {width}x{height}@{fps}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            ir = frames.get_infrared_frame(index)
            if not ir:
                continue
            img = np.asanyarray(ir.get_data())

            cv2.imshow(window, img)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("e") and depth_sensor.supports(rs.option.emitter_enabled):
                emitter_on = not emitter_on
                depth_sensor.set_option(rs.option.emitter_enabled, 1 if emitter_on else 0)
                print(f"emitter {'on' if emitter_on else 'off'}")
            if key == ord("s"):
                name = f"ir{index}_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(name, img)
                print(f"saved {name}")
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
