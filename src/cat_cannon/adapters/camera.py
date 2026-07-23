"""GPU-accelerated camera capture helpers.

On Jetson (or any system with NVIDIA GStreamer plugins), V4L2 capture is
routed through ``nvv4l2camerasrc`` → ``nvvidconv`` so that colour-space
conversion runs on the GPU instead of the CPU.  Falls back transparently
to plain ``cv2.VideoCapture`` when GStreamer is unavailable or the
pipeline fails to open.
"""

from __future__ import annotations

import os
from typing import Any


def _gst_pipeline(
    device: str, width: int = 640, height: int = 480, fps: int = 30, *, rotate_180: bool = False
) -> str:
    """Build a GStreamer pipeline string for NVIDIA hardware-accelerated V4L2 capture."""
    flip = "videoflip method=rotate-180 ! " if rotate_180 else ""
    return (
        f"v4l2src device={device} ! "
        f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
        f"{flip}"
        "videoconvert ! "
        "video/x-raw,format=BGRx ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink max-buffers=1 drop=true sync=false"
    )


class _RotatedCapture:
    """Thin wrapper that applies 180° rotation on each frame read."""

    def __init__(self, cap: Any, cv2: Any) -> None:
        self._cap = cap
        self._cv2 = cv2

    def read(self):
        ret, frame = self._cap.read()
        if ret:
            frame = self._cv2.rotate(frame, self._cv2.ROTATE_180)
        return ret, frame

    def isOpened(self):
        return self._cap.isOpened()

    def release(self):
        return self._cap.release()

    def __getattr__(self, name: str):
        return getattr(self._cap, name)


def _is_jetson() -> bool:
    """Detect NVIDIA Jetson by checking for the tegra chip-id sysfs node."""
    return os.path.isfile("/sys/module/tegra_fuse/parameters/tegra_chip_id")


def open_camera(
    cv2: Any,
    device: int | str,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    *,
    rotate_180: bool = False,
) -> Any:
    """Open a camera with GPU-accelerated capture when available.

    On Jetson, uses a GStreamer pipeline for hardware colour conversion.
    Elsewhere (laptop, CI), falls back to standard ``cv2.VideoCapture``.

    If *rotate_180* is True the image is flipped 180° — on Jetson this
    is done in the GStreamer pipeline (zero-copy), otherwise via cv2.
    """
    if _is_jetson() and isinstance(device, str):
        pipeline = _gst_pipeline(device, width=width, height=height, fps=fps, rotate_180=rotate_180)
        camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if camera.isOpened():
            return camera
        # GStreamer failed — fall through to standard capture

    camera = cv2.VideoCapture(device)
    if not camera.isOpened():
        raise SystemExit(f"Failed to open camera {device}")
    _set_capture_property(cv2, camera, "CAP_PROP_FRAME_WIDTH", width)
    _set_capture_property(cv2, camera, "CAP_PROP_FRAME_HEIGHT", height)
    _set_capture_property(cv2, camera, "CAP_PROP_FPS", fps)
    _set_capture_property(cv2, camera, "CAP_PROP_BUFFERSIZE", 1)
    if rotate_180:
        return _RotatedCapture(camera, cv2)
    return camera


def _set_capture_property(cv2: Any, camera: Any, property_name: str, value: int) -> None:
    prop = getattr(cv2, property_name, None)
    if prop is None or not hasattr(camera, "set"):
        return
    camera.set(prop, value)
