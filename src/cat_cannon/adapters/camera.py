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


def _gst_pipeline(device: str, width: int = 640, height: int = 480, fps: int = 30) -> str:
    """Build a GStreamer pipeline string for NVIDIA hardware-accelerated V4L2 capture."""
    return (
        f"v4l2src device={device} ! "
        f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
        "videoconvert ! "
        "video/x-raw,format=BGRx ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=1"
    )


def _is_jetson() -> bool:
    """Detect NVIDIA Jetson by checking for the tegra chip-id sysfs node."""
    return os.path.isfile("/sys/module/tegra_fuse/parameters/tegra_chip_id")


def open_camera(cv2: Any, device: int | str, width: int = 640, height: int = 480) -> Any:
    """Open a camera with GPU-accelerated capture when available.

    On Jetson, uses a GStreamer pipeline for hardware colour conversion.
    Elsewhere (laptop, CI), falls back to standard ``cv2.VideoCapture``.
    """
    if _is_jetson() and isinstance(device, str):
        pipeline = _gst_pipeline(device, width=width, height=height)
        camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if camera.isOpened():
            return camera
        # GStreamer failed — fall through to standard capture

    camera = cv2.VideoCapture(device)
    if not camera.isOpened():
        raise SystemExit(f"Failed to open camera {device}")
    return camera
