"""Optical-flow helpers for confirming that the turret actually moved.

This is a thin wrapper around OpenCV so the pure motion-decision logic in
``cat_cannon.app.heartbeat`` stays import-free (cv2 lives only in the ``[bench]``
extra). ``cv2`` is injected by the caller exactly like ``adapters/camera.py``.
"""

from __future__ import annotations

from typing import Any

MeanFlow = tuple[float, float, float]


def to_gray(cv2: Any, frame: Any) -> Any:
    """Convert a BGR (or already-gray) frame to single-channel grayscale."""
    if frame is None:
        raise ValueError("frame must not be None")
    if getattr(frame, "ndim", 2) == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def mean_flow(cv2: Any, prev_gray: Any, gray: Any) -> MeanFlow:
    """Return ``(dx, dy, magnitude)`` mean dense optical flow between two frames.

    ``dx``/``dy`` are the mean per-pixel displacement (in pixels) and
    ``magnitude`` is the Euclidean length of that mean vector. Uses Farneback
    dense flow, which is robust for the small whole-frame shifts a deliberate
    heartbeat move produces.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        0.5,  # pyr_scale
        3,    # levels
        15,   # winsize
        3,    # iterations
        5,    # poly_n
        1.2,  # poly_sigma
        0,    # flags
    )
    dx = float(flow[..., 0].mean())
    dy = float(flow[..., 1].mean())
    magnitude = (dx * dx + dy * dy) ** 0.5
    return dx, dy, magnitude
