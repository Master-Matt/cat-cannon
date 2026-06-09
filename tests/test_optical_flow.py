import pytest

from cat_cannon.adapters.optical_flow import mean_flow, to_gray

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def _textured_frame(width=160, height=120):
    rng = np.random.default_rng(1234)
    return (rng.integers(0, 256, size=(height, width, 3))).astype(np.uint8)


def test_to_gray_passthrough_for_gray():
    gray = np.zeros((10, 10), dtype=np.uint8)
    assert to_gray(cv2, gray) is gray


def test_to_gray_converts_bgr():
    frame = _textured_frame()
    gray = to_gray(cv2, frame)
    assert gray.ndim == 2
    assert gray.shape == frame.shape[:2]


def test_mean_flow_detects_horizontal_shift():
    frame = _textured_frame()
    gray = to_gray(cv2, frame)
    shift = 4
    shifted = np.roll(gray, shift, axis=1)
    dx, dy, magnitude = mean_flow(cv2, gray, shifted)
    # A rightward roll should register meaningful positive-x flow.
    assert magnitude > 0.5
    assert dx > abs(dy)


def test_mean_flow_zero_for_identical_frames():
    frame = _textured_frame()
    gray = to_gray(cv2, frame)
    _, _, magnitude = mean_flow(cv2, gray, gray)
    assert magnitude < 0.1
