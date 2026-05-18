from cat_cannon.adapters import camera as camera_adapter


class FakeCapture:
    def __init__(self, opened: bool = True) -> None:
        self._opened = opened
        self.props: list[tuple[int, float]] = []

    def isOpened(self):
        return self._opened

    def set(self, prop: int, value: float) -> bool:
        self.props.append((prop, value))
        return True


class FakeCv2:
    CAP_GSTREAMER = 1800
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5

    def __init__(self, *, gst_opened: bool = True) -> None:
        self.gst_opened = gst_opened
        self.video_capture_calls = []
        self.fallback_capture = FakeCapture()

    def VideoCapture(self, source, backend=None):
        self.video_capture_calls.append((source, backend))
        if backend == self.CAP_GSTREAMER:
            return FakeCapture(opened=self.gst_opened)
        return self.fallback_capture


def test_gstreamer_pipeline_uses_requested_capture_size() -> None:
    pipeline = camera_adapter._gst_pipeline(
        "/dev/video0",
        width=1280,
        height=720,
        fps=30,
    )

    assert "width=1280" in pipeline
    assert "height=720" in pipeline
    assert "framerate=30/1" in pipeline


def test_open_camera_sets_fallback_capture_size(monkeypatch) -> None:
    monkeypatch.setattr(camera_adapter, "_is_jetson", lambda: False)
    cv2 = FakeCv2()

    camera = camera_adapter.open_camera(cv2, 0, width=1280, height=720, fps=30)

    assert camera is cv2.fallback_capture
    assert cv2.fallback_capture.props == [
        (cv2.CAP_PROP_FRAME_WIDTH, 1280),
        (cv2.CAP_PROP_FRAME_HEIGHT, 720),
        (cv2.CAP_PROP_FPS, 30),
    ]
