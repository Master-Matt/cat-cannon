from pathlib import Path

from cat_cannon.app.yolo_dataset import CatDatasetRecorder
from cat_cannon.domain.models import BoundingBox, Detection
from cat_cannon.domain.safety import DetectionPolicy


class FakeFrame:
    shape = (200, 100, 3)


class FakeCv2:
    def __init__(self) -> None:
        self.writes: list[tuple[str, object]] = []

    def imwrite(self, path: str, frame: object) -> bool:
        self.writes.append((path, frame))
        Path(path).write_bytes(b"jpeg")
        return True


def _policy() -> DetectionPolicy:
    return DetectionPolicy(
        cat_class="cat",
        person_class="person",
        cat_confidence_threshold=0.4,
        person_confidence_threshold=0.5,
        consecutive_counter_frames=2,
    )


def _detection(label: str, confidence: float, bbox: BoundingBox) -> Detection:
    return Detection(track_id=f"{label}-1", label=label, confidence=confidence, bbox=bbox)


def test_cat_dataset_recorder_writes_yolo_image_and_label(tmp_path: Path) -> None:
    recorder = CatDatasetRecorder(root=tmp_path, sample_interval_s=1.0)
    cv2 = FakeCv2()
    frame = FakeFrame()

    sample = recorder.maybe_record(
        cv2=cv2,
        source_id="fixed",
        frame=frame,
        detections=[
            _detection("person", 0.99, BoundingBox(x=0, y=0, width=10, height=20)),
            _detection("cat", 0.9, BoundingBox(x=10, y=20, width=30, height=40)),
        ],
        policy=_policy(),
        now=1234.567,
    )

    assert sample is not None
    assert sample.image_path == tmp_path / "images" / "fixed" / "fixed_1234567.jpg"
    assert sample.label_path == tmp_path / "labels" / "fixed" / "fixed_1234567.txt"
    assert (
        sample.label_path.read_text(encoding="utf-8")
        == "0 0.250000 0.200000 0.300000 0.200000\n"
    )
    assert cv2.writes == [(str(sample.image_path), frame)]
    assert (tmp_path / "classes.txt").read_text(encoding="utf-8") == "cat\n"


def test_cat_dataset_recorder_throttles_each_camera_independently(tmp_path: Path) -> None:
    recorder = CatDatasetRecorder(root=tmp_path, sample_interval_s=1.0)
    cv2 = FakeCv2()
    detections = [_detection("cat", 0.9, BoundingBox(x=10, y=20, width=30, height=40))]

    first = recorder.maybe_record(
        cv2=cv2,
        source_id="fixed",
        frame=FakeFrame(),
        detections=detections,
        policy=_policy(),
        now=10.0,
    )
    throttled = recorder.maybe_record(
        cv2=cv2,
        source_id="fixed",
        frame=FakeFrame(),
        detections=detections,
        policy=_policy(),
        now=10.5,
    )
    turret = recorder.maybe_record(
        cv2=cv2,
        source_id="turret",
        frame=FakeFrame(),
        detections=detections,
        policy=_policy(),
        now=10.5,
    )

    assert first is not None
    assert throttled is None
    assert turret is not None
    assert turret.image_path.parent == tmp_path / "images" / "turret"


def test_cat_dataset_recorder_skips_frames_without_confident_cats(tmp_path: Path) -> None:
    recorder = CatDatasetRecorder(root=tmp_path, sample_interval_s=1.0)
    cv2 = FakeCv2()

    result = recorder.maybe_record(
        cv2=cv2,
        source_id="fixed",
        frame=FakeFrame(),
        detections=[
            _detection("cat", 0.1, BoundingBox(x=10, y=20, width=30, height=40)),
            _detection("person", 0.9, BoundingBox(x=10, y=20, width=30, height=40)),
        ],
        policy=_policy(),
        now=10.0,
    )

    assert result is None
    assert cv2.writes == []
