from types import SimpleNamespace

from cat_cannon.adapters.ultralytics_yolo import (
    YoloRuntimeConfig,
    build_detection_summary,
    parse_ultralytics_result,
)
from cat_cannon.domain.models import BoundingBox, Detection
from cat_cannon.domain.safety import DetectionPolicy


def _policy() -> DetectionPolicy:
    return DetectionPolicy(
        cat_class="cat",
        person_class="person",
        cat_confidence_threshold=0.45,
        person_confidence_threshold=0.55,
        consecutive_counter_frames=3,
    )


class FakeTensor:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value

    def tolist(self):
        return list(self._value)


def _box(*, cls_id: int, conf: float, xyxy: list[float], track_id: int | None = None):
    payload = {
        "cls": FakeTensor(cls_id),
        "conf": FakeTensor(conf),
        "xyxy": [FakeTensor(xyxy)],
    }
    if track_id is not None:
        payload["id"] = FakeTensor(track_id)
    return SimpleNamespace(**payload)


def test_parse_ultralytics_result_filters_to_cat_and_person_thresholds() -> None:
    result = SimpleNamespace(
        names={0: "person", 15: "cat", 16: "dog"},
        boxes=[
            _box(cls_id=15, conf=0.91, xyxy=[10, 20, 30, 50], track_id=7),
            _box(cls_id=0, conf=0.82, xyxy=[40, 50, 90, 150]),
            _box(cls_id=15, conf=0.30, xyxy=[1, 2, 3, 4]),
            _box(cls_id=16, conf=0.99, xyxy=[5, 6, 7, 8]),
        ],
    )

    detections = parse_ultralytics_result(result=result, policy=_policy())

    assert detections == [
        Detection(
            track_id="track-7",
            label="cat",
            confidence=0.91,
            bbox=BoundingBox(x=10.0, y=20.0, width=20.0, height=30.0),
        ),
        Detection(
            track_id="person-1",
            label="person",
            confidence=0.82,
            bbox=BoundingBox(x=40.0, y=50.0, width=50.0, height=100.0),
        ),
    ]


def test_parse_ultralytics_result_returns_empty_for_missing_boxes() -> None:
    result = SimpleNamespace(names={0: "person"}, boxes=None)

    assert parse_ultralytics_result(result=result, policy=_policy()) == []


def test_build_detection_summary_counts_cats_and_people() -> None:
    detections = [
        Detection(track_id="cat-1", label="cat", confidence=0.9, bbox=BoundingBox(x=0, y=0, width=1, height=1)),
        Detection(track_id="person-1", label="person", confidence=0.8, bbox=BoundingBox(x=0, y=0, width=1, height=1)),
        Detection(track_id="cat-2", label="cat", confidence=0.7, bbox=BoundingBox(x=0, y=0, width=1, height=1)),
    ]

    assert build_detection_summary(detections=detections, policy=_policy()) == "cats=2 people=1"


def test_runtime_config_defaults_to_yolo11n_model() -> None:
    config = YoloRuntimeConfig()

    assert config.model_path == "yolo11n.pt"
    assert config.imgsz == 640
