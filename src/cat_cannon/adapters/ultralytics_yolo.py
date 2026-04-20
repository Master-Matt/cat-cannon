from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cat_cannon.adapters.interfaces import PerceptionFrame
from cat_cannon.domain.models import BoundingBox, Detection
from cat_cannon.domain.safety import DetectionPolicy


@dataclass(frozen=True)
class YoloRuntimeConfig:
    model_path: str = "yolo11n.pt"
    device: str | None = None
    imgsz: int = 640


def build_detection_summary(detections: list[Detection], policy: DetectionPolicy) -> str:
    cat_count = sum(1 for detection in detections if detection.label == policy.cat_class)
    person_count = sum(1 for detection in detections if detection.label == policy.person_class)
    return f"cats={cat_count} people={person_count}"


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _tolist(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return list(value)


def _label_threshold(label: str, policy: DetectionPolicy) -> float | None:
    if label == policy.cat_class:
        return policy.cat_confidence_threshold
    if label == policy.person_class:
        return policy.person_confidence_threshold
    return None


def parse_ultralytics_result(result: Any, policy: DetectionPolicy) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if not boxes:
        return []

    names = getattr(result, "names", {})
    detections: list[Detection] = []
    for index, box in enumerate(boxes):
        class_id = int(_scalar(box.cls))
        label = names[class_id]
        threshold = _label_threshold(label=label, policy=policy)
        if threshold is None:
            continue

        confidence = float(_scalar(box.conf))
        if confidence < threshold:
            continue

        coords = _tolist(box.xyxy[0])
        if len(coords) != 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in coords)

        track_token = getattr(box, "id", None)
        if track_token is not None:
            track_id = f"track-{int(_scalar(track_token))}"
        else:
            track_id = f"{label}-{index}"

        detections.append(
            Detection(
                track_id=track_id,
                label=label,
                confidence=confidence,
                bbox=BoundingBox(
                    x=x1,
                    y=y1,
                    width=max(0.0, x2 - x1),
                    height=max(0.0, y2 - y1),
                ),
            )
        )

    detections.sort(key=lambda detection: detection.confidence, reverse=True)
    return detections


class UltralyticsYoloDetector:
    def __init__(self, model: Any, policy: DetectionPolicy, runtime: YoloRuntimeConfig) -> None:
        self._model = model
        self._policy = policy
        self._runtime = runtime

    @classmethod
    def open(
        cls,
        *,
        policy: DetectionPolicy,
        runtime: YoloRuntimeConfig | None = None,
    ) -> "UltralyticsYoloDetector":
        runtime_config = runtime or YoloRuntimeConfig()
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Ultralytics is required for YOLO bench detection. Install with: "
                "pip install -e '.[bench,vision]'"
            ) from exc
        return cls(model=YOLO(runtime_config.model_path), policy=policy, runtime=runtime_config)

    def detect(self, frame, source_id: str = "primary") -> PerceptionFrame:
        kwargs: dict[str, Any] = {
            "source": frame,
            "verbose": False,
            "imgsz": self._runtime.imgsz,
        }
        if self._runtime.device:
            kwargs["device"] = self._runtime.device

        results = self._model.predict(**kwargs)
        detections = parse_ultralytics_result(result=results[0], policy=self._policy)
        height, width = frame.shape[:2]
        return PerceptionFrame(
            source_id=source_id,
            width=int(width),
            height=int(height),
            detections=detections,
        )
