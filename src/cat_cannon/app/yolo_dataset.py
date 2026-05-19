from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cat_cannon.domain.models import Detection
from cat_cannon.domain.safety import DetectionPolicy


@dataclass(frozen=True)
class YoloDatasetSample:
    source_id: str
    image_path: Path
    label_path: Path
    detection_count: int


class CatDatasetRecorder:
    def __init__(
        self,
        *,
        root: str | Path,
        sample_interval_s: float = 1.0,
        class_id: int = 0,
        class_name: str = "cat",
    ) -> None:
        self.root = Path(root)
        self.sample_interval_s = max(0.0, float(sample_interval_s))
        self.class_id = int(class_id)
        self.class_name = class_name
        self._last_sample_at: dict[str, float] = {}
        self._write_dataset_metadata()

    def maybe_record(
        self,
        *,
        cv2: Any,
        source_id: str,
        frame: Any,
        detections: list[Detection],
        policy: DetectionPolicy,
        now: float | None = None,
    ) -> YoloDatasetSample | None:
        now_s = time.time() if now is None else float(now)
        source_key = _safe_source_id(source_id)
        last_sample = self._last_sample_at.get(source_key)
        if last_sample is not None and now_s - last_sample < self.sample_interval_s:
            return None

        height, width = frame.shape[:2]
        label_lines = [
            line
            for detection in detections
            if _is_confident_cat(detection, policy)
            for line in [_to_yolo_line(detection, frame_width=width, frame_height=height)]
            if line is not None
        ]
        if not label_lines:
            return None

        timestamp_ms = int(now_s * 1000)
        image_path = self.root / "images" / source_key / f"{source_key}_{timestamp_ms}.jpg"
        label_path = self.root / "labels" / source_key / f"{source_key}_{timestamp_ms}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(image_path), frame):
            raise RuntimeError(f"Failed to write dataset image {image_path}")
        label_path.write_text("".join(label_lines), encoding="utf-8")
        self._last_sample_at[source_key] = now_s

        return YoloDatasetSample(
            source_id=source_key,
            image_path=image_path,
            label_path=label_path,
            detection_count=len(label_lines),
        )

    def _write_dataset_metadata(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "classes.txt").write_text(f"{self.class_name}\n", encoding="utf-8")
        (self.root / "dataset.yaml").write_text(
            "path: .\n"
            "train: images\n"
            "val: images\n"
            "names:\n"
            f"  {self.class_id}: {self.class_name}\n",
            encoding="utf-8",
        )


def _safe_source_id(source_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id.strip())
    return normalized.strip("._-") or "camera"


def _is_confident_cat(detection: Detection, policy: DetectionPolicy) -> bool:
    return (
        detection.label == policy.cat_class
        and detection.confidence >= policy.cat_confidence_threshold
    )


def _to_yolo_line(
    detection: Detection,
    *,
    frame_width: int,
    frame_height: int,
) -> str | None:
    if frame_width <= 0 or frame_height <= 0:
        return None

    bbox = detection.bbox
    x1 = _clamp(bbox.x, 0.0, float(frame_width))
    y1 = _clamp(bbox.y, 0.0, float(frame_height))
    x2 = _clamp(bbox.x + bbox.width, 0.0, float(frame_width))
    y2 = _clamp(bbox.y + bbox.height, 0.0, float(frame_height))
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    if box_width <= 0.0 or box_height <= 0.0:
        return None

    center_x = (x1 + box_width / 2.0) / frame_width
    center_y = (y1 + box_height / 2.0) / frame_height
    normalized_width = box_width / frame_width
    normalized_height = box_height / frame_height
    return (
        f"0 {center_x:.6f} {center_y:.6f} "
        f"{normalized_width:.6f} {normalized_height:.6f}\n"
    )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))
