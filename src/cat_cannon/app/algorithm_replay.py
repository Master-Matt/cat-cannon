from __future__ import annotations

import csv
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cat_cannon.adapters.controller import NullTurretController
from cat_cannon.app.supervisor import SupervisorLoop
from cat_cannon.config import SystemConfig
from cat_cannon.domain.models import BoundingBox, CounterZone, Detection, SupervisorState

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
TIMESTAMP_RE = re.compile(r"^(?P<camera>fixed|turret)_(?P<timestamp>\d+)")


@dataclass(frozen=True)
class ReplayDatasetSample:
    camera: str
    timestamp_ms: int
    image_path: Path
    label_path: Path
    frame_width: int
    frame_height: int
    detections: list[Detection]


@dataclass(frozen=True)
class PairedReplaySample:
    fixed: ReplayDatasetSample
    turret: ReplayDatasetSample
    timestamp_delta_ms: int


@dataclass(frozen=True)
class AlgorithmReplayRow:
    index: int
    fixed_timestamp_ms: int
    turret_timestamp_ms: int
    timestamp_delta_ms: int
    fixed_image: Path
    turret_image: Path
    fixed_cat_count: int
    turret_cat_count: int
    fixed_in_valid_zone: bool
    active_zone_id: str | None
    counter_confirmed: bool
    target_visible: bool
    turret_aim_locked: bool
    state: SupervisorState
    fire_commanded: bool
    controller_fire_count: int
    pan_delta: float | None
    tilt_delta: float | None
    turret_error_x_px: float | None
    turret_error_y_px: float | None
    tracking_commanded: bool
    applied_pan_delta: float | None
    applied_tilt_delta: float | None


@dataclass(frozen=True)
class AlgorithmReplaySummary:
    rows: list[AlgorithmReplayRow]
    fire_count: int
    stopped_count: int


FrameSizeReader = Callable[[Path], tuple[int, int]]


def discover_replay_samples(
    dataset_root: str | Path,
    *,
    class_name: str = "cat",
    confidence: float = 0.99,
    stable_track_ids: bool = True,
    include_augmented: bool = False,
    frame_size_reader: FrameSizeReader | None = None,
) -> dict[str, list[ReplayDatasetSample]]:
    root = Path(dataset_root)
    image_root = root / "images"
    label_root = root / "labels"
    if not image_root.exists():
        raise ValueError(f"missing image root: {image_root}")
    if not label_root.exists():
        raise ValueError(f"missing label root: {label_root}")

    read_frame_size = frame_size_reader or read_image_size
    samples: dict[str, list[ReplayDatasetSample]] = {"fixed": [], "turret": []}
    image_paths = sorted(
        path for path in image_root.glob("**/*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    for image_path in image_paths:
        parsed = parse_sample_stem(image_path.stem)
        if parsed is None:
            continue
        camera, timestamp_ms = parsed
        if not include_augmented and _is_augmented_or_repeated(image_path.stem):
            continue

        relative = image_path.relative_to(image_root)
        label_path = label_root / relative.with_suffix(".txt")
        if not label_path.exists():
            raise ValueError(f"missing label for {image_path}: {label_path}")

        frame_width, frame_height = read_frame_size(image_path)
        detections = yolo_label_file_to_detections(
            label_path,
            frame_width=frame_width,
            frame_height=frame_height,
            class_name=class_name,
            confidence=confidence,
            track_id_prefix="cat" if stable_track_ids else image_path.stem,
        )
        samples[camera].append(
            ReplayDatasetSample(
                camera=camera,
                timestamp_ms=timestamp_ms,
                image_path=image_path,
                label_path=label_path,
                frame_width=frame_width,
                frame_height=frame_height,
                detections=detections,
            )
        )

    for camera in samples:
        samples[camera].sort(key=lambda sample: sample.timestamp_ms)
    return samples


def parse_sample_stem(stem: str) -> tuple[str, int] | None:
    match = TIMESTAMP_RE.match(stem)
    if match is None:
        return None
    return match.group("camera"), int(match.group("timestamp"))


def pair_fixed_with_nearest_turret(
    fixed_samples: list[ReplayDatasetSample],
    turret_samples: list[ReplayDatasetSample],
    *,
    max_timestamp_delta_ms: int | None = None,
) -> list[PairedReplaySample]:
    if not fixed_samples:
        return []
    if not turret_samples:
        raise ValueError("cannot replay fixed detections without turret samples")

    sorted_turret = sorted(turret_samples, key=lambda sample: sample.timestamp_ms)
    pairs: list[PairedReplaySample] = []
    for fixed in sorted(fixed_samples, key=lambda sample: sample.timestamp_ms):
        turret = min(
            sorted_turret,
            key=lambda sample: abs(sample.timestamp_ms - fixed.timestamp_ms),
        )
        delta = abs(turret.timestamp_ms - fixed.timestamp_ms)
        if max_timestamp_delta_ms is not None and delta > max_timestamp_delta_ms:
            continue
        pairs.append(PairedReplaySample(fixed=fixed, turret=turret, timestamp_delta_ms=delta))
    return pairs


def run_dataset_algorithm_replay(
    *,
    dataset_root: str | Path,
    config: SystemConfig,
    zones: list[CounterZone],
    max_timestamp_delta_ms: int | None = 5000,
    include_augmented: bool = False,
) -> AlgorithmReplaySummary:
    samples = discover_replay_samples(
        dataset_root,
        include_augmented=include_augmented,
    )
    pairs = pair_fixed_with_nearest_turret(
        samples["fixed"],
        samples["turret"],
        max_timestamp_delta_ms=max_timestamp_delta_ms,
    )
    return run_paired_algorithm_replay(pairs=pairs, config=config, zones=zones)


def run_paired_algorithm_replay(
    *,
    pairs: list[PairedReplaySample],
    config: SystemConfig,
    zones: list[CounterZone],
) -> AlgorithmReplaySummary:
    controller = NullTurretController()
    supervisor = SupervisorLoop(config=config, zones=zones, controller=controller)
    rows: list[AlgorithmReplayRow] = []

    for index, pair in enumerate(pairs):
        pan_command_count = len(controller.pan_commands)
        turret_error_x, turret_error_y = _turret_target_error(
            pair.turret,
            config=config,
        )
        result = supervisor.process_frame(
            detections=pair.fixed.detections,
            frame_width=pair.fixed.frame_width,
            frame_height=pair.fixed.frame_height,
            armed=True,
            turret_detections=pair.turret.detections,
            turret_frame_width=pair.turret.frame_width,
            turret_frame_height=pair.turret.frame_height,
        )
        tracking_commanded = len(controller.pan_commands) > pan_command_count
        rows.append(
            AlgorithmReplayRow(
                index=index,
                fixed_timestamp_ms=pair.fixed.timestamp_ms,
                turret_timestamp_ms=pair.turret.timestamp_ms,
                timestamp_delta_ms=pair.timestamp_delta_ms,
                fixed_image=pair.fixed.image_path,
                turret_image=pair.turret.image_path,
                fixed_cat_count=len(pair.fixed.detections),
                turret_cat_count=len(pair.turret.detections),
                fixed_in_valid_zone=result.active_zone_id is not None,
                active_zone_id=result.active_zone_id,
                counter_confirmed=result.counter_confirmed,
                target_visible=result.target_visible,
                turret_aim_locked=result.aim_locked,
                state=result.state,
                fire_commanded=result.fire_commanded,
                controller_fire_count=controller.fired,
                pan_delta=result.correction.pan_delta if result.correction else None,
                tilt_delta=result.correction.tilt_delta if result.correction else None,
                turret_error_x_px=turret_error_x,
                turret_error_y_px=turret_error_y,
                tracking_commanded=tracking_commanded,
                applied_pan_delta=controller.pan_commands[-1] if tracking_commanded else None,
                applied_tilt_delta=controller.tilt_commands[-1] if tracking_commanded else None,
            )
        )

    return AlgorithmReplaySummary(
        rows=rows,
        fire_count=controller.fired,
        stopped_count=controller.stopped,
    )


def yolo_label_file_to_detections(
    label_path: str | Path,
    *,
    frame_width: int,
    frame_height: int,
    class_name: str = "cat",
    confidence: float = 0.99,
    track_id_prefix: str = "cat",
) -> list[Detection]:
    detections: list[Detection] = []
    for index, line in enumerate(Path(label_path).read_text(encoding="utf-8").splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(f"invalid YOLO label at {label_path}:{index + 1}")
        class_id, center_x, center_y, width, height = parts
        if int(class_id) != 0:
            continue
        bbox = _denormalize_yolo_bbox(
            center_x=float(center_x),
            center_y=float(center_y),
            width=float(width),
            height=float(height),
            frame_width=frame_width,
            frame_height=frame_height,
        )
        detections.append(
            Detection(
                track_id=f"{track_id_prefix}-{index}",
                label=class_name,
                confidence=confidence,
                bbox=bbox,
            )
        )
    return detections


def read_image_size(path: str | Path) -> tuple[int, int]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "OpenCV is required for dataset algorithm replay. Install with: "
            "pip install -e '.[bench]'"
        ) from exc

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"failed to read image size: {path}")
    height, width = image.shape[:2]
    return int(width), int(height)


def write_replay_csv(path: str | Path, rows: list[AlgorithmReplayRow]) -> None:
    fieldnames = [
        "index",
        "fixed_timestamp_ms",
        "turret_timestamp_ms",
        "timestamp_delta_ms",
        "fixed_image",
        "turret_image",
        "fixed_cat_count",
        "turret_cat_count",
        "fixed_in_valid_zone",
        "active_zone_id",
        "counter_confirmed",
        "target_visible",
        "turret_aim_locked",
        "state",
        "fire_commanded",
        "controller_fire_count",
        "pan_delta",
        "tilt_delta",
        "turret_error_x_px",
        "turret_error_y_px",
        "tracking_commanded",
        "applied_pan_delta",
        "applied_tilt_delta",
    ]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_to_csv(row))


def _row_to_csv(row: AlgorithmReplayRow) -> dict[str, object]:
    return {
        "index": row.index,
        "fixed_timestamp_ms": row.fixed_timestamp_ms,
        "turret_timestamp_ms": row.turret_timestamp_ms,
        "timestamp_delta_ms": row.timestamp_delta_ms,
        "fixed_image": row.fixed_image.as_posix(),
        "turret_image": row.turret_image.as_posix(),
        "fixed_cat_count": row.fixed_cat_count,
        "turret_cat_count": row.turret_cat_count,
        "fixed_in_valid_zone": row.fixed_in_valid_zone,
        "active_zone_id": row.active_zone_id or "",
        "counter_confirmed": row.counter_confirmed,
        "target_visible": row.target_visible,
        "turret_aim_locked": row.turret_aim_locked,
        "state": row.state.value,
        "fire_commanded": row.fire_commanded,
        "controller_fire_count": row.controller_fire_count,
        "pan_delta": "" if row.pan_delta is None else f"{row.pan_delta:.4f}",
        "tilt_delta": "" if row.tilt_delta is None else f"{row.tilt_delta:.4f}",
        "turret_error_x_px": (
            "" if row.turret_error_x_px is None else f"{row.turret_error_x_px:.2f}"
        ),
        "turret_error_y_px": (
            "" if row.turret_error_y_px is None else f"{row.turret_error_y_px:.2f}"
        ),
        "tracking_commanded": row.tracking_commanded,
        "applied_pan_delta": (
            "" if row.applied_pan_delta is None else f"{row.applied_pan_delta:.4f}"
        ),
        "applied_tilt_delta": (
            "" if row.applied_tilt_delta is None else f"{row.applied_tilt_delta:.4f}"
        ),
    }


def _turret_target_error(
    sample: ReplayDatasetSample,
    *,
    config: SystemConfig,
) -> tuple[float | None, float | None]:
    if not sample.detections:
        return None, None
    target = sample.detections[0].bbox.center
    calibration = config.tracking_calibration
    error_x = target.x - (sample.frame_width / 2.0 + calibration.aim_offset_x_px)
    error_y = target.y - (sample.frame_height / 2.0 + calibration.aim_offset_y_px)
    return error_x, error_y


def _denormalize_yolo_bbox(
    *,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    frame_width: int,
    frame_height: int,
) -> BoundingBox:
    pixel_width = width * frame_width
    pixel_height = height * frame_height
    x = center_x * frame_width - pixel_width / 2.0
    y = center_y * frame_height - pixel_height / 2.0
    return BoundingBox(x=x, y=y, width=pixel_width, height=pixel_height)


def _is_augmented_or_repeated(stem: str) -> bool:
    return "_aug" in stem or "_repeat" in stem
