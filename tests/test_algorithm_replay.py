from pathlib import Path

from cat_cannon.app.algorithm_replay import (
    ReplayDatasetSample,
    discover_replay_samples,
    pair_fixed_with_nearest_turret,
    run_paired_algorithm_replay,
)
from cat_cannon.config import SystemConfig
from cat_cannon.domain.models import BoundingBox, CounterZone, Detection, Point
from cat_cannon.domain.safety import DetectionPolicy
from cat_cannon.domain.targeting import TrackingCalibration


def _sample(camera: str, timestamp_ms: int, bbox: BoundingBox) -> ReplayDatasetSample:
    return ReplayDatasetSample(
        camera=camera,
        timestamp_ms=timestamp_ms,
        image_path=Path(f"{camera}_{timestamp_ms}.jpg"),
        label_path=Path(f"{camera}_{timestamp_ms}.txt"),
        frame_width=200,
        frame_height=200,
        detections=[Detection("cat-0", "cat", 0.99, bbox)],
    )


def _config(*, consecutive_frames: int = 2, deadband_px: float = 8) -> SystemConfig:
    return SystemConfig(
        cooldown_frames=4,
        detection_policy=DetectionPolicy(
            cat_class="cat",
            person_class="person",
            cat_confidence_threshold=0.4,
            person_confidence_threshold=0.5,
            consecutive_counter_frames=consecutive_frames,
        ),
        tracking_calibration=TrackingCalibration(
            horizontal_deadband_px=deadband_px,
            vertical_deadband_px=deadband_px,
            horizontal_gain=0.05,
            vertical_gain=0.05,
            aim_offset_x_px=0,
            aim_offset_y_px=0,
        ),
    )


def _zone() -> CounterZone:
    return CounterZone(
        zone_id="counter",
        polygon=(
            Point(20, 20),
            Point(180, 20),
            Point(180, 180),
            Point(20, 180),
        ),
    )


def test_pairs_fixed_frames_with_nearest_turret_timestamp() -> None:
    fixed = [
        _sample("fixed", 1000, BoundingBox(80, 80, 30, 30)),
        _sample("fixed", 3000, BoundingBox(80, 80, 30, 30)),
    ]
    turret = [
        _sample("turret", 1200, BoundingBox(80, 80, 30, 30)),
        _sample("turret", 2600, BoundingBox(80, 80, 30, 30)),
        _sample("turret", 3400, BoundingBox(80, 80, 30, 30)),
    ]

    pairs = pair_fixed_with_nearest_turret(fixed, turret)

    assert [pair.fixed.timestamp_ms for pair in pairs] == [1000, 3000]
    assert [pair.turret.timestamp_ms for pair in pairs] == [1200, 2600]
    assert [pair.timestamp_delta_ms for pair in pairs] == [200, 400]


def test_discovers_yolo_split_samples_by_filename_prefix(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "fixed_123.jpg").write_bytes(b"not-a-real-image")
    (label_dir / "fixed_123.txt").write_text(
        "0 0.500000 0.500000 0.250000 0.250000\n",
        encoding="utf-8",
    )
    (image_dir / "fixed_123_aug01.jpg").write_bytes(b"not-a-real-image")
    (label_dir / "fixed_123_aug01.txt").write_text(
        "0 0.500000 0.500000 0.250000 0.250000\n",
        encoding="utf-8",
    )
    (image_dir / "turret_456.jpg").write_bytes(b"not-a-real-image")
    (label_dir / "turret_456.txt").write_text(
        "0 0.250000 0.750000 0.100000 0.200000\n",
        encoding="utf-8",
    )

    samples = discover_replay_samples(
        tmp_path,
        frame_size_reader=lambda _path: (400, 200),
    )

    assert [sample.timestamp_ms for sample in samples["fixed"]] == [123]
    assert [sample.timestamp_ms for sample in samples["turret"]] == [456]
    assert samples["fixed"][0].detections[0].bbox == BoundingBox(
        x=150.0,
        y=75.0,
        width=100.0,
        height=50.0,
    )


def test_algorithm_replay_fires_when_zone_confirmed_and_turret_centered() -> None:
    fixed_bbox = BoundingBox(80, 80, 30, 30)
    turret_centered_bbox = BoundingBox(90, 90, 20, 20)
    pairs = pair_fixed_with_nearest_turret(
        [_sample("fixed", 1000 + index, fixed_bbox) for index in range(5)],
        [_sample("turret", 1000 + index, turret_centered_bbox) for index in range(5)],
    )

    summary = run_paired_algorithm_replay(
        pairs=pairs,
        config=_config(consecutive_frames=2),
        zones=[_zone()],
    )

    assert summary.fire_count == 1
    assert [row.fire_commanded for row in summary.rows].count(True) == 1
    assert summary.rows[-1].controller_fire_count == 1
    assert not summary.rows[0].fixed_in_valid_zone
    assert all(row.fixed_in_valid_zone for row in summary.rows[1:])
    assert [row.turret_aim_locked for row in summary.rows] == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert not any(row.tracking_commanded for row in summary.rows)


def test_algorithm_replay_tracks_but_does_not_fire_until_turret_aim_locks() -> None:
    fixed_bbox = BoundingBox(80, 80, 30, 30)
    turret_off_center_bbox = BoundingBox(125, 90, 20, 20)
    pairs = pair_fixed_with_nearest_turret(
        [_sample("fixed", 1000 + index, fixed_bbox) for index in range(6)],
        [_sample("turret", 1000 + index, turret_off_center_bbox) for index in range(6)],
    )

    summary = run_paired_algorithm_replay(
        pairs=pairs,
        config=_config(consecutive_frames=2, deadband_px=5),
        zones=[_zone()],
    )

    assert summary.fire_count == 0
    assert not any(row.fire_commanded for row in summary.rows)
    assert not any(row.turret_aim_locked for row in summary.rows)
    assert any(row.tracking_commanded for row in summary.rows)
    assert any(row.applied_pan_delta is not None for row in summary.rows)
    assert summary.rows[0].turret_error_x_px == 35.0
