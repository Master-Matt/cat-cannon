from pathlib import Path

from cat_cannon.app.calibrate_zones import preview_padding
from cat_cannon.app.zone_calibration import (
    CalibrationLayout,
    ZoneCalibrationSession,
    map_display_to_frame,
)
from cat_cannon.config import load_counter_zones, save_counter_zones
from cat_cannon.domain.models import CounterZone, Point


def test_calibration_session_finalizes_zone_after_four_points() -> None:
    session = ZoneCalibrationSession(zone_prefix="counter")

    assert session.add_point(Point(10, 10)) is None
    assert session.add_point(Point(20, 10)) is None
    assert session.add_point(Point(20, 20)) is None

    zone = session.add_point(Point(10, 20))

    assert zone is not None
    assert zone.zone_id == "counter-1"
    assert len(zone.polygon) == 4
    assert session.pending_points == []
    assert len(session.zones) == 1


def test_calibration_session_undo_prefers_pending_points_then_last_zone() -> None:
    session = ZoneCalibrationSession(zone_prefix="zone")
    session.add_point(Point(1, 1))
    session.add_point(Point(2, 2))

    session.undo()
    assert session.pending_points == [Point(1, 1)]

    session.add_point(Point(2, 1))
    session.add_point(Point(2, 2))
    session.add_point(Point(1, 2))
    assert len(session.zones) == 1

    session.undo()
    assert len(session.zones) == 0


def test_map_display_to_frame_accounts_for_scaled_preview() -> None:
    layout = CalibrationLayout(
        window_width=1024,
        window_height=600,
        preview_width=800,
        preview_height=450,
        preview_offset_x=0,
        preview_offset_y=75,
        panel_x=800,
    )

    mapped = map_display_to_frame(
        display_x=400,
        display_y=300,
        frame_width=1600,
        frame_height=900,
        layout=layout,
    )

    assert mapped == Point(800.0, 450.0)


def test_save_counter_zones_round_trips_yaml(tmp_path: Path) -> None:
    path = tmp_path / "zones.yaml"
    zones = [
        CounterZone(
            zone_id="kitchen-island",
            polygon=(
                Point(10.0, 20.0),
                Point(30.0, 20.0),
                Point(30.0, 40.0),
                Point(10.0, 40.0),
            ),
        )
    ]

    save_counter_zones(path, zones)
    loaded = load_counter_zones(path)

    assert loaded == zones


def test_preview_padding_is_non_negative_for_1024x600_touch_layout() -> None:
    layout = CalibrationLayout(
        window_width=1024,
        window_height=600,
        preview_width=800,
        preview_height=600,
        preview_offset_x=0,
        preview_offset_y=0,
        panel_x=800,
    )

    assert preview_padding(layout) == (0, 0, 0, 0)
