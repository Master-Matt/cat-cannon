from pathlib import Path

from cat_cannon.config import (
    clear_servo_calibration,
    clear_servo_limits,
    load_system_config,
    save_servo_limit,
    save_servo_limit_calibration,
)


def test_example_config_uses_tight_pixel_deadband_for_small_cat_boxes() -> None:
    config = load_system_config("configs/app.example.yaml")

    assert config.tracking_calibration.horizontal_deadband_px == 5
    assert config.tracking_calibration.vertical_deadband_px == 5
    assert config.tracking_calibration.aim_offset_x_px == 0
    assert config.tracking_calibration.aim_offset_y_px == 20


def test_example_config_exposes_servo_motion_limits() -> None:
    config = load_system_config("configs/app.example.yaml")

    assert config.servo_limits.pan_min_deg == 0
    assert config.servo_limits.pan_max_deg == 180
    assert config.servo_limits.tilt_min_deg == 30
    assert config.servo_limits.tilt_max_deg == 150


def test_example_config_exposes_yolo_runtime_image_size() -> None:
    config = load_system_config("configs/app.example.yaml")

    assert config.vision.yolo_imgsz == 640
    assert config.vision.yolo_detector == "yolo"
    assert config.vision.yolo_model == ""
    assert config.vision.yoloe_model == "yoloe-11s-seg.pt"
    assert [(prompt.label, prompt.text) for prompt in config.vision.yoloe_prompts] == [
        ("person", "people"),
        ("cat", "cats"),
    ]


def test_missing_vision_config_defaults_to_fast_yolo_image_size(tmp_path: Path) -> None:
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        """
system:
  arm_required: true
  cooldown_frames: 10
detection:
  cat_class: cat
  person_class: person
  cat_confidence_threshold: 0.45
  person_confidence_threshold: 0.55
  consecutive_counter_frames: 3
tracking:
  horizontal_deadband_px: 5
  vertical_deadband_px: 5
  horizontal_gain: 0.03
  vertical_gain: 0.03
  aim_offset_x_px: 0
  aim_offset_y_px: 20
""",
        encoding="utf-8",
    )

    config = load_system_config(config_path)

    assert config.vision.yolo_imgsz == 640
    assert config.vision.yolo_detector == "yolo"


def test_vision_config_can_select_yoloe_with_prompt_map(tmp_path: Path) -> None:
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        """
system:
  arm_required: true
  cooldown_frames: 10
vision:
  yolo_detector: yoloe
  yolo_imgsz: 704
  yolo_model: custom-standard.pt
  yoloe_model: yoloe-11s-seg.pt
  yoloe_prompts:
    person: people
    cat:
      - cats
      - kitten
detection:
  cat_class: cat
  person_class: person
  cat_confidence_threshold: 0.45
  person_confidence_threshold: 0.55
  consecutive_counter_frames: 3
tracking:
  horizontal_deadband_px: 5
  vertical_deadband_px: 5
  horizontal_gain: 0.03
  vertical_gain: 0.03
  aim_offset_x_px: 0
  aim_offset_y_px: 20
""",
        encoding="utf-8",
    )

    config = load_system_config(config_path)

    assert config.vision.yolo_detector == "yoloe"
    assert config.vision.yolo_imgsz == 704
    assert config.vision.selected_model_path == "yoloe-11s-seg.pt"
    assert config.vision.yolo_model == "custom-standard.pt"
    assert config.vision.prompt_texts == ("people", "cats", "kitten")
    assert config.vision.prompt_label_aliases == {
        "people": "person",
        "cats": "cat",
        "kitten": "cat",
    }


def test_vision_config_rejects_unknown_detector(tmp_path: Path) -> None:
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        """
system:
  arm_required: true
  cooldown_frames: 10
vision:
  yolo_detector: bogus
detection:
  cat_class: cat
  person_class: person
  cat_confidence_threshold: 0.45
  person_confidence_threshold: 0.55
  consecutive_counter_frames: 3
tracking:
  horizontal_deadband_px: 5
  vertical_deadband_px: 5
  horizontal_gain: 0.03
  vertical_gain: 0.03
  aim_offset_x_px: 0
  aim_offset_y_px: 20
""",
        encoding="utf-8",
    )

    try:
        load_system_config(config_path)
    except ValueError as exc:
        assert "vision.yolo_detector" in str(exc)
    else:
        raise AssertionError("Expected invalid detector to be rejected")


def test_save_servo_limit_persists_and_normalizes_bounds(tmp_path: Path) -> None:
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        """
system:
  arm_required: true
  cooldown_frames: 10
detection:
  cat_class: cat
  person_class: person
  cat_confidence_threshold: 0.45
  person_confidence_threshold: 0.55
  consecutive_counter_frames: 3
tracking:
  horizontal_deadband_px: 5
  vertical_deadband_px: 5
  horizontal_gain: 0.03
  vertical_gain: 0.03
  aim_offset_x_px: 0
  aim_offset_y_px: 20
  servo_center_pan_deg: 0
  servo_center_tilt_deg: 0
  servo_min_pan_deg: 120
  servo_max_pan_deg: 180
  servo_min_tilt_deg: 30
  servo_max_tilt_deg: 150
""",
        encoding="utf-8",
    )

    limits = save_servo_limit(config_path, "servo_max_pan_deg", 60)

    assert limits.pan_min_deg == 60
    assert limits.pan_max_deg == 120
    reloaded = load_system_config(config_path)
    assert reloaded.servo_limits.pan_min_deg == 60
    assert reloaded.servo_limits.pan_max_deg == 120


def test_save_servo_limit_calibration_preserves_camera_pov_and_derives_orientation(tmp_path: Path) -> None:
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        """
system:
  arm_required: true
  cooldown_frames: 10
detection:
  cat_class: cat
  person_class: person
  cat_confidence_threshold: 0.45
  person_confidence_threshold: 0.55
  consecutive_counter_frames: 3
tracking:
  horizontal_deadband_px: 5
  vertical_deadband_px: 5
  horizontal_gain: 0.03
  vertical_gain: 0.03
  aim_offset_x_px: 0
  aim_offset_y_px: 20
""",
        encoding="utf-8",
    )

    limits = save_servo_limit_calibration(
        config_path,
        top_tilt_deg=112,
        bottom_tilt_deg=74,
        left_pan_deg=141,
        right_pan_deg=39,
    )

    assert limits.pan_min_deg == 39
    assert limits.pan_max_deg == 141
    assert limits.tilt_min_deg == 74
    assert limits.tilt_max_deg == 112
    assert limits.pan_delta_sign == -1
    assert limits.tilt_delta_sign == -1
    reloaded = load_system_config(config_path)
    assert reloaded.servo_limits.pan_left_deg == 141
    assert reloaded.servo_limits.pan_right_deg == 39
    assert reloaded.servo_limits.tilt_top_deg == 112
    assert reloaded.servo_limits.tilt_bottom_deg == 74


def test_clear_servo_limits_removes_saved_bounds_and_returns_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        """
system:
  arm_required: true
  cooldown_frames: 10
detection:
  cat_class: cat
  person_class: person
  cat_confidence_threshold: 0.45
  person_confidence_threshold: 0.55
  consecutive_counter_frames: 3
tracking:
  horizontal_deadband_px: 5
  vertical_deadband_px: 5
  horizontal_gain: 0.03
  vertical_gain: 0.03
  aim_offset_x_px: 0
  aim_offset_y_px: 20
  servo_center_pan_deg: 0
  servo_center_tilt_deg: 0
  servo_min_pan_deg: 88
  servo_max_pan_deg: 89
  servo_min_tilt_deg: 91
  servo_max_tilt_deg: 92
  servo_left_pan_deg: 95
  servo_right_pan_deg: 75
  servo_top_tilt_deg: 105
  servo_bottom_tilt_deg: 80
""",
        encoding="utf-8",
    )

    limits = clear_servo_limits(config_path)

    assert limits.pan_min_deg == 0
    assert limits.pan_max_deg == 180
    assert limits.tilt_min_deg == 30
    assert limits.tilt_max_deg == 150
    reloaded = load_system_config(config_path)
    assert reloaded.servo_limits == limits
    saved = config_path.read_text(encoding="utf-8")
    assert "servo_left_pan_deg" not in saved
    assert "servo_bottom_tilt_deg" not in saved


def test_clear_servo_calibration_removes_bounds_and_center(tmp_path: Path) -> None:
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        """
system:
  arm_required: true
  cooldown_frames: 10
detection:
  cat_class: cat
  person_class: person
  cat_confidence_threshold: 0.45
  person_confidence_threshold: 0.55
  consecutive_counter_frames: 3
tracking:
  horizontal_deadband_px: 5
  vertical_deadband_px: 5
  horizontal_gain: 0.03
  vertical_gain: 0.03
  aim_offset_x_px: 0
  aim_offset_y_px: 20
  servo_center_pan_deg: 77
  servo_center_tilt_deg: 101
  servo_min_pan_deg: 88
  servo_max_pan_deg: 89
  servo_min_tilt_deg: 91
  servo_max_tilt_deg: 92
""",
        encoding="utf-8",
    )

    limits = clear_servo_calibration(config_path)

    reloaded = load_system_config(config_path)
    assert reloaded.servo_limits == limits
    assert reloaded.tracking_calibration.servo_center_pan_deg == 0
    assert reloaded.tracking_calibration.servo_center_tilt_deg == 0
