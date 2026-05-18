from pathlib import Path

from cat_cannon.app.main import parse_args


def _write_app_config(path: Path, *, yolo_imgsz: int) -> None:
    path.write_text(
        f"""
system:
  arm_required: true
  cooldown_frames: 10
vision:
  yolo_imgsz: {yolo_imgsz}
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


def test_main_uses_yolo_image_size_from_app_config(tmp_path: Path) -> None:
    config_path = tmp_path / "app.yaml"
    _write_app_config(config_path, yolo_imgsz=832)

    config = parse_args(["--config", str(config_path)])

    assert config.yolo_imgsz == 832


def test_main_yolo_image_size_cli_overrides_app_config(tmp_path: Path) -> None:
    config_path = tmp_path / "app.yaml"
    _write_app_config(config_path, yolo_imgsz=832)

    config = parse_args(["--config", str(config_path), "--imgsz", "640"])

    assert config.yolo_imgsz == 640


def test_main_uses_yoloe_detector_and_prompts_from_app_config(tmp_path: Path) -> None:
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        """
system:
  arm_required: true
  cooldown_frames: 10
vision:
  yolo_detector: yoloe
  yolo_imgsz: 640
  yoloe_model: yoloe-11s-seg.pt
  yoloe_prompts:
    person: people
    cat: cats
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

    config = parse_args(["--config", str(config_path)])

    assert config.yolo_detector == "yoloe"
    assert config.yolo_model == "yoloe-11s-seg.pt"
    assert [(prompt.label, prompt.text) for prompt in config.yolo_prompts] == [
        ("person", "people"),
        ("cat", "cats"),
    ]
