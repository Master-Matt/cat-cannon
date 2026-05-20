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
    person: person
    cat: cat
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
        ("person", "person"),
        ("cat", "cat"),
    ]


def test_main_parses_cat_dataset_collection_options(tmp_path: Path) -> None:
    config_path = tmp_path / "app.yaml"
    dataset_dir = tmp_path / "dataset"
    _write_app_config(config_path, yolo_imgsz=640)

    config = parse_args(
        [
            "--config",
            str(config_path),
            "--collect-cat-dataset",
            "--dataset-dir",
            str(dataset_dir),
            "--dataset-sample-hz",
            "2",
        ]
    )

    assert config.collect_cat_dataset is True
    assert config.dataset_dir == str(dataset_dir)
    assert config.dataset_sample_hz == 2.0


def test_eye_config_accepts_main_dataset_collection_options(tmp_path: Path) -> None:
    from cat_cannon.app.eye_screen import EyeConfig, build_eye_dataset_recorder

    config = EyeConfig(
        collect_cat_dataset=True,
        dataset_dir=str(tmp_path / "dataset"),
        dataset_sample_hz=2.0,
    )

    assert config.collect_cat_dataset is True
    assert config.dataset_dir == str(tmp_path / "dataset")
    assert config.dataset_sample_hz == 2.0

    recorder = build_eye_dataset_recorder(config)

    assert recorder is not None
    assert recorder.root == tmp_path / "dataset"
    assert recorder.sample_interval_s == 0.5
    assert (tmp_path / "dataset" / "dataset.yaml").exists()


def test_main_uses_event_recording_config(tmp_path: Path) -> None:
    config_path = tmp_path / "app.yaml"
    _write_app_config(config_path, yolo_imgsz=640)
    with config_path.open("a", encoding="utf-8") as handle:
        handle.write(
            """
event_recording:
  enabled: true
  output_dir: data/events
  post_shot_seconds: 15
  zone_confirm_seconds: 5
  zone_confirm_detections: 20
  zone_lost_seconds: 5
  discord_webhook_env: CAT_CANNON_TEST_WEBHOOK
"""
        )

    config = parse_args(["--config", str(config_path)])

    assert config.event_recording.enabled is True
    assert config.event_recording.output_dir == "data/events"
    assert config.event_recording.post_shot_seconds == 15.0
    assert config.event_recording.zone_confirm_seconds == 5.0
    assert config.event_recording.zone_confirm_detections == 20
    assert config.event_recording.zone_lost_seconds == 5.0
    assert config.event_recording.discord_webhook_env == "CAT_CANNON_TEST_WEBHOOK"
