# Graph Report - .  (2026-05-19)

## Corpus Check
- 68 files · ~32,051 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 701 nodes · 1598 edges · 53 communities detected
- Extraction: 66% EXTRACTED · 34% INFERRED · 0% AMBIGUOUS · INFERRED: 545 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]

## God Nodes (most connected - your core abstractions)
1. `Detection` - 52 edges
2. `RP2040SerialController` - 43 edges
3. `SupervisorLoop` - 43 edges
4. `DetectionPolicy` - 37 edges
5. `YoloPrompt` - 35 edges
6. `NullTurretController` - 35 edges
7. `CounterZone` - 32 edges
8. `YoloRuntimeConfig` - 32 edges
9. `UltralyticsYoloDetector` - 31 edges
10. `TurretController` - 30 edges

## Surprising Connections (you probably didn't know these)
- `FakeTensor` --uses--> `YoloPrompt`  [INFERRED]
  tests/test_ultralytics_yolo.py → src/cat_cannon/config.py
- `FakeController` --uses--> `ServoLimits`  [INFERRED]
  tests/test_controller_session.py → src/cat_cannon/config.py
- `FakeSession` --uses--> `BoundingBox`  [INFERRED]
  tests/test_tracking_test.py → src/cat_cannon/domain/models.py
- `FakeDetector` --uses--> `BoundingBox`  [INFERRED]
  tests/test_tracking_test.py → src/cat_cannon/domain/models.py
- `FakeFrame` --uses--> `BoundingBox`  [INFERRED]
  tests/test_tracking_test.py → src/cat_cannon/domain/models.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.12
Nodes (53): BenchConfig, CalibrationConfig, UiButton, ServoLimits, YoloPrompt, NullTurretController, ControllerSession, EyeConfig (+45 more)

### Community 1 - "Community 1"
Cohesion: 0.08
Nodes (37): AlgorithmReplayRow, AlgorithmReplaySummary, _denormalize_yolo_bbox(), discover_replay_samples(), _is_augmented_or_repeated(), pair_fixed_with_nearest_turret(), PairedReplaySample, parse_sample_stem() (+29 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (38): _annotate_camera(), _build_buttons(), _control_from_key(), _control_label_rect(), _copy_tracking_state(), detect_tracking_cameras(), _draw_button(), _draw_camera_status_in_left_margin() (+30 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (20): NullTurretController, FakeCv2, FakeLimitController, FakeSession, _policy(), test_tracking_buttons_draw_labels_on_button_surfaces(), test_tracking_camera_annotations_stay_top_left(), test_tracking_camera_status_uses_left_image_margin() (+12 more)

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (15): autodetect_port(), list_candidate_ports(), _looks_like_rp2040(), SerialPortInfo, build_request(), ControllerRequest, ControllerResponse, open() (+7 more)

### Community 5 - "Community 5"
Cohesion: 0.13
Nodes (9): AppConfig, Controller, _error(), main(), _ok(), parse_args(), Stop PWM signal to eliminate servo buzz., run_app() (+1 more)

### Community 6 - "Community 6"
Cohesion: 0.14
Nodes (16): clear_servo_calibration(), clear_servo_limits(), _frame_size(), load_counter_zones(), load_vision_config(), _load_yaml(), _normalize_yolo_detector(), _optional_float() (+8 more)

### Community 7 - "Community 7"
Cohesion: 0.15
Nodes (13): handle_key(), main(), parse_args(), _print_help(), TeleopState, FakeController, FakeSession, test_handle_key_arms_disarms_and_safe_stops() (+5 more)

### Community 8 - "Community 8"
Cohesion: 0.17
Nodes (22): build_bootstrap_command(), build_deploy_steps(), build_jetson_gpu_setup_command(), build_rsync_command(), build_ssh_command(), build_udev_install_command(), deploy(), DeployStep (+14 more)

### Community 9 - "Community 9"
Cohesion: 0.21
Nodes (18): _augment_sample(), _bbox_params(), _build_augmenter(), _coarse_dropout(), _copy_sample(), discover_yolo_samples(), FinetuneDatasetSummary, _import_cv2() (+10 more)

### Community 10 - "Community 10"
Cohesion: 0.18
Nodes (9): _exec_responses(), FakeTransport, test_deploy_files_uploads_all_files_and_resets_board(), test_write_text_file_accepts_existing_raw_repl_prompt(), test_write_text_file_enters_raw_repl_and_chunks_writes(), test_write_text_file_raises_after_exhausting_raw_repl_retries(), test_write_text_file_resets_input_before_raw_repl_to_drop_stale_prompts(), test_write_text_file_retries_after_normal_repl_banner() (+1 more)

### Community 11 - "Community 11"
Cohesion: 0.2
Nodes (9): _bundled_model_path(), _default_device(), _label_threshold(), _models_dir(), open(), parse_ultralytics_result(), _promptable_model_path(), _scalar() (+1 more)

### Community 12 - "Community 12"
Cohesion: 0.17
Nodes (10): _gst_pipeline(), _is_jetson(), open_camera(), GPU-accelerated camera capture helpers.  On Jetson (or any system with NVIDIA GS, Build a GStreamer pipeline string for NVIDIA hardware-accelerated V4L2 capture., Thin wrapper that applies 180° rotation on each frame read., Detect NVIDIA Jetson by checking for the tegra chip-id sysfs node., Open a camera with GPU-accelerated capture when available.      On Jetson, uses (+2 more)

### Community 13 - "Community 13"
Cohesion: 0.26
Nodes (14): _build_buttons(), _build_layout(), _draw_button(), _draw_detections_on_preview(), _draw_pending_points(), _draw_zones(), main(), parse_args() (+6 more)

### Community 14 - "Community 14"
Cohesion: 0.21
Nodes (8): _box(), FakeTensor, _policy(), test_build_detection_summary_counts_cats_and_people(), test_parse_ultralytics_result_filters_to_cat_and_person_thresholds(), test_parse_ultralytics_result_maps_yoloe_prompts_to_policy_labels(), test_parse_ultralytics_result_returns_empty_for_missing_boxes(), test_yoloe_detector_sets_prompt_classes()

### Community 15 - "Community 15"
Cohesion: 0.22
Nodes (8): FakeSerial, test_controller_raises_on_failed_response(), test_controller_raises_when_only_noise_is_received(), test_controller_sends_expected_fire_command(), test_controller_sends_expected_servo_limits_command(), test_controller_sends_expected_set_fire_output_command(), test_controller_skips_non_json_serial_noise_before_valid_response(), test_controller_uses_detected_motion_directions_for_camera_pov_deltas()

### Community 16 - "Community 16"
Cohesion: 0.18
Nodes (6): FakePin, FakePoll, FakePWM, _load_pico_main(), test_soft_servo_limits_do_not_remap_pwm_calibration(), test_soft_servo_limits_only_clamp_target_angle()

### Community 17 - "Community 17"
Cohesion: 0.19
Nodes (4): FakeController, test_controller_session_applies_camera_pov_motion_directions(), test_controller_session_applies_servo_limits_after_handshake(), test_controller_session_manages_handshake_enable_disable_and_stop()

### Community 18 - "Community 18"
Cohesion: 0.17
Nodes (0):

### Community 19 - "Community 19"
Cohesion: 0.42
Nodes (10): deploy_files(), _enter_raw_repl(), _exec_raw(), _hard_reset(), MicroPythonDeployError, Raised when firmware deployment to the Pico fails., _read_until(), _reset_input_buffer() (+2 more)

### Community 20 - "Community 20"
Cohesion: 0.35
Nodes (10): _annotate_frame(), _draw_detections(), _draw_zones(), main(), _open_camera(), parse_args(), _print_status(), _require_cv2() (+2 more)

### Community 21 - "Community 21"
Cohesion: 0.42
Nodes (10): _cat_detection(), _person_detection(), _supervisor(), test_supervisor_does_not_fallback_track_while_disarmed_after_confirmation(), test_supervisor_does_not_track_turret_people_by_default(), test_supervisor_loop_does_not_apply_tracking_delta_when_disarmed(), test_supervisor_loop_reports_human_lockout_and_safe_stop(), test_supervisor_loop_returns_tracking_state_and_zone_after_confirmation() (+2 more)

### Community 22 - "Community 22"
Cohesion: 0.18
Nodes (0):

### Community 23 - "Community 23"
Cohesion: 0.47
Nodes (7): _detection(), FakeCv2, FakeFrame, _policy(), test_cat_dataset_recorder_skips_frames_without_confident_cats(), test_cat_dataset_recorder_throttles_each_camera_independently(), test_cat_dataset_recorder_writes_yolo_image_and_label()

### Community 24 - "Community 24"
Cohesion: 0.27
Nodes (3): FakeCapture, FakeCv2, test_open_camera_sets_fallback_capture_size()

### Community 25 - "Community 25"
Cohesion: 0.25
Nodes (6): DeepStreamPerceptionAdapter, Placeholder for the fixed-camera DeepStream pipeline integration., PerceptionAdapter, Return the latest frame metadata and detections., PerceptionAdapter, Protocol

### Community 26 - "Community 26"
Cohesion: 0.32
Nodes (1): ZoneCalibrationSession

### Community 27 - "Community 27"
Cohesion: 0.46
Nodes (7): _annotate_frame(), _combine_frames(), _draw_detections(), main(), _open_camera(), parse_args(), _require_cv2()

### Community 28 - "Community 28"
Cohesion: 0.43
Nodes (6): _draw_circular_button(), _draw_eye(), EyeState, _lerp(), _require_cv2(), run_eye_screen()

### Community 29 - "Community 29"
Cohesion: 0.5
Nodes (6): _config(), _sample(), test_algorithm_replay_does_not_fire_when_turret_is_not_centered(), test_algorithm_replay_fires_when_zone_confirmed_and_turret_centered(), test_pairs_fixed_frames_with_nearest_turret_timestamp(), _zone()

### Community 30 - "Community 30"
Cohesion: 0.25
Nodes (0):

### Community 31 - "Community 31"
Cohesion: 0.43
Nodes (6): bbox_intersects_zone(), detection_footpoint_in_zone(), point_in_polygon(), Check if line segment a1-a2 intersects segment b1-b2., Check if any part of the detection bbox overlaps with the zone polygon., _segments_intersect()

### Community 32 - "Community 32"
Cohesion: 0.48
Nodes (5): _clamp(), _is_confident_cat(), _safe_source_id(), _to_yolo_line(), YoloDatasetSample

### Community 33 - "Community 33"
Cohesion: 0.43
Nodes (4): test_main_parses_cat_dataset_collection_options(), test_main_uses_yolo_image_size_from_app_config(), test_main_yolo_image_size_cli_overrides_app_config(), _write_app_config()

### Community 34 - "Community 34"
Cohesion: 0.67
Nodes (5): _cat_detection(), _person_detection(), _supervisor(), test_replay_fires_after_confirmation_and_aim_lock(), test_replay_human_presence_forces_safe_stop_and_blocks_fire()

### Community 35 - "Community 35"
Cohesion: 0.4
Nodes (1): Application entrypoints.

### Community 36 - "Community 36"
Cohesion: 0.7
Nodes (4): _format_float(), main(), _print_rows(), _yes_no()

### Community 37 - "Community 37"
Cohesion: 0.6
Nodes (3): test_prepare_finetune_dataset_can_repeat_fixed_training_samples(), test_prepare_finetune_dataset_splits_pairs_by_camera(), _write_sample()

### Community 38 - "Community 38"
Cohesion: 0.6
Nodes (3): _policy(), test_assess_scene_flags_human_presence_even_when_cat_is_on_counter(), _zone()

### Community 39 - "Community 39"
Cohesion: 0.83
Nodes (3): main(), parse_args(), _repo_root()

### Community 40 - "Community 40"
Cohesion: 0.5
Nodes (0):

### Community 41 - "Community 41"
Cohesion: 0.5
Nodes (0):

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (2): main(), parse_args()

### Community 43 - "Community 43"
Cohesion: 0.67
Nodes (0):

### Community 44 - "Community 44"
Cohesion: 0.67
Nodes (0):

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (0):

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (0):

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (0):

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (0):

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (0):

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (0):

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (0):

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (0):

## Knowledge Gaps
- **12 isolated node(s):** `Raised when firmware deployment to the Pico fails.`, `Raised when a Pico-compatible port cannot be resolved unambiguously.`, `GPU-accelerated camera capture helpers.  On Jetson (or any system with NVIDIA GS`, `Build a GStreamer pipeline string for NVIDIA hardware-accelerated V4L2 capture.`, `Thin wrapper that applies 180° rotation on each frame read.` (+7 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 45`** (2 nodes): `main()`, `prepare_cat_finetune_dataset.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (2 nodes): `tag_zone_frame_size.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (2 nodes): `test_rebuild_yolo_engine_script_exports_configurable_tensor_rt_engine()`, `test_rebuild_yolo_engine_script.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (2 nodes): `test_x11_tracking_runner_checks_display_and_launches_tracking_ui()`, `test_tracking_x11_script.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `patch_torchvision_jetson.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `pico_config.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `conftest.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RP2040ProtocolError` connect `Community 0` to `Community 4`, `Community 15`, `Community 7`?**
  _High betweenness centrality (0.076) - this node is a cross-community bridge._
- **Why does `Detection` connect `Community 0` to `Community 32`, `Community 1`, `Community 3`, `Community 14`, `Community 23`, `Community 25`, `Community 31`?**
  _High betweenness centrality (0.070) - this node is a cross-community bridge._
- **Why does `TrackingTestConfig` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 5`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._
- **Are the 51 inferred relationships involving `Detection` (e.g. with `Check if line segment a1-a2 intersects segment b1-b2.` and `Check if any part of the detection bbox overlaps with the zone polygon.`) actually correct?**
  _`Detection` has 51 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `RP2040SerialController` (e.g. with `TurretController` and `ControllerResponse`) actually correct?**
  _`RP2040SerialController` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Are the 36 inferred relationships involving `SupervisorLoop` (e.g. with `TurretController` and `SystemConfig`) actually correct?**
  _`SupervisorLoop` has 36 INFERRED edges - model-reasoned connections that need verification._
- **Are the 36 inferred relationships involving `DetectionPolicy` (e.g. with `YoloPrompt` and `VisionConfig`) actually correct?**
  _`DetectionPolicy` has 36 INFERRED edges - model-reasoned connections that need verification._