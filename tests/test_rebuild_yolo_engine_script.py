from pathlib import Path


def test_rebuild_yolo_engine_script_exports_configurable_tensor_rt_engine() -> None:
    script = Path("scripts/rebuild_yolo_engine.sh").read_text(encoding="utf-8")

    assert 'IMGSZ="${1:-}"' in script
    assert "load_vision_config" in script
    assert 'format="engine"' in script
    assert "imgsz=imgsz" in script
    assert "yolo11s.pt" in script
    assert "engine_path" in script
    assert "YOLOE" in script
    assert "set_classes" in script
    assert "yoloe-11s-seg.pt" in script
