#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
IMGSZ="${1:-}"
CONFIG_PATH="${CAT_CANNON_CONFIG:-configs/app.yaml}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python3}"
fi

if [[ "$CONFIG_PATH" == "configs/app.yaml" && ! -f "$ROOT_DIR/$CONFIG_PATH" ]]; then
  CONFIG_PATH="configs/app.example.yaml"
fi

PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" - "$ROOT_DIR" "$CONFIG_PATH" "$IMGSZ" <<'PY'
from __future__ import annotations

from pathlib import Path
import shutil
import sys
import time

from cat_cannon.config import load_vision_config
from ultralytics import YOLO, YOLOE


def _resolve_model_path(models_dir: Path, model_name: str, fallback: str) -> str:
    name = model_name or fallback
    model_path = Path(name)
    if model_path.is_absolute() or model_path.exists():
        return str(model_path)
    local_model = models_dir / model_path.name
    if local_model.exists():
        return str(local_model)
    return name


root_dir = Path(sys.argv[1])
config_path = Path(sys.argv[2])
imgsz_override = sys.argv[3]
vision = load_vision_config(root_dir / config_path)
imgsz = int(imgsz_override) if imgsz_override else vision.yolo_imgsz
models_dir = root_dir / "src" / "cat_cannon" / "models"

if vision.yolo_detector == "yoloe":
    model_name = vision.yoloe_model or "yoloe-11s-seg.pt"
    model_path = _resolve_model_path(models_dir, model_name, "yoloe-11s-seg.pt")
    engine_path = models_dir / f"{Path(model_name).stem}.engine"
    prompt_texts = list(vision.prompt_texts)
    if not prompt_texts:
        raise SystemExit("[engine] YOLOE selected but no prompts were configured")
    print(f"[engine] Exporting YOLOE TensorRT engine from {model_name} at imgsz={imgsz}")
    print(f"[engine] YOLOE prompts: {', '.join(prompt_texts)}")
    model = YOLOE(model_path)
    model.set_classes(prompt_texts)
else:
    model_name = vision.yolo_model or "yolo11s.pt"
    model_path = _resolve_model_path(models_dir, model_name, "yolo11s.pt")
    engine_path = models_dir / f"{Path(model_name).stem}.engine"
    print(f"[engine] Exporting TensorRT engine from {model_name} at imgsz={imgsz}")
    model = YOLO(model_path, task="detect")

if engine_path.exists():
    backup_path = engine_path.with_name(f"{engine_path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    print(f"[engine] Backing up existing engine to {backup_path}")
    shutil.copy2(engine_path, backup_path)
    engine_path.unlink()

exported = Path(
    model.export(
        format="engine",
        imgsz=imgsz,
        device=0,
        half=True,
        batch=1,
        simplify=True,
    )
)
if exported.resolve() != engine_path.resolve():
    if not exported.exists():
        raise SystemExit(f"[engine] Export reported missing engine: {exported}")
    shutil.move(str(exported), str(engine_path))

if not engine_path.exists():
    raise SystemExit(f"[engine] Export completed but expected engine was not created: {engine_path}")

print(f"[engine] Exported {engine_path}")
print(f"[engine] Size: {engine_path.stat().st_size} bytes")
PY
