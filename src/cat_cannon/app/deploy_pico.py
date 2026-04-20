from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cat_cannon.adapters.micropython_deploy import MicroPythonDeployError, deploy_files
from cat_cannon.adapters.rp2040_discovery import RP2040DiscoveryError, autodetect_port


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy Pico MicroPython application files.")
    parser.add_argument("--port", help="Explicit serial port, e.g. /dev/ttyACM0")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    port = args.port
    if not port:
        try:
            port = autodetect_port()
        except RP2040DiscoveryError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    firmware_dir = _repo_root() / "firmware" / "pico"
    files = [
        ("pico_config.py", firmware_dir / "pico_config.py"),
        ("main.py", firmware_dir / "main.py"),
    ]

    try:
        deploy_files(port=port, files=files)
    except (MicroPythonDeployError, OSError) as exc:
        print(f"Pico deployment failed: {exc}", file=sys.stderr)
        return 1

    print(f"[pico] Deployment complete on {port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
