from __future__ import annotations

import argparse
import time

from cat_cannon.adapters.rp2040_discovery import RP2040DiscoveryError, autodetect_port
from cat_cannon.adapters.rp2040_serial import RP2040ProtocolError, RP2040SerialController
from cat_cannon.app.controller_session import ControllerSession


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RP2040 smoke test for Cat Cannon")
    parser.add_argument("--port", help="RP2040 serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--fire-ms", type=int, default=120)
    parser.add_argument("--step-deg", type=float, default=5.0)
    parser.add_argument("--move-delay-s", type=float, default=0.4)
    parser.add_argument("--dry-fire", action="store_true", help="Skip the fire command")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        port = args.port or autodetect_port()
    except RP2040DiscoveryError as exc:
        raise SystemExit(str(exc)) from exc

    controller = RP2040SerialController.open(
        port=port,
        baudrate=args.baudrate,
        fire_pulse_ms=args.fire_ms,
    )
    session = ControllerSession(controller=controller)
    try:
        print(f"[smoke] connecting to {port}")
        session.start()
        print(f"[smoke] handshake ok: {controller.status().payload}")

        print("[smoke] enabling controller")
        session.enable()
        print(f"[smoke] status: {controller.status().payload}")

        print("[smoke] moving left")
        controller.apply_tracking_delta(-args.step_deg, 0.0)
        time.sleep(args.move_delay_s)

        print("[smoke] moving right")
        controller.apply_tracking_delta(args.step_deg, 0.0)
        time.sleep(args.move_delay_s)

        print("[smoke] moving up")
        controller.apply_tracking_delta(0.0, -args.step_deg)
        time.sleep(args.move_delay_s)

        print("[smoke] moving down")
        controller.apply_tracking_delta(0.0, args.step_deg)
        time.sleep(args.move_delay_s)

        if args.dry_fire:
            print("[smoke] dry-fire enabled; skipping fire command")
        else:
            print("[smoke] firing once")
            controller.fire()

        print(f"[smoke] final status: {controller.status().payload}")
        print("[smoke] success")
    except RP2040ProtocolError as exc:
        raise SystemExit(f"Smoke test failed: {exc}") from exc
    finally:
        session.stop()


if __name__ == "__main__":
    main()
