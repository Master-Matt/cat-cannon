from __future__ import annotations

import argparse
import select
import sys
import termios
import tty
from dataclasses import dataclass

from cat_cannon.adapters.rp2040_discovery import RP2040DiscoveryError, autodetect_port
from cat_cannon.adapters.rp2040_serial import RP2040ProtocolError, RP2040SerialController
from cat_cannon.app.controller_session import ControllerSession


@dataclass(frozen=True)
class TeleopState:
    armed: bool
    step_deg: float


def handle_key(
    key: str,
    *,
    state: TeleopState,
    session: ControllerSession,
    controller: RP2040SerialController,
) -> tuple[TeleopState, str, bool]:
    if key == "q":
        return state, "quit requested", True

    if key == "e":
        session.enable()
        return TeleopState(armed=True, step_deg=state.step_deg), "armed", False

    if key == "x":
        session.disable()
        controller.safe_stop()
        return TeleopState(armed=False, step_deg=state.step_deg), "disarmed and safe-stopped", False

    if key == "p":
        status = controller.status().payload
        return state, f"status {status}", False

    if key == "f":
        controller.set_fire_output(False)
        return state, "fire output off", False

    if key in {"w", "a", "s", "d", " ", "r"} and not state.armed:
        return state, "controller is disarmed; press 'e' to arm first", False

    if key == "w":
        controller.apply_tracking_delta(0.0, -state.step_deg)
        return state, f"tilt up {state.step_deg:.1f} deg", False
    if key == "s":
        controller.apply_tracking_delta(0.0, state.step_deg)
        return state, f"tilt down {state.step_deg:.1f} deg", False
    if key == "a":
        controller.apply_tracking_delta(-state.step_deg, 0.0)
        return state, f"pan left {state.step_deg:.1f} deg", False
    if key == "d":
        controller.apply_tracking_delta(state.step_deg, 0.0)
        return state, f"pan right {state.step_deg:.1f} deg", False
    if key == "r":
        controller.set_fire_output(True)
        return state, "fire output on", False
    if key == " ":
        controller.fire()
        return state, "fired", False

    return state, "", False


def _print_help(state: TeleopState) -> None:
    print("[teleop] connected")
    print("[teleop] keys: e arm  x disarm  w/a/s/d move  space fire  r fire-on  f fire-off  p status  q quit")
    print(f"[teleop] step_deg={state.step_deg:.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RP2040 keyboard teleop for SSH sessions")
    parser.add_argument("--port", help="RP2040 serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--fire-ms", type=int, default=120)
    parser.add_argument("--step-deg", type=float, default=3.0)
    parser.add_argument("--arm-on-start", action="store_true", help="Arm the controller immediately")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not sys.stdin.isatty():
        raise SystemExit("Teleop requires an interactive TTY. SSH in with a terminal and rerun it.")

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
    state = TeleopState(armed=False, step_deg=args.step_deg)

    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        print(f"[teleop] connecting to {port}")
        session.start()
        if args.arm_on_start:
            session.enable()
            state = TeleopState(armed=True, step_deg=state.step_deg)
        else:
            session.disable()
        _print_help(state)
        print(f"[teleop] status {controller.status().payload}")

        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.25)
            if not ready:
                continue
            key = sys.stdin.read(1)
            if not key:
                continue
            try:
                state, message, should_exit = handle_key(
                    key,
                    state=state,
                    session=session,
                    controller=controller,
                )
            except RP2040ProtocolError as exc:
                print(f"[teleop] error: {exc}")
                continue
            if message:
                print(f"[teleop] {message}")
            if should_exit:
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)
        session.stop()


if __name__ == "__main__":
    main()
