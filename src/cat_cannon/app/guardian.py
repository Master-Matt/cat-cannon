"""Out-of-process guardian: relaunches the app and escalates to reboot.

The guardian runs the Cat Cannon app as a child process. When the app exits with
the heartbeat sentinel code (self-detected hang / no-motion) or simply crashes,
the guardian records a fault against a :class:`RestartPolicy`, posts a Discord
notice, and either relaunches the app or — once restarts keep recurring — reboots
the machine. A clean exit (code 0, e.g. the operator pressed ``q``) stops the
guardian.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from cat_cannon.app.heartbeat import (
    SENTINEL_EXIT_CODE,
    RestartDecision,
    RestartPolicy,
    RestartPolicyConfig,
)
from cat_cannon.app.notify import post_discord_message
from cat_cannon.config import HeartbeatConfig, load_heartbeat_config

DEFAULT_APP_COMMAND: tuple[str, ...] = ("python", "-m", "cat_cannon.app.main")


@dataclass
class GuardianConfig:
    app_command: Sequence[str]
    restart_backoff_s: float = 3.0
    max_restarts: int | None = None  # None = unlimited


class Guardian:
    """Supervises the app process with restart/reboot escalation.

    All side-effecting collaborators are injectable so the escalation logic is
    fully unit-testable without spawning real processes or rebooting.
    """

    def __init__(
        self,
        config: GuardianConfig,
        heartbeat: HeartbeatConfig,
        *,
        runner: Callable[[Sequence[str]], int] | None = None,
        notifier: Callable[[str], bool] | None = None,
        rebooter: Callable[[], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
        policy: RestartPolicy | None = None,
    ) -> None:
        self._config = config
        self._heartbeat = heartbeat
        self._runner = runner or self._default_runner
        self._notifier = notifier or self._default_notifier
        self._rebooter = rebooter or self._default_rebooter
        self._sleep = sleep
        self._clock = clock
        self._policy = policy or RestartPolicy(
            RestartPolicyConfig(
                reboot_threshold=heartbeat.restart_reboot_threshold,
                window_s=heartbeat.restart_window_s,
            ),
            state_path=heartbeat.state_path,
        )

    # -- default collaborators -------------------------------------------------
    def _default_runner(self, command: Sequence[str]) -> int:
        completed = subprocess.run(list(command), check=False)
        return completed.returncode

    def _default_notifier(self, content: str) -> bool:
        return post_discord_message(
            self._heartbeat.resolved_discord_webhook_url(),
            content,
        )

    def _default_rebooter(self) -> None:
        if not self._heartbeat.reboot_enabled:
            print("[guardian] reboot requested but disabled by config", flush=True)
            return
        command = self._heartbeat.reboot_command.split()
        subprocess.run(command, check=False)

    # -- main loop -------------------------------------------------------------
    def run_forever(self) -> int:
        restarts = 0
        while True:
            exit_code = self._runner(self._config.app_command)

            if exit_code == 0:
                print("[guardian] app exited cleanly; stopping", flush=True)
                return 0

            reason = "missed heartbeat" if exit_code == SENTINEL_EXIT_CODE else "crash"
            decision = self._policy.record_fault(reason, now=self._clock())

            if decision is RestartDecision.REBOOT:
                self._notifier(
                    f"🔁 Cat Cannon keeps failing ({reason}); rebooting the machine."
                )
                self._rebooter()
                # If reboot is disabled or a no-op (tests), fall through to a
                # restart so we don't spin instantly.
                self._sleep(self._config.restart_backoff_s)
                if self._heartbeat.reboot_enabled:
                    return SENTINEL_EXIT_CODE
            else:
                self._notifier(
                    f"⚠️ Cat Cannon restarted (reason: {reason})."
                )

            restarts += 1
            if self._config.max_restarts is not None and restarts >= self._config.max_restarts:
                print("[guardian] max restarts reached; stopping", flush=True)
                return exit_code
            self._sleep(self._config.restart_backoff_s)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cat Cannon process guardian")
    parser.add_argument(
        "--config",
        default="configs/app.yaml",
        help="System config path (for the heartbeat: block)",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=3.0,
        help="Seconds to wait before relaunching the app",
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=None,
        help="Stop after this many restarts (default: unlimited)",
    )
    parser.add_argument(
        "app_command",
        nargs=argparse.REMAINDER,
        help="Command to run (default: python -m cat_cannon.app.main)",
    )
    return parser.parse_args(argv)


def _resolve_heartbeat_config(path: str) -> HeartbeatConfig:
    from cat_cannon.app.tracking_test import _resolve_config_path

    try:
        return load_heartbeat_config(_resolve_config_path(path))
    except (OSError, KeyError, ValueError):
        return HeartbeatConfig()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    app_command = args.app_command or list(DEFAULT_APP_COMMAND)
    if app_command and app_command[0] == "--":
        app_command = app_command[1:]
    if not app_command:
        app_command = list(DEFAULT_APP_COMMAND)
    heartbeat = _resolve_heartbeat_config(args.config)
    guardian = Guardian(
        GuardianConfig(
            app_command=app_command,
            restart_backoff_s=args.backoff,
            max_restarts=args.max_restarts,
        ),
        heartbeat,
    )
    return guardian.run_forever()


if __name__ == "__main__":
    sys.exit(main())
