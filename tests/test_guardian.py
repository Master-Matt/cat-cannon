from cat_cannon.app.guardian import Guardian, GuardianConfig
from cat_cannon.app.heartbeat import (
    SENTINEL_EXIT_CODE,
    RestartPolicy,
    RestartPolicyConfig,
)
from cat_cannon.config import HeartbeatConfig


class _Recorder:
    def __init__(self, exit_codes):
        self.exit_codes = list(exit_codes)
        self.runs = 0
        self.messages = []
        self.reboots = 0
        self.sleeps = []
        self._t = 0.0

    def runner(self, command):
        self.runs += 1
        if self.exit_codes:
            return self.exit_codes.pop(0)
        return 0

    def notifier(self, content):
        self.messages.append(content)
        return True

    def rebooter(self):
        self.reboots += 1

    def sleep(self, seconds):
        self.sleeps.append(seconds)

    def clock(self):
        self._t += 1.0
        return self._t


def _guardian(rec, heartbeat=None, **cfg):
    heartbeat = heartbeat or HeartbeatConfig()
    # In-memory policy (no state_path) so tests are hermetic and never touch the
    # real ~/.cache restart-state file.
    policy = RestartPolicy(
        RestartPolicyConfig(
            reboot_threshold=heartbeat.restart_reboot_threshold,
            window_s=heartbeat.restart_window_s,
        )
    )
    return Guardian(
        GuardianConfig(app_command=["echo", "hi"], restart_backoff_s=0.0, **cfg),
        heartbeat,
        runner=rec.runner,
        notifier=rec.notifier,
        rebooter=rec.rebooter,
        sleep=rec.sleep,
        clock=rec.clock,
        policy=policy,
    )


def test_clean_exit_stops_guardian():
    rec = _Recorder([0])
    rc = _guardian(rec).run_forever()
    assert rc == 0
    assert rec.runs == 1
    assert rec.messages == []


def test_sentinel_triggers_restart_and_notify():
    rec = _Recorder([SENTINEL_EXIT_CODE, 0])
    rc = _guardian(rec).run_forever()
    assert rc == 0
    assert rec.runs == 2
    assert any("missed heartbeat" in m for m in rec.messages)
    assert rec.reboots == 0


def test_crash_triggers_restart_with_crash_reason():
    rec = _Recorder([1, 0])
    rc = _guardian(rec).run_forever()
    assert rc == 0
    assert any("crash" in m for m in rec.messages)


def test_escalates_to_reboot_after_threshold():
    heartbeat = HeartbeatConfig(
        restart_reboot_threshold=3,
        restart_window_s=600.0,
        reboot_enabled=True,
    )
    rec = _Recorder([SENTINEL_EXIT_CODE, SENTINEL_EXIT_CODE, SENTINEL_EXIT_CODE])
    rc = _guardian(rec, heartbeat=heartbeat).run_forever()
    assert rc == SENTINEL_EXIT_CODE
    assert rec.reboots == 1
    assert any("rebooting" in m for m in rec.messages)


def test_max_restarts_stops_loop():
    rec = _Recorder([1, 1, 1, 1, 1])
    rc = _guardian(rec, max_restarts=2).run_forever()
    assert rec.runs == 2
    assert rc == 1
