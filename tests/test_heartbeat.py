from cat_cannon.app.heartbeat import (
    SENTINEL_EXIT_CODE,
    FaultReason,
    Liveness,
    MotionConfig,
    MotionWatchdog,
    RestartDecision,
    RestartPolicy,
    RestartPolicyConfig,
)


def test_sentinel_exit_code_is_stable():
    assert SENTINEL_EXIT_CODE == 70


def test_fault_reasons():
    assert FaultReason.HANG.value == "hang"
    assert FaultReason.NO_MOTION.value == "no_motion"


def test_liveness_detects_stale_loop():
    live = Liveness(timeout_s=15.0, now=100.0)
    assert not live.is_stale(now=110.0)
    assert not live.is_stale(now=115.0)
    assert live.is_stale(now=116.0)
    live.beat(now=200.0)
    assert not live.is_stale(now=210.0)
    assert live.age(now=205.0) == 5.0


def _watchdog(**overrides):
    base = dict(
        flow_min_magnitude_px=0.6,
        direction_dot_min=0.15,
        max_consecutive_failures=3,
        pan_flow_sign=1,
        tilt_flow_sign=1,
    )
    base.update(overrides)
    return MotionWatchdog(MotionConfig(**base))


def test_motion_confirmed_when_flow_matches_command():
    wd = _watchdog()
    assert wd.is_confirmed(pan_cmd=6.0, tilt_cmd=0.0, dx=4.0, dy=0.2)


def test_motion_failed_when_no_flow():
    wd = _watchdog()
    assert not wd.is_confirmed(pan_cmd=6.0, tilt_cmd=0.0, dx=0.1, dy=0.0)


def test_motion_failed_when_flow_opposes_command():
    wd = _watchdog()
    assert not wd.is_confirmed(pan_cmd=6.0, tilt_cmd=0.0, dx=-4.0, dy=0.0)


def test_motion_noop_command_never_fails():
    wd = _watchdog()
    assert wd.is_confirmed(pan_cmd=0.0, tilt_cmd=0.0, dx=0.0, dy=0.0)


def test_motion_flow_sign_inversion():
    wd = _watchdog(tilt_flow_sign=-1)
    # commanded tilt up (+3) but camera flow goes -y -> with sign -1 it matches
    assert wd.is_confirmed(pan_cmd=0.0, tilt_cmd=3.0, dx=0.0, dy=-4.0)
    assert not wd.is_confirmed(pan_cmd=0.0, tilt_cmd=3.0, dx=0.0, dy=4.0)


def test_motion_watchdog_trips_after_consecutive_failures():
    wd = _watchdog(max_consecutive_failures=3)
    for _ in range(2):
        wd.record(pan_cmd=6.0, tilt_cmd=0.0, dx=0.0, dy=0.0)
    assert not wd.tripped
    wd.record(pan_cmd=6.0, tilt_cmd=0.0, dx=0.0, dy=0.0)
    assert wd.tripped
    assert wd.consecutive_failures == 3


def test_motion_watchdog_resets_on_confirm():
    wd = _watchdog()
    wd.record(pan_cmd=6.0, tilt_cmd=0.0, dx=0.0, dy=0.0)
    wd.record(pan_cmd=6.0, tilt_cmd=0.0, dx=0.0, dy=0.0)
    assert wd.consecutive_failures == 2
    confirmed = wd.record(pan_cmd=6.0, tilt_cmd=0.0, dx=5.0, dy=0.0)
    assert confirmed
    assert wd.consecutive_failures == 0
    assert not wd.tripped


def test_restart_policy_restarts_below_threshold():
    policy = RestartPolicy(RestartPolicyConfig(reboot_threshold=3, window_s=600.0))
    assert policy.record_fault("crash", now=0.0) is RestartDecision.RESTART
    assert policy.record_fault("crash", now=10.0) is RestartDecision.RESTART


def test_restart_policy_escalates_to_reboot():
    policy = RestartPolicy(RestartPolicyConfig(reboot_threshold=3, window_s=600.0))
    policy.record_fault("crash", now=0.0)
    policy.record_fault("crash", now=10.0)
    assert policy.record_fault("crash", now=20.0) is RestartDecision.REBOOT
    # history cleared after reboot decision -> next fault is a plain restart
    assert policy.record_fault("crash", now=30.0) is RestartDecision.RESTART


def test_restart_policy_window_prunes_old_faults():
    policy = RestartPolicy(RestartPolicyConfig(reboot_threshold=3, window_s=600.0))
    policy.record_fault("crash", now=0.0)
    policy.record_fault("crash", now=10.0)
    # Third fault is outside the window from the first two -> stays a restart
    assert policy.record_fault("crash", now=700.0) is RestartDecision.RESTART


def test_restart_policy_persists_to_disk(tmp_path):
    state = tmp_path / "restart-state.json"
    config = RestartPolicyConfig(reboot_threshold=3, window_s=600.0)
    p1 = RestartPolicy(config, state_path=str(state))
    p1.record_fault("crash", now=100.0)
    p1.record_fault("crash", now=110.0)
    # Reload from disk: the two faults should still count toward the threshold
    p2 = RestartPolicy(config, state_path=str(state))
    assert len(p2.history) == 2
    assert p2.record_fault("crash", now=120.0) is RestartDecision.REBOOT
