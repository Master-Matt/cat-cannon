from cat_cannon.domain.models import SupervisorState
from cat_cannon.domain.state_machine import SupervisorInputs, SupervisorStateMachine


def test_state_machine_advances_to_fire_then_cooldown() -> None:
    machine = SupervisorStateMachine(cooldown_frames=2)

    machine.advance(
        SupervisorInputs(
            armed=True,
            human_present=False,
            counter_confirmed=True,
            target_visible=False,
            aim_locked=False,
        )
    )
    tracking = machine.advance(
        SupervisorInputs(
            armed=True,
            human_present=False,
            counter_confirmed=True,
            target_visible=True,
            aim_locked=False,
        )
    )
    assert tracking.state == SupervisorState.TRACKING

    fire = machine.advance(
        SupervisorInputs(
            armed=True,
            human_present=False,
            counter_confirmed=True,
            target_visible=True,
            aim_locked=True,
        )
    )
    assert fire.fire_commanded is True
    assert fire.state == SupervisorState.FIRE

    cooldown = machine.advance(
        SupervisorInputs(
            armed=True,
            human_present=False,
            counter_confirmed=True,
            target_visible=False,
            aim_locked=False,
        )
    )
    assert cooldown.state == SupervisorState.COOLDOWN


def test_state_machine_does_not_fire_when_aim_lock_lacks_fire_permission() -> None:
    machine = SupervisorStateMachine(cooldown_frames=2)

    machine.advance(
        SupervisorInputs(
            armed=True,
            human_present=False,
            counter_confirmed=True,
            target_visible=False,
            aim_locked=False,
        )
    )
    machine.advance(
        SupervisorInputs(
            armed=True,
            human_present=False,
            counter_confirmed=True,
            target_visible=True,
            aim_locked=False,
        )
    )
    aim_lock_without_permission = machine.advance(
        SupervisorInputs(
            armed=True,
            human_present=False,
            counter_confirmed=True,
            target_visible=True,
            aim_locked=True,
            fire_permitted=False,
        )
    )
    assert aim_lock_without_permission.state == SupervisorState.AIM_LOCK
    assert aim_lock_without_permission.fire_commanded is False

    lost_lock = machine.advance(
        SupervisorInputs(
            armed=True,
            human_present=False,
            counter_confirmed=True,
            target_visible=True,
            aim_locked=False,
            fire_permitted=False,
        )
    )

    assert lost_lock.fire_commanded is False
    assert lost_lock.state == SupervisorState.TRACKING


def test_state_machine_enters_human_lockout_immediately() -> None:
    machine = SupervisorStateMachine(cooldown_frames=5)

    result = machine.advance(
        SupervisorInputs(
            armed=True,
            human_present=True,
            counter_confirmed=True,
            target_visible=True,
            aim_locked=True,
        )
    )

    assert result.state == SupervisorState.HUMAN_LOCKOUT
    assert result.fire_commanded is False


def test_state_machine_can_use_wall_clock_fire_cooldown() -> None:
    machine = SupervisorStateMachine(cooldown_frames=99, cooldown_seconds=0.5)
    locked = SupervisorInputs(
        armed=True,
        human_present=False,
        counter_confirmed=True,
        target_visible=True,
        aim_locked=True,
    )

    assert machine.advance(locked, now=0.0).fire_commanded is False
    assert machine.advance(locked, now=0.1).fire_commanded is True
    assert machine.advance(locked, now=0.2).fire_commanded is False
    assert machine.advance(locked, now=0.3).fire_commanded is False
    assert machine.advance(locked, now=0.59).state == SupervisorState.COOLDOWN
    assert machine.advance(locked, now=0.6).fire_commanded is True


def test_state_machine_fires_configured_burst_after_aim_lock() -> None:
    machine = SupervisorStateMachine(
        cooldown_frames=99,
        cooldown_seconds=0.5,
        burst_count=3,
        burst_interval_seconds=0.5,
    )
    locked = SupervisorInputs(
        armed=True,
        human_present=False,
        counter_confirmed=True,
        target_visible=True,
        aim_locked=True,
    )

    results = [
        machine.advance(locked, now=0.0),
        machine.advance(locked, now=0.1),
        machine.advance(locked, now=0.2),
        machine.advance(locked, now=0.59),
        machine.advance(locked, now=0.6),
        machine.advance(locked, now=0.7),
        machine.advance(locked, now=1.09),
        machine.advance(locked, now=1.1),
    ]

    assert [result.fire_commanded for result in results] == [
        False,
        True,
        False,
        False,
        True,
        False,
        False,
        True,
    ]


def test_state_machine_stops_burst_if_target_disappears() -> None:
    machine = SupervisorStateMachine(
        cooldown_frames=99,
        cooldown_seconds=0.5,
        burst_count=3,
        burst_interval_seconds=0.5,
    )
    locked = SupervisorInputs(
        armed=True,
        human_present=False,
        counter_confirmed=True,
        target_visible=True,
        aim_locked=True,
    )
    missing = SupervisorInputs(
        armed=True,
        human_present=False,
        counter_confirmed=False,
        target_visible=False,
        aim_locked=False,
    )

    assert machine.advance(locked, now=0.0).fire_commanded is False
    assert machine.advance(locked, now=0.1).fire_commanded is True
    assert machine.advance(missing, now=0.2).fire_commanded is False
    assert machine.advance(missing, now=0.7).fire_commanded is False


def test_state_machine_does_not_start_fire_without_aim_lock_even_when_permitted() -> None:
    machine = SupervisorStateMachine(cooldown_frames=99, cooldown_seconds=0.5)
    moving_on_counter = SupervisorInputs(
        armed=True,
        human_present=False,
        counter_confirmed=True,
        target_visible=True,
        aim_locked=False,
        fire_permitted=True,
    )

    assert machine.advance(moving_on_counter, now=0.0).fire_commanded is False
    assert machine.advance(moving_on_counter, now=0.1).fire_commanded is False
    assert machine.advance(moving_on_counter, now=0.2).state == SupervisorState.TRACKING


def test_state_machine_fires_on_first_locked_frame_when_fire_is_permitted() -> None:
    machine = SupervisorStateMachine(cooldown_frames=99, cooldown_seconds=0.5)
    moving = SupervisorInputs(
        armed=True,
        human_present=False,
        counter_confirmed=True,
        target_visible=True,
        aim_locked=False,
        fire_permitted=False,
    )
    locked = SupervisorInputs(
        armed=True,
        human_present=False,
        counter_confirmed=True,
        target_visible=True,
        aim_locked=True,
        fire_permitted=True,
    )

    assert machine.advance(moving, now=0.0).fire_commanded is False
    assert machine.advance(moving, now=0.1).state == SupervisorState.TRACKING
    fire = machine.advance(locked, now=0.2)

    assert fire.fire_commanded is True
    assert fire.state == SupervisorState.FIRE


def test_state_machine_leaves_cooldown_tracking_when_lock_is_lost() -> None:
    machine = SupervisorStateMachine(cooldown_frames=99, cooldown_seconds=0.5)
    locked = SupervisorInputs(
        armed=True,
        human_present=False,
        counter_confirmed=True,
        target_visible=True,
        aim_locked=True,
        fire_permitted=True,
    )
    tracking = SupervisorInputs(
        armed=True,
        human_present=False,
        counter_confirmed=True,
        target_visible=True,
        aim_locked=False,
        fire_permitted=False,
    )

    assert machine.advance(locked, now=0.0).fire_commanded is False
    assert machine.advance(locked, now=0.1).fire_commanded is True
    assert machine.advance(tracking, now=0.2).state == SupervisorState.COOLDOWN
    assert machine.advance(tracking, now=0.6).state == SupervisorState.TRACKING


def test_state_machine_does_not_fire_while_moving_without_fire_permission() -> None:
    machine = SupervisorStateMachine(cooldown_frames=99, cooldown_seconds=0.5)
    moving_without_permission = SupervisorInputs(
        armed=True,
        human_present=False,
        counter_confirmed=True,
        target_visible=True,
        aim_locked=False,
        fire_permitted=False,
    )

    assert machine.advance(moving_without_permission, now=0.0).fire_commanded is False
    assert machine.advance(moving_without_permission, now=0.1).fire_commanded is False
