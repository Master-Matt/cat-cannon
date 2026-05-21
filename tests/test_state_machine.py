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

    aim_lock = machine.advance(
        SupervisorInputs(
            armed=True,
            human_present=False,
            counter_confirmed=True,
            target_visible=True,
            aim_locked=True,
        )
    )
    assert aim_lock.state == SupervisorState.AIM_LOCK
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


def test_state_machine_does_not_fire_if_aim_lock_is_lost_before_fire() -> None:
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
    aim_lock = machine.advance(
        SupervisorInputs(
            armed=True,
            human_present=False,
            counter_confirmed=True,
            target_visible=True,
            aim_locked=True,
        )
    )
    assert aim_lock.state == SupervisorState.AIM_LOCK

    lost_lock = machine.advance(
        SupervisorInputs(
            armed=True,
            human_present=False,
            counter_confirmed=True,
            target_visible=True,
            aim_locked=False,
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
    assert machine.advance(locked, now=0.1).fire_commanded is False
    assert machine.advance(locked, now=0.2).fire_commanded is True
    assert machine.advance(locked, now=0.3).fire_commanded is False
    assert machine.advance(locked, now=0.69).state == SupervisorState.COOLDOWN
    assert machine.advance(locked, now=0.7).state == SupervisorState.AIM_LOCK
    assert machine.advance(locked, now=0.71).fire_commanded is True
