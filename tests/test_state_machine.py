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

