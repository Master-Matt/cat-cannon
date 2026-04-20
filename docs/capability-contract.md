# Capability Contract

## Purpose

Cat Cannon is a fail-closed edge application that detects a cat standing on a kitchen counter,
tracks the cat with a pan/tilt turret, and conditionally triggers a bounded deterrent pulse.

## Operating Model

- Fixed camera owns scene understanding and safety.
- Turret camera owns fine centering.
- External controller owns servo PWM, solenoid timing, and actuator watchdog behavior.
- The supervisor owns all fire/no-fire decisions.

## Safety Invariants

- If a person is visible on the fixed camera, the system must not fire.
- If the system is not armed, the system must not fire.
- If counter confirmation is lost, the system must return to a non-firing state.
- If any fault condition is raised by perception, control, or orchestration, the system must enter
  `FAULT` and stop issuing fire commands.
- Solenoid control must be edge-triggered and bounded by the external controller.

## Counter Qualification

- Counter presence is modeled as calibrated polygons in the fixed camera image plane.
- A cat is considered on-counter when the cat footpoint lies within a counter polygon.
- A fireable target requires repeated positive confirmation across consecutive frames for the same
  tracked cat.

## Supervisor States

- `DISARMED`
- `IDLE`
- `HUMAN_LOCKOUT`
- `COUNTER_CONFIRMED`
- `TURRET_ACQUIRE`
- `TRACKING`
- `AIM_LOCK`
- `FIRE`
- `COOLDOWN`
- `FAULT`

## Hardware Boundaries

- The Jetson does not directly own precise actuator timing.
- Servo and solenoid outputs are sent as high-level commands to a dedicated controller.
- Emergency stop and hard limits remain hardware responsibilities even if software fails.

## Replay Requirement

- The decision core must be runnable without live cameras.
- Recorded or synthetic frame sequences must exercise:
  - counter confirmation
  - human lockout
  - aim acquisition
  - cooldown behavior

