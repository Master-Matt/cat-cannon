# RP2040 Interface

The Jetson communicates with the Raspberry Pi Pico over USB CDC serial using newline-delimited
JSON messages.

## Transport Choice

Chosen transport: `wired USB serial`

Reasons:

- lower operational risk than Wi-Fi for a safety-critical actuation path
- easier device discovery and logging on the Jetson
- simple firmware deployment over the same USB serial link
- easier recovery after a bad update

## Request Format

```json
{"seq":1,"command":"ping","payload":{}}
```

## Response Format

```json
{"ok":true,"seq":1,"status":"pong","payload":{"enabled":false,"pan_deg":90.0,"tilt_deg":90.0,"solenoid_active":false}}
```

## Supported Commands

- `ping`
- `heartbeat`
- `status`
- `set_enabled`
- `set_angles`
- `apply_delta`
- `set_servo_limits`
- `set_velocity`
- `relax`
- `safe_stop`
- `set_fire_output`
- `fire`

## Safety Behavior

- `fire` is rejected unless the controller is enabled
- watchdog timeout disables actuation if host heartbeats stop
- `safe_stop` always drops the solenoid output immediately
- servo motion is clamped to configured angle limits
- guided limits from the tracking screen are sent to the Pico before live movement
- `set_fire_output` exists for bench debugging; normal operation should use bounded `fire` pulses

## Current Pico Pin Assumptions

`firmware/pico/pico_config.py` is the source of truth for board wiring:

- pan servo: GPIO0
- tilt servo: GPIO1
- solenoid/MOSFET signal: GPIO2
- solenoid output mode: active high for the current D4184-style low-side MOSFET module

The Pico should be redeployed from the host it is attached to. In the current Jetson setup, that
means running `./scripts/deploy_pico.sh --port /dev/ttyACM0` on the Jetson.
