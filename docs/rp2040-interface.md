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
- `safe_stop`
- `fire`

## Safety Behavior

- `fire` is rejected unless the controller is enabled
- watchdog timeout disables actuation if host heartbeats stop
- `safe_stop` always drops the solenoid output immediately
- servo motion is clamped to configured angle limits
