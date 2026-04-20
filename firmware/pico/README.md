# Pico Firmware

This firmware targets a standard Raspberry Pi Pico running MicroPython and exposes a
USB serial JSON protocol to the Jetson or laptop host.

## Why USB Serial

- simpler than any wireless link for a safety-critical actuator path
- deterministic device discovery on the Jetson or laptop
- easier field recovery and bench debugging
- uses the same USB serial cable for deployment and runtime control

## Expected Pins

- `pan_servo_pin`
- `tilt_servo_pin`
- `solenoid_pin`
- optional onboard `status_led_pin`

Update `pico_config.py` before deployment if your wiring differs.

## Included UF2

This repo includes the latest plain-Pico MicroPython UF2 that this project is targeting:

- `RPI_PICO-20260406-v1.28.0.uf2`

Official source:

- https://micropython.org/download/RPI_PICO/

## First-Time Provisioning

1. Put the Pico into `BOOTSEL` mode.
2. Flash the bundled MicroPython UF2:

```bash
./scripts/flash_pico_uf2.sh
```

3. After the board reboots into MicroPython, deploy the application files:

```bash
./scripts/deploy_pico.sh
```
