SERVO_FREQUENCY_HZ = 50

PAN_SERVO_PIN = 0
TILT_SERVO_PIN = 1
SOLENOID_PIN = 2
# D4184-style MOSFET driver input: GPIO low is off, GPIO high is fire.
SOLENOID_ACTIVE_LOW = False
STATUS_LED_PIN = "LED"

PAN_MIN_DEG = 0.0
PAN_MAX_DEG = 180.0
PAN_HOME_DEG = 90.0

TILT_MIN_DEG = 30.0
TILT_MAX_DEG = 150.0
TILT_HOME_DEG = 90.0

SERVO_MIN_US = 500
SERVO_MAX_US = 2500

# Max servo speed: degrees per second (60 deg in 0.2s = 300 deg/s)
SERVO_MAX_SPEED_DEG_S = 300.0

WATCHDOG_TIMEOUT_MS = 1500
DEFAULT_FIRE_DURATION_MS = 300
MAX_FIRE_DURATION_MS = 1000

# Servo idle relax: stop PWM after this many ms without movement to prevent buzzing
SERVO_IDLE_RELAX_MS = 2000
# Idle-relax behaviour. The tilt axis carries a gravity load, so holding it
# energized continuously draws constant torque/current and can overheat (and
# burn out) the servo over time. Relax tilt on idle like pan to protect the
# servo; the brief settle on the next move is an acceptable trade-off.
SERVO_IDLE_RELAX_TILT = True
SERVO_EPS_DEG = 0.01
