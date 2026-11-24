"""Autonomous classroom attendance robot controller for Webots.

Uses Pioneer 3-DX wheel motors and sonar sensors. The controller is a Webots
Supervisor so it can read the robot's simulated pose directly; no GPS device is
required. It follows a serpentine route, stops at each of nine seat waypoints,
turns its camera toward the desk, captures a frame when possible, and writes a
CSV attendance log. Attendance is simulated by reaching a seat waypoint; this
script does NOT perform face recognition.
"""

from controller import Supervisor
import csv
import math
import os
import time

# ------------------------ Navigation configuration -------------------------
MAX_WHEEL_SPEED = 5.0       # rad/s; conservative for Pioneer 3-DX
TURN_WHEEL_SPEED = 1.9       # rad/s for point turns
WHEEL_RADIUS = 0.0975        # m, approximate Pioneer 3-DX wheel radius
WHEEL_BASE = 0.33            # m, approximate distance between wheel centers
ARRIVAL_RADIUS = 0.18        # m
HEADING_TOLERANCE = 0.16     # rad
LINEAR_SPEED_MAX = 0.32      # m/s
LINEAR_GAIN = 0.65
ANGULAR_GAIN = 1.5
SCAN_DWELL_SECONDS = 1.8
OBSTACLE_DISTANCE_M = 0.28
SONAR_MAX_RANGE_M = 5.0

# Coordinates use Webots ground plane (x, z), metres.
# None means transit waypoint; seat IDs trigger the simulated attendance scan.
WAYPOINTS = [
    (-3.0, -3.0, "S01"),
    ( 0.0, -3.0, "S02"),
    ( 3.0, -3.0, "S03"),
    ( 4.2, -3.0, None),
    ( 4.2, -1.0, None),
    ( 3.0, -1.0, "S06"),
    ( 0.0, -1.0, "S05"),
    (-3.0, -1.0, "S04"),
    (-4.2, -1.0, None),
    (-4.2,  1.0, None),
    (-3.0,  1.0, "S07"),
    ( 0.0,  1.0, "S08"),
    ( 3.0,  1.0, "S09"),
    ( 4.2,  1.0, None),
]

ROSTER = {
    "S01": "Student 01", "S02": "Student 02", "S03": "Student 03",
    "S04": "Student 04", "S05": "Student 05", "S06": "Student 06",
    "S07": "Student 07", "S08": "Student 08", "S09": "Student 09",
}


def clamp(value, low, high):
    return max(low, min(high, value))


def wrap_angle(angle):
    """Return an angle in [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def safe_get_device(robot, names):
    """Try supported Webots device-name variants without stopping the controller."""
    for name in names:
        try:
            device = robot.getDevice(name)
            if device is not None:
                return device, name
        except Exception:
            pass
    return None, None


def device_yaw(rotation):
    """Extract signed yaw from a Webots axis-angle rotation (y-axis rotation)."""
    axis_y = rotation[1]
    angle = rotation[3]
    if abs(axis_y) < 0.5:
        return angle
    return wrap_angle(angle * (1.0 if axis_y >= 0.0 else -1.0))


def set_wheels(left_motor, right_motor, left_speed, right_speed):
    left_motor.setVelocity(clamp(left_speed, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))
    right_motor.setVelocity(clamp(right_speed, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))


def stop_robot(left_motor, right_motor):
    set_wheels(left_motor, right_motor, 0.0, 0.0)


def sonar_distance_m(sensor):
    """Convert standard Pioneer sonar reading to approximate metres.

    The standard Pioneer 3-DX sonar commonly maps 0..1000 to 5..0 m
    (larger readings mean a closer object). If a sensor exposes small metric
    values instead, interpret the returned value directly as metres.
    """
    try:
        value = float(sensor.getValue())
    except Exception:
        return SONAR_MAX_RANGE_M
    try:
        max_value = float(sensor.getMaxValue())
    except Exception:
        max_value = 1000.0
    if max_value > 10.0:
        return max(0.0, (max_value - value) / max_value * SONAR_MAX_RANGE_M)
    return max(0.0, value)


def save_camera_ppm(camera, directory, seat_id):
    """Save an optional Webots BGRA camera frame as a portable PPM image."""
    if camera is None:
        return ""
    try:
        raw = camera.getImage()
        width = int(camera.getWidth())
        height = int(camera.getHeight())
        if not raw or width <= 0 or height <= 0:
            return ""
        raw = bytes(raw)
        expected = width * height * 4
        if len(raw) < expected:
            return ""
        rgb = bytearray(width * height * 3)
        out_i = 0
        for in_i in range(0, expected, 4):
            # Webots camera image bytes are BGRA.
            b, g, r = raw[in_i], raw[in_i + 1], raw[in_i + 2]
            rgb[out_i] = r
            rgb[out_i + 1] = g
            rgb[out_i + 2] = b
            out_i += 3
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, f"{seat_id}.ppm")
        with open(filename, "wb") as image_file:
            image_file.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            image_file.write(rgb)
        return filename
    except Exception as exc:
        print(f"[Camera] Could not save frame for {seat_id}: {exc}")
        return ""


# ----------------------------- Start controller ----------------------------
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

def find_motor(robot, names):
    """Find a usable motor by probing its speed-control methods."""
    for name in names:
        try:
            motor = robot.getDevice(name)
            if motor is None:
                continue
            # Probe the handle here so a missing name can fall through to alias.
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)
            return motor, name
        except Exception:
            continue
    return None, None


left_motor, left_name = find_motor(
    robot, ["left wheel motor", "left wheel"])
right_motor, right_name = find_motor(
    robot, ["right wheel motor", "right wheel"])

if left_motor is None or right_motor is None:
    raise RuntimeError(
        "Could not find usable Pioneer wheel motors. Expected 'left wheel motor' "
        "and 'right wheel motor' (or 'left wheel'/'right wheel'). Check the robot PROTO."
    )

left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
stop_robot(left_motor, right_motor)
print(f"[Attendance] Wheel devices: {left_name}, {right_name}")

# Pioneer 3-DX sonar names are so0..so15. Also accept ds0..ds15 for custom models.
sonars = {}
for i in range(16):
    sensor, sensor_name = safe_get_device(robot, [f"so{i}", f"ds{i}"])
    if sensor is not None:
        try:
            sensor.enable(timestep)
            sonars[sensor_name] = sensor
        except Exception:
            pass

# The center/front sonar is so3 on the standard Pioneer 3-DX.
front_sonar_names = [name for name in ("so2", "so3", "so4", "ds2", "ds3", "ds4")
                     if name in sonars]
left_sonar_names = [name for name in ("so4", "so5", "so6", "ds4", "ds5", "ds6")
                    if name in sonars]
right_sonar_names = [name for name in ("so1", "so2", "so3", "ds1", "ds2", "ds3")
                     if name in sonars]

camera, camera_name = safe_get_device(robot, ["attendance camera", "camera"])
if camera is not None:
    try:
        camera.enable(timestep)
        print(f"[Attendance] Camera enabled: {camera_name}")
    except Exception as exc:
        print(f"[Camera] Camera found but could not be enabled: {exc}")
        camera = None
else:
    print("[Camera] No camera device found; attendance waypoint logging will still run.")

self_node = robot.getSelf()
if self_node is None:
    raise RuntimeError("Supervisor could not access the robot's own node.")
translation_field = self_node.getField("translation")
rotation_field = self_node.getField("rotation")

controller_dir = os.path.dirname(os.path.abspath(__file__))
captures_dir = os.path.join(controller_dir, "captures")
log_path = os.path.join(controller_dir, "attendance_log.csv")
log_has_header = os.path.exists(log_path) and os.path.getsize(log_path) > 0
log_file = open(log_path, "a", newline="", encoding="utf-8")
log_writer = csv.writer(log_file)
if not log_has_header:
    log_writer.writerow(["simulation_time_s", "student_id", "student_name",
                         "status", "camera_snapshot"])
    log_file.flush()


def get_pose():
    pos = translation_field.getSFVec3f()
    rot = rotation_field.getSFRotation()
    return float(pos[0]), float(pos[2]), device_yaw(rot)


def sensor_distance(names):
    distances = []
    for name in names:
        if name in sonars:
            distances.append(sonar_distance_m(sonars[name]))
    return min(distances) if distances else SONAR_MAX_RANGE_M


def rotate_toward(target_yaw, current_yaw):
    error = wrap_angle(target_yaw - current_yaw)
    if abs(error) <= HEADING_TOLERANCE:
        stop_robot(left_motor, right_motor)
        return True
    # Positive Webots yaw is produced by right wheel faster than left.
    speed = TURN_WHEEL_SPEED if error > 0.0 else -TURN_WHEEL_SPEED
    set_wheels(left_motor, right_motor, -speed, speed)
    return False


def drive_to_target(target_x, target_z, x, z, yaw):
    dx = target_x - x
    dz = target_z - z
    distance = math.hypot(dx, dz)
    if distance <= ARRIVAL_RADIUS:
        stop_robot(left_motor, right_motor)
        return True

    # Pioneer front is local -Z; this converts a desired (dx,dz) vector to yaw.
    desired_yaw = math.atan2(-dx, -dz)
    heading_error = wrap_angle(desired_yaw - yaw)

    # Emergency sonar override: stop and turn toward the clearer side.
    nearest_front = sensor_distance(front_sonar_names)
    if nearest_front < OBSTACLE_DISTANCE_M:
        left_clear = sensor_distance(left_sonar_names)
        right_clear = sensor_distance(right_sonar_names)
        turn_sign = 1.0 if left_clear >= right_clear else -1.0
        turn_speed = TURN_WHEEL_SPEED * turn_sign
        set_wheels(left_motor, right_motor, -turn_speed, turn_speed)
        return False

    if abs(heading_error) > 0.42:
        return rotate_toward(desired_yaw, yaw)

    v = min(LINEAR_SPEED_MAX, LINEAR_GAIN * distance)
    omega = ANGULAR_GAIN * heading_error
    left_speed = (v - omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    right_speed = (v + omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    set_wheels(left_motor, right_motor, left_speed, right_speed)
    return False


print("[Attendance] Starting serpentine classroom route (9 seat scans).")
waypoint_index = 0
state = "NAVIGATE"
dwell_start = None
scanned_students = set()
avoidance_note_time = -100.0

try:
    while robot.step(timestep) != -1:
        if waypoint_index >= len(WAYPOINTS):
            stop_robot(left_motor, right_motor)
            print("[Attendance] Route complete. Attendance log:", log_path)
            break

        target_x, target_z, seat_id = WAYPOINTS[waypoint_index]
        x, z, yaw = get_pose()
        sim_time = robot.getTime()

        # Emit one warning per second at most if the sonar avoidance activates.
        if sensor_distance(front_sonar_names) < OBSTACLE_DISTANCE_M:
            if sim_time - avoidance_note_time > 1.0:
                print("[Navigation] Obstacle close; steering toward clearer side.")
                avoidance_note_time = sim_time

        if state == "NAVIGATE":
            arrived = drive_to_target(target_x, target_z, x, z, yaw)
            if arrived:
                stop_robot(left_motor, right_motor)
                if seat_id is None:
                    waypoint_index += 1
                else:
                    print(f"[Attendance] Reached {seat_id}; aligning camera toward desk.")
                    state = "FACE_SEAT"

        elif state == "FACE_SEAT":
            # Desk/student markers are on the +Z side of each approach lane.
            if rotate_toward(math.pi, yaw):
                dwell_start = sim_time
                state = "SCAN_DWELL"

        elif state == "SCAN_DWELL":
            stop_robot(left_motor, right_motor)
            if dwell_start is not None and sim_time - dwell_start >= SCAN_DWELL_SECONDS:
                if seat_id not in scanned_students:
                    snapshot_path = save_camera_ppm(camera, captures_dir, seat_id)
                    # This is a simulation waypoint event, not face recognition.
                    log_writer.writerow([
                        f"{sim_time:.2f}", seat_id, ROSTER.get(seat_id, seat_id),
                        "PRESENT_SIMULATED", snapshot_path
                    ])
                    log_file.flush()
                    scanned_students.add(seat_id)
                    print(f"[Attendance] {seat_id} ({ROSTER.get(seat_id, seat_id)}): "
                          f"PRESENT_SIMULATED" +
                          (f"; image: {snapshot_path}" if snapshot_path else ""))
                waypoint_index += 1
                state = "NAVIGATE"
                dwell_start = None

finally:
    stop_robot(left_motor, right_motor)
    try:
        log_file.flush()
        log_file.close()
    except Exception:
        pass
