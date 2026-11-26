import numpy as np

CLASSIC = (slice(10, 20), slice(20, 30))
SKETCH = (slice(6, 20), slice(20, 34))

LINK_LENGTH, LINK_WIDTH, SENSOR = 2.9, 0.5, 0.2
# sensor positions along each link (0 = joint end, 1 = free end), read off the drawing sensors.jpg
PROXIMAL_IN = [0.055, 0.135, 0.21, 0.27, 0.655, 0.74, 0.81, 0.875, 0.945]
PROXIMAL_OUT = [0.48, 0.675, 0.88]
DISTAL_IN = [0.51, 0.61, 0.72, 0.83, 0.96]
DISTAL_OUT = [0.145, 0.51, 0.91]
# finger: (proximal link, distal link, face of the inner side in the link frame)
FINGERS = (("Claw21", "Claw22", "high"), ("Claw11", "Claw12", "low"))


def inner_slices(obj_set):
    """Sensor index ranges of the inner face of the left and of the right finger."""
    return SKETCH if obj_set == "grasp3s" else CLASSIC


def plan():
    """40 entries (link, face, t) in sensor order: outer left 6, inner left 14, inner right 14, outer right 6."""
    entries = []
    for finger, (prox, dist, _) in enumerate(FINGERS):
        outer = [(prox, "out", t) for t in PROXIMAL_OUT] + [(dist, "out", t) for t in DISTAL_OUT]
        inner = [(prox, "in", t) for t in PROXIMAL_IN] + [(dist, "in", t) for t in DISTAL_IN]
        entries += outer + inner if finger == 0 else inner + outer
    return entries


def hand_drawing(proximal_angle=52.0, distal_angle=34.0, base_gap=0.5):
    """Clean drawing of the hand for the touch plots, from the real link geometry: a symmetric open pose with the
    proximal links leaning outward and the distal links inward. Returns the sensor points (40, 2), whether each
    sensor is on an inner face (40,), and the four link rectangles, all scaled into [0.1, 0.9] with y up."""
    a, b = np.radians(proximal_angle), np.radians(distal_angle)
    half, face = LINK_WIDTH / 2, LINK_WIDTH / 2 + SENSOR / 2
    left = {}
    base = np.array([-base_gap / 2, 0.0])
    u_prox = np.array([-np.sin(a), np.cos(a)])
    elbow = base + LINK_LENGTH * u_prox
    u_dist = np.array([np.sin(b), np.cos(b)])
    left["Claw21"] = (base, u_prox)
    left["Claw22"] = (elbow, u_dist)
    mirror = np.array([-1.0, 1.0])
    frames = {name: (p, u) for name, (p, u) in left.items()}
    frames |= {"Claw11": (left["Claw21"][0] * mirror, left["Claw21"][1] * mirror), "Claw12": (left["Claw22"][0] * mirror, left["Claw22"][1] * mirror)}
    points, inner_flags = [], []
    for link, side, t in plan():
        origin, u = frames[link]
        toward_centre = np.array([u[1], -u[0]]) if link in ("Claw21", "Claw22") else np.array([-u[1], u[0]])
        y = SENSOR / 2 + (LINK_LENGTH - SENSOR) * t
        sign = 1.0 if side == "in" else -1.0
        points.append(origin + u * y + sign * toward_centre * face)
        inner_flags.append(side == "in")
    rects = []
    for link in ("Claw21", "Claw22", "Claw12", "Claw11"):
        origin, u = frames[link]
        n = np.array([u[1], -u[0]])
        rects.append(np.array([origin - n * half, origin + n * half, origin + LINK_LENGTH * u + n * half, origin + LINK_LENGTH * u - n * half]))
    allp = np.vstack([np.array(points)] + rects)
    lo = allp.min(0)
    span = (allp.max(0) - lo).max()
    centre_shift = (span - (allp.max(0) - lo)) / 2
    scale = lambda q: (np.asarray(q) - lo + centre_shift) / span * 0.8 + 0.1
    return scale(np.array(points)), np.array(inner_flags), [scale(r) for r in rects]


HAND_POINTS, HAND_INNER, HAND_LINKS = hand_drawing()
INNER_COLOUR, OUTER_COLOUR, LINK_COLOUR = "#b03a2e", "#1f618d", "#d5d8dc"


def sensor_links(obj_set):
    """Finger links drawn behind the sensors of the touch plots (none for the classic layout)."""
    return HAND_LINKS if obj_set == "grasp3s" else []
