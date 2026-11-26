import argparse
import json
import re
from pathlib import Path

import box2dsim

MODELS = Path(box2dsim.__file__).parent / "models"
NAMES = ("grasp3_s.json", "grasp3_m.json", "grasp3_l.json")
LINK_LENGTH, SENSOR = 2.9, 0.2
# sensor positions along each link (0 = joint end, 1 = free end), read off the drawing sensors.jpg
PROXIMAL_IN = [0.055, 0.135, 0.21, 0.27, 0.655, 0.74, 0.81, 0.875, 0.945]
PROXIMAL_OUT = [0.48, 0.675, 0.88]
DISTAL_IN = [0.51, 0.61, 0.72, 0.83, 0.96]
DISTAL_OUT = [0.145, 0.51, 0.91]
# finger: (proximal link, distal link, face of the inner side in the link frame)
FINGERS = (("Claw21", "Claw22", "high"), ("Claw11", "Claw12", "low"))


def plan():
    """40 entries (link, face, t) in sensor order: outer left 6, inner left 14, inner right 14, outer right 6."""
    entries = []
    for finger, (prox, dist, _) in enumerate(FINGERS):
        outer = [(prox, "out", t) for t in PROXIMAL_OUT] + [(dist, "out", t) for t in DISTAL_OUT]
        inner = [(prox, "in", t) for t in PROXIMAL_IN] + [(dist, "in", t) for t in DISTAL_IN]
        entries += outer + inner if finger == 0 else inner + outer
    return entries


def templates(world, bodies):
    """anchor x on the link and anchor on the sensor for each (link, face), taken from the existing sensors."""
    found = {}
    inner_side = {prox: side for prox, _, side in FINGERS} | {dist: side for _, dist, side in FINGERS}
    for joint in world["joint"]:
        match = re.match(r"(Claw\d\d)_to_s\d+$", joint["name"])
        if not match:
            continue
        link = match.group(1)
        high = joint["anchorA"]["x"] > 0.25
        face = "in" if (inner_side[link] == "high") == high else "out"
        found.setdefault((link, face), (joint["anchorA"]["x"], dict(joint["anchorB"])))
    return found


def make():
    for name in NAMES:
        world = json.load(open(MODELS / name))
        bodies = world["body"]
        index = {b["name"]: i for i, b in enumerate(bodies)}
        template = templates(world, bodies)
        sensor_joints = {int(re.search(r"s(\d+)$", j["name"]).group(1)): j for j in world["joint"] if re.match(r"Claw\d\d_to_s\d+$", j["name"])}
        for number, (link, face, t) in enumerate(plan(), start=1):
            ax, anchor_b = template[(link, face)]
            y = SENSOR / 2 + (LINK_LENGTH - SENSOR) * t
            joint = sensor_joints[number]
            joint["name"] = f"{link}_to_s{number}"
            joint["bodyA"] = index[link]
            joint["anchorA"] = {"x": ax, "y": y}
            joint["anchorB"] = dict(anchor_b)
            parent, sensor = bodies[index[link]], bodies[index[f"s{number}"]]
            sensor["position"] = {"x": parent["position"]["x"] + ax - anchor_b["x"], "y": parent["position"]["y"] + y - anchor_b["y"]}
        out = MODELS / name.replace("grasp3_", "grasp3s_")
        json.dump(world, open(out, "w"), indent=4)
        print(f"{out.name}: {len(sensor_joints)} sensors re-anchored")


if __name__ == "__main__":
    argparse.ArgumentParser(description="Write the grasp3s world files: the grasp3 worlds with the sensor layout of sensors.jpg").parse_args()
    make()
