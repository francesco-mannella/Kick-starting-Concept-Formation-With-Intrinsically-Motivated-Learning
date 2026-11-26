import argparse
import json
import re
from pathlib import Path

import box2dsim
from touch_layout import FINGERS, LINK_LENGTH, SENSOR, plan

MODELS = Path(box2dsim.__file__).parent / "models"
NAMES = ("grasp3_s.json", "grasp3_m.json", "grasp3_l.json")


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
