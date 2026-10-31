#!/usr/bin/env python
"""Insert a wrist link between the tip of Arm3 and the two claws in the one-eye world files.

The claws are re-hinged on the new Wrist body at their original world anchors, so the gripper pose at
zero action does not change. A revolute motor joint Arm3_to_Wrist joins Arm3 and Wrist.
"""
import json
import sys
from pathlib import Path

WORLDS = ("unreachable", "still", "movable", "controllable", "noobject")
# Collision categories. The claws are hinged to the wrist, not to Arm3, and the palm overlaps the base of the claws and
# of their touch sensors, so Arm3 and Wrist must not collide with the claws or the sensors. Every other pair
# (claw with claw, sensor with object, anything with the objects) collides as before.
ARM_CATEGORY, CLAW_CATEGORY, SENSOR_CATEGORY = 2, 4, 8
CLAW_BODIES = ("Claw11", "Claw12", "Claw21", "Claw22")
PALM_WIDTH, PALM_HEIGHT, PALM_DENSITY = 1.4, 0.4, 5.0


def index(items, name):
    return next(i for i, item in enumerate(items) if item.get("name") == name)


def set_filters(world):
    for body in world["body"]:
        name = body.get("name", "")
        if name in ("Arm3", "Wrist"):
            category, mask = ARM_CATEGORY, 65535 & ~(CLAW_CATEGORY | SENSOR_CATEGORY)
        elif name in CLAW_BODIES:
            category, mask = CLAW_CATEGORY, 65535 & ~ARM_CATEGORY
        elif name[:1] == "s" and name[1:].isdigit():
            category, mask = SENSOR_CATEGORY, 65535 & ~ARM_CATEGORY
        else:
            continue
        for fixture in body["fixture"]:
            fixture["filter-categoryBits"], fixture["filter-maskBits"] = category, mask


def set_palm_density(world):
    for body in world["body"]:
        if body.get("name") == "Wrist":
            for fixture in body["fixture"]:
                fixture["density"] = PALM_DENSITY


def add_wrist(world):
    bodies, joints = world["body"], world["joint"]
    if any(b.get("name") == "Wrist" for b in bodies):
        return False
    arm3 = bodies[index(bodies, "Arm3")]
    xs, ys = arm3["fixture"][0]["polygon"]["vertices"]["x"], arm3["fixture"][0]["polygon"]["vertices"]["y"]
    tip_local = (max(xs) / 2, max(ys))
    tip_world = (arm3["position"]["x"] + tip_local[0], arm3["position"]["y"] + tip_local[1])
    position = {"x": tip_world[0] - PALM_WIDTH / 2, "y": tip_world[1]}
    wrist = {
        "angle": 0.0,
        "name": "Wrist",
        "color": [0.7, 0.7, 0.7],
        "position": position,
        "type": 2,
        "fixture": [{
            "density": PALM_DENSITY,
            "group_index": 0,
            "polygon": {"vertices": {"x": [0.0, PALM_WIDTH, PALM_WIDTH, 0.0, 0.0],
                                     "y": [PALM_HEIGHT, PALM_HEIGHT, 0.0, 0.0, PALM_HEIGHT]}},
            "friction": 0.5,
        }],
    }
    bodies.append(wrist)
    wrist_id, arm3_id = len(bodies) - 1, index(bodies, "Arm3")

    template = joints[index(joints, "Arm2_to_Arm3")]
    joint = dict(template)
    joint.update(name="Arm3_to_Wrist", bodyA=arm3_id, bodyB=wrist_id,
                 anchorA={"x": tip_local[0], "y": tip_local[1]}, anchorB={"x": PALM_WIDTH / 2, "y": 0.0})
    for old, new in (("Arm3_to_Claw11", "Wrist_to_Claw11"), ("Arm3_to_Claw21", "Wrist_to_Claw21")):
        claw = joints[index(joints, old)]
        world_anchor = (arm3["position"]["x"] + claw["anchorA"]["x"], arm3["position"]["y"] + claw["anchorA"]["y"])
        claw["name"], claw["bodyA"] = new, wrist_id
        claw["anchorA"] = {"x": world_anchor[0] - position["x"], "y": world_anchor[1] - position["y"]}
    joints.append(joint)
    return True


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    for name in WORLDS:
        path = folder / f"{name}.json"
        world = json.loads(path.read_text())
        changed = add_wrist(world)
        set_filters(world)
        set_palm_density(world)
        path.write_text(json.dumps(world, indent=4) + "\n")
        print(f"{name}: {'wrist added' if changed else 'already has a wrist'}, collision filters set ({len(world['body'])} bodies, {len(world['joint'])} joints)")
