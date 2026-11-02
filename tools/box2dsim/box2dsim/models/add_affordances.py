#!/usr/bin/env python
"""Write the three affordance worlds from noobject.json.

blue: pinned at its centre, rotates anticlockwise only (one-way ratchet, no limit)
red: slides along its main axis like a button, +-TRAVEL units, spring back to its rest position
green: pinned at its centre, rotates clockwise only (one-way ratchet, no limit)

Every world holds a static anchor body (collides with nothing) and the object body, appended after the existing
bodies so no body index shifts. The anchor sits at the object position and the joint anchors are local, so the
position and the shape of the object can change at every reset (Box2DSim_env.randomize_objects moves the anchor
and sets the button axis).
"""
import copy
import json
from pathlib import Path

TRAVEL = 2.0
BUTTON_FORCE = 40.0
OBJECT_POSITION = {"x": 10.73, "y": 9.32}
SQUARE = {"x": [0.0, 1.9, 1.9, 0.0, 0.0], "y": [1.9, 1.9, 0.0, 0.0, 1.9]}
ANCHOR_CATEGORY = 16

WORLDS = {
    "affordance_blue": dict(object="blue_rotor_ccw", color=[0.0, 0.0, 1.0], kind="revolute", joint="ratchet_ccw_blue"),
    "affordance_red": dict(object="red_button", color=[1.0, 0.0, 0.0], kind="prismatic", joint="spring_red"),
    "affordance_green": dict(object="green_rotor_cw", color=[0.0, 1.0, 0.0], kind="revolute", joint="ratchet_cw_green"),
}


def make_world(base, spec):
    world = copy.deepcopy(base)
    bodies, joints = world["body"], world["joint"]
    anchor_id, object_id = len(bodies), len(bodies) + 1
    bodies.append({
        "angle": 0.0, "name": spec["object"] + "_anchor", "color": [1.0, 1.0, 1.0], "position": dict(OBJECT_POSITION), "type": 0,
        "fixture": [{"density": 1.0, "group_index": 0, "friction": 0.0,
                     "polygon": {"vertices": {"x": [0.0, 0.1, 0.1, 0.0, 0.0], "y": [0.1, 0.1, 0.0, 0.0, 0.1]}},
                     "filter-categoryBits": ANCHOR_CATEGORY, "filter-maskBits": 0}],
    })
    bodies.append({
        "angle": 0.0, "name": spec["object"], "color": spec["color"], "position": dict(OBJECT_POSITION), "type": 2,
        "fixture": [{"density": 0.2, "group_index": 0, "friction": 0.5, "polygon": {"vertices": copy.deepcopy(SQUARE)}}],
        "angularDamping": 2.0, "linearDamping": 2.0,
    })
    joint = {"name": spec["joint"], "type": spec["kind"], "bodyA": anchor_id, "bodyB": object_id,
             "anchorA": {"x": 0.0, "y": 0.0}, "anchorB": {"x": 0.0, "y": 0.0}, "collideConnected": False, "enableLimit": True}
    if spec["kind"] == "revolute":
        joint.update(refAngle=0, lowerLimit=-1000.0, upperLimit=1000.0, enableMotor=False, maxMotorTorque=0, motorSpeed=0, jointSpeed=0)
    else:
        joint.update(refAngle=0, lowerLimit=-TRAVEL, upperLimit=TRAVEL, enableMotor=True, maxMotorForce=BUTTON_FORCE, motorSpeed=0,
                     localAxisA={"x": 1.0, "y": 0.0})
    joints.append(joint)
    return world


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    base = json.loads((folder / "noobject.json").read_text())
    for name, spec in WORLDS.items():
        world = make_world(base, spec)
        (folder / f"{name}.json").write_text(json.dumps(world, indent=4) + "\n")
        print(f"{name}: {len(world['body'])} bodies, {len(world['joint'])} joints, object {spec['object']}")
