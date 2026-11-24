import argparse
import json
from pathlib import Path

import box2dsim

MODELS = Path(box2dsim.__file__).parent / "models"
WORLDS = (("grasp3_s.json", "still"), ("grasp3_m.json", "movable"), ("grasp3_l.json", "controllable"))


def make(sizes):
    for (name, body_name), width in zip(WORLDS, sizes):
        world = json.load(open(MODELS / "still.json"))
        body = next(b for b in world["body"] if b["name"] == "still")
        body["name"] = body_name
        body["type"] = 0
        body["fixture"][0]["polygon"]["vertices"] = {"x": [0.0, width, width, 0.0, 0.0], "y": [width, width, 0.0, 0.0, width]}
        json.dump(world, open(MODELS / name, "w"), indent=4)
        print(f"{name}: static square {width} x {width}, body {body_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write the three grasp3 world files (static squares, same colour)")
    parser.add_argument("--sizes", type=float, nargs=3, default=[1.5, 3.0, 5.0])
    make(parser.parse_args().sizes)
