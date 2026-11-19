import json
import pickle
from pathlib import Path

H = Path.home() / "tmp/kss"


def ctx(k):
    shape, scale, density, pos, rot = k
    return dict(shape=shape, scale=scale, density=density, rotation_deg=rot, position=None if pos is None else list(pos))


out = {"note": "contexts found by pattern-transfer search on the 5-output hand (claw_5); position None = default place (11.23, 11.32); champion physics (friction 2, damping 20)"}
for K in (3, 6, 9, 18):
    out[f"unstructured_K{K}"] = [ctx(k) for k in pickle.load(open(H / f"claw_5/dataset_K{K}.pkl", "rb"))]
for hand in ("claw_5", "claw_8"):
    d = pickle.load(open(H / f"{hand}/dataset_structured.pkl", "rb"))
    out[f"structured_3kinds_x_2rotations_{hand}"] = [dict(shape=t[0], scale=t[1], density=1.0, position=list(t[2]), rotations_deg=[0, 90]) for t in d["trio"]]
(H / "dataset_proposals.json").write_text(json.dumps(out, indent=1))
print({k: (len(v) if isinstance(v, list) else "note") for k, v in out.items()})
