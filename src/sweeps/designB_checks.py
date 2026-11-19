import json
import math
import sqlite3
from pathlib import Path

H = Path.home() / "tmp/kss"
ROOT = H / "fig567_designB"
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


con = sqlite3.connect(f"file:{ROOT}/runs.db?mode=ro", uri=True)
runs = con.execute("select name, grp, config, status from runs").fetchall()
check("20 runs in the database (10 parasite, 10 former)", len(runs) == 20 and sum(r[1] == "parasite" for r in runs) == 10, f"found {len(runs)}")
bad = []
for name, grp, cfg, status in runs:
    c = json.loads(cfg)
    ok = (c.get("obj_set") == "mix" and c.get("obj_damping") == 20.0 and c.get("obj_friction") == 2.0 and c.get("epochs") == 1500
          and c.get("obj_pos_kind") == {"still": [14.0, 12.0], "movable": [7.0, 12.0], "controllable": [14.0, 4.0]} and not c.get("saliency"))
    if not ok:
        bad.append(name)
check("every run logged design B (obj_set mix, per-kind positions), friction 20/2, 1500 epochs, saliency off", not bad, str(bad))
for stage in ("parasite", "former"):
    missing = []
    for s in range(100, 110):
        name = f"{stage}_{s:03d}"
        last = ROOT / name / "simulations" / name / "storage" / "001499"
        files = ["weights.npy"] + (["weights-parasite.npy"] if stage == "parasite" else [])
        if not all((last / f).is_file() for f in files):
            missing.append(name)
    check(f"all {stage} runs reached the final checkpoint (epoch 1499)", not missing, str(missing))
nan = con.execute("select count(*) from metrics where value is null or value != value").fetchone()[0]
check("no NaN or null metric values", nan == 0, f"{nan} found")
epochs = {name: con.execute("select max(step) from metrics m join runs r on r.id=m.run_id where r.name=?", (name,)).fetchone()[0] for name, *_ in runs}
check("every run logged metrics up to epoch 1499", all(v == 1499 for v in epochs.values()), str({k: v for k, v in epochs.items() if v != 1499}))
stats = [ROOT / "stats" / f"agent_{s}_static.json" for s in range(100, 110)]
cf = [ROOT / "stats" / f"agent_{s}_counterfactual_by_context.csv" for s in range(100, 110)]
check("10 static statistics and 10 counterfactual tables written", all(p.is_file() for p in stats + cf))
vals = []
for p in stats:
    if p.is_file():
        vals += [v for v in json.load(open(p)).values() if isinstance(v, float)]
check("no NaN in the static statistics", all(not math.isnan(v) for v in vals))
early = {}
for name, grp, *_ in runs:
    if grp == "former":
        early[name] = con.execute("select avg(value) from metrics m join runs r on r.id=m.run_id where r.name=? and m.key='grid_comp_mean' and m.step < 100", (name,)).fetchone()[0]
check("former runs start from the parasite weights (early alignment > 0.4; a fresh run starts near 0.1)", all(v is not None and v > 0.4 for v in early.values()),
      str({k: round(v, 2) for k, v in early.items() if v is None or v <= 0.4}))
print(f"\n{sum(results)}/{len(results)} checks passed")
