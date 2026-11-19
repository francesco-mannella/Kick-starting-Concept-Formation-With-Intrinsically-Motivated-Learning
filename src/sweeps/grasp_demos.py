import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WT = Path("/home/fmannella/Dropbox/Projects/current/Kick-starting-Sources/.claude/worktrees/l2-objects")
K = Path("/home/fmannella/tmp/kss")
WORK = K / "designB_grasp"
OUT = K / "fig567_designB" / "demos" / "grasp_demos"
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MPLBACKEND": "agg", "PYTHONPATH": f"{WT / 'src'}:{WT / 'tools/box2dsim'}",
       "OBJECT_LABELS": "blue plate,red triangle,green wedge"}
NAMES = {1: "plate", 2: "triangle", 3: "wedge"}
CASES = [  # (seed, eval seed, episode, note)
    (105, 304, 9, "plate_both_fingers"), (105, 319, 9, "plate_both_fingers"), (105, 308, 9, "plate_both_fingers"), (105, 311, 9, "plate_both_fingers"),
    (105, 303, 0, "plate_both_fingers"), (106, 300, 16, "triangle_along_claw"), (104, 313, 13, "triangle_dragged"),
    (105, 315, 7, "triangle_long_contact"), (105, 310, 2, "wedge_both_fingers"),
]
sys.path.insert(0, str(WT / "src"))
from params import Parameters  # noqa: E402
from SMMain import build_episode_dataset  # noqa: E402

dataset, _ = build_episode_dataset(Parameters())
ctx = dataset.set_index("index")


def folder(seed, es):
    return WORK / f"seed{seed}_e{es}"


procs = []
for seed, es, ep, _ in CASES:
    d = folder(seed, es)
    if (d / "trajectory_df.csv").exists():
        continue
    d.mkdir(parents=True, exist_ok=True)
    src = K / f"designB_demos_work/seed_{seed}/parasite_{seed}/simulations/parasite_{seed}/storage/final_ckpt"
    for f in ("main.dump.npy", "weights.npy"):
        shutil.copy(src / f, d / f)
    log = open(d / "render.log", "w")
    procs.append(subprocess.Popen([sys.executable, str(WT / "src/run_evaluation.py"), "-n", "1", "-s", str(es), "-p"], cwd=d, env=ENV, stdout=log, stderr=subprocess.STDOUT))
for p in procs:
    p.wait()
print("episodes rendered:", len(procs), "new evaluation folders", flush=True)

OUT.mkdir(parents=True, exist_ok=True)
rows, jobs = [], []
for seed, es, ep, note in CASES:
    d = folder(seed, es)
    df = pd.read_csv(d / "trajectory_df.csv")
    scols = sorted([c for c in df.columns if c.startswith("s") and c[1:].isdigit()], key=lambda c: int(c[1:]))
    g = df[(df.e_seed == es) & (df.episode_id == ep)].sort_values("ts")
    best = None
    for goal, gg in g.groupby("goal_id"):
        t = gg[scols].values > 0
        both = int((t[:, :20].any(1) & t[:, 20:].any(1)).sum())
        contact = int(t.any(1).sum())
        key = (both, contact)
        if best is None or key > best[0]:
            best = (key, int(goal), len(gg), int(t.sum(1).max()))
    (both, contact), goal, n, ms = best
    final = d / "final"
    if not final.exists():
        (final / "episodes").mkdir(parents=True)
        for f in ("demos", "logs", "goal_frames"):
            (final / f).mkdir()
        shutil.copy(d / "trajectory_df.csv", final / "trajectory_df.csv")
        shutil.copy(d / "weights.npy", final / "weights.npy")
        for gif in (d / "episodes").glob("episode_*_0.gif"):
            shutil.copy(gif, final / "episodes" / gif.name)
    jobs.append((seed, es, ep, note, goal, both, contact, n, ms, final))
running = []
for seed, es, ep, note, goal, both, contact, n, ms, final in jobs:
    running.append((subprocess.Popen([sys.executable, str(WT / "src/plot_postures.py"), "-e", str(ep), "-r", "0", "-g", str(goal)], cwd=final, env=ENV,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE), (seed, es, ep, note, goal, both, contact, n, ms, final)))
    while sum(p.poll() is None for p, _ in running) >= 5:
        running[[p.poll() is None for p, _ in running].index(True)][0].wait()
for p, (seed, es, ep, note, goal, both, contact, n, ms, final) in running:
    err = p.communicate()[1].decode()[-200:] if p.poll() is None else p.stderr.read().decode()[-200:]
    found = sorted((final / "demos").glob(f"e{ep:02d}_g{goal:02d}_r0_*.gif"))
    if p.returncode != 0 or not found:
        print("FAILED", seed, es, ep, err)
        continue
    t = ctx.loc[ep]
    name = (f"{note}_seed{seed}_ep{ep:02d}_{NAMES[int(t.context)]}_stretch{t.stretch}_rot{int(round(np.degrees(t.rotation)))}_eval{es}_goal{goal}.gif")
    shutil.copy(found[0], OUT / name)
    rows.append(dict(file=name, seed=seed, eval_seed=es, episode=ep, object=NAMES[int(t.context)], stretch=t.stretch, rotation_deg=int(round(np.degrees(t.rotation))),
                     goal_id=goal, segment_steps=n, two_finger_steps=both, contact_steps=contact, max_sensors=ms))
index = pd.DataFrame(rows)
index.to_csv(OUT / "index.csv", index=False)
print(index.to_string(index=False))
print(len(index), "demo GIFs in", OUT)
