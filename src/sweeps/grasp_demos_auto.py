import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WT = Path(sys.argv[1])
WORK = Path(sys.argv[2])
ROOT = Path(sys.argv[3])
SEEDS = range(int(sys.argv[4]), int(sys.argv[5]) + 1)
N_TWO, N_LONG = int(sys.argv[6]), int(sys.argv[7])
G3 = len(sys.argv) > 8 and sys.argv[8] == "grasp3"
LABELS = "small square,medium square,large square" if G3 else "blue plate,red triangle,green wedge"
LEFT, RIGHT = (slice(10, 20), slice(20, 30)) if G3 else (slice(0, 20), slice(20, 40))
OUT = ROOT / "demos" / "grasp_demos"
CASEDIR = WORK / "grasp_cases"
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MPLBACKEND": "agg", "PYTHONPATH": f"{WT / 'src'}:{WT / 'tools/box2dsim'}", "OBJECT_LABELS": LABELS}
NAMES = {1: "small", 2: "medium", 3: "large"} if G3 else {1: "plate", 2: "triangle", 3: "wedge"}
sys.path.insert(0, str(WT / "src"))
from params import Parameters  # noqa: E402
from SMMain import build_episode_dataset  # noqa: E402

if G3:
    from SMMain import Main  # noqa: E402

    _ = Main
    first = WORK / f"seed_{SEEDS[0]}/parasite_{SEEDS[0]}/simulations/parasite_{SEEDS[0]}/storage/final_ckpt/main.dump.npy"
    dataset, _ = build_episode_dataset(np.load(first, allow_pickle=True)[0].params)
else:
    dataset, _ = build_episode_dataset(Parameters())
ctx = dataset.set_index("index")


def longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


rows = []
for seed in SEEDS:
    p = WORK / f"seed_{seed}/parasite_{seed}/simulations/parasite_{seed}/storage/final_ckpt/eval/trajectory_df.csv"
    d = pd.read_csv(p)
    scols = sorted([c for c in d.columns if c.startswith("s") and c[1:].isdigit()], key=lambda c: int(c[1:]))
    for (es, ep), g in d.groupby(["e_seed", "episode_id"]):
        t = g.sort_values("ts")[scols].values > 0
        anyc, both = t.any(1), t[:, LEFT].any(1) & t[:, RIGHT].any(1)
        rows.append(dict(seed=seed, e_seed=int(es), episode=int(ep), context=int(ctx.loc[int(ep) % len(ctx), "context"]), longest_bilateral=longest_run(both),
                         bilateral_steps=int(both.sum()), longest_contact=longest_run(anyc), max_sensors=int(t.sum(1).max())))
scan = pd.DataFrame(rows)
print(f"{len(scan)} episode runs; two fingers >= 5 steps: {(scan.longest_bilateral >= 5).sum()}; >= 10: {(scan.longest_bilateral >= 10).sum()}; contact >= 20: {(scan.longest_contact >= 20).sum()}; >= 40: {(scan.longest_contact >= 40).sum()}", flush=True)
print(scan.groupby("seed").apply(lambda g: pd.Series(dict(two_finger5=(g.longest_bilateral >= 5).sum(), contact40=(g.longest_contact >= 40).sum()))).T.to_string(), flush=True)


def pick(df, key, n, per_seed, taken):
    out, count = [], {}
    for _, r in df.sort_values(key, ascending=False).iterrows():
        k = (r.seed, r.e_seed, r.episode)
        if k in taken or count.get(r.seed, 0) >= per_seed:
            continue
        out.append(r)
        taken.add(k)
        count[r.seed] = count.get(r.seed, 0) + 1
        if len(out) >= n:
            break
    return out


taken = set()
two = pick(scan[scan.longest_bilateral >= 5], ["longest_bilateral", "longest_contact"], N_TWO, 3, taken)
long = pick(scan[(scan.context >= (1 if G3 else 2)) & (scan.longest_contact >= 30)], ["longest_contact", "max_sensors"], N_LONG, 2, taken)
cases = [("two_fingers", r) for r in two] + [("sustained_contact", r) for r in long]
print(len(cases), "cases selected", flush=True)

procs, folders = [], {}
for _, r in cases:
    key = (int(r.seed), int(r.e_seed))
    if key in folders:
        continue
    d = CASEDIR / f"seed{key[0]}_e{key[1]}"
    folders[key] = d
    if (d / "trajectory_df.csv").exists():
        continue
    d.mkdir(parents=True, exist_ok=True)
    src = WORK / f"seed_{key[0]}/parasite_{key[0]}/simulations/parasite_{key[0]}/storage/final_ckpt"
    for f in ("main.dump.npy", "weights.npy"):
        shutil.copy(src / f, d / f)
    procs.append(subprocess.Popen([sys.executable, str(WT / "src/run_evaluation.py"), "-n", "1", "-s", str(key[1]), "-p"], cwd=d, env=ENV,
                                  stdout=open(d / "render.log", "w"), stderr=subprocess.STDOUT))
    while sum(p.poll() is None for p in procs) >= 8:
        [p for p in procs if p.poll() is None][0].wait()
for p in procs:
    p.wait()
print("episodes rendered:", len(procs), flush=True)

OUT.mkdir(parents=True, exist_ok=True)
jobs = []
for cat, r in cases:
    d = folders[(int(r.seed), int(r.e_seed))]
    df = pd.read_csv(d / "trajectory_df.csv")
    scols = sorted([c for c in df.columns if c.startswith("s") and c[1:].isdigit()], key=lambda c: int(c[1:]))
    g = df[(df.e_seed == r.e_seed) & (df.episode_id == r.episode)].sort_values("ts")
    best = None
    for goal, gg in g.groupby("goal_id"):
        t = gg[scols].values > 0
        both, contact = int((t[:, LEFT].any(1) & t[:, RIGHT].any(1)).sum()), int(t.any(1).sum())
        key = (both, contact) if cat == "two_fingers" else (contact, both)
        if best is None or key > best[0]:
            best = (key, int(goal), len(gg), both, contact, int(t.sum(1).max()))
    _, goal, n, both, contact, ms = best
    final = d / "final"
    if not final.exists():
        (final / "episodes").mkdir(parents=True)
        for f in ("demos", "logs", "goal_frames"):
            (final / f).mkdir()
        shutil.copy(d / "trajectory_df.csv", final / "trajectory_df.csv")
        shutil.copy(d / "weights.npy", final / "weights.npy")
        for gif in (d / "episodes").glob("episode_*_0.gif"):
            shutil.copy(gif, final / "episodes" / gif.name)
    jobs.append((cat, int(r.seed), int(r.e_seed), int(r.episode), goal, n, both, contact, ms, final))
running = []
for job in jobs:
    cat, seed, es, ep, goal, n, both, contact, ms, final = job
    running.append((subprocess.Popen([sys.executable, str(WT / "src/plot_postures.py"), "-e", str(ep), "-r", "0", "-g", str(goal)], cwd=final, env=ENV,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE), job))
    while sum(p.poll() is None for p, _ in running) >= 6:
        [p for p, _ in running if p.poll() is None][0].wait()
index = []
for p, (cat, seed, es, ep, goal, n, both, contact, ms, final) in running:
    err = p.communicate()[1].decode()[-200:]
    found = sorted((final / "demos").glob(f"e{ep:02d}_g{goal:02d}_r0_*.gif"))
    if p.returncode != 0 or not found:
        print("FAILED", seed, es, ep, err, flush=True)
        continue
    t = ctx.loc[ep % len(ctx)]
    name = f"{cat}_seed{seed}_ep{ep:02d}_{NAMES[int(t.context)]}_stretch{t.stretch}_rot{int(round(np.degrees(t.rotation)))}_eval{es}_goal{goal}.gif"
    shutil.copy(found[0], OUT / name)
    index.append(dict(file=name, category=cat, seed=seed, eval_seed=es, episode=ep, object=NAMES[int(t.context)], stretch=t.stretch, rotation_deg=int(round(np.degrees(t.rotation))),
                      goal_id=goal, segment_steps=n, two_finger_steps=both, contact_steps=contact, max_sensors=ms))
index = pd.DataFrame(index)
index.to_csv(OUT / "index.csv", index=False)
print(index.to_string(index=False))
figs = ROOT / "figures"
figs.mkdir(exist_ok=True)
for f in OUT.glob("*.gif"):
    shutil.copy(f, figs / f"grasp_demo_{f.name}")
shutil.copy(OUT / "index.csv", figs / "grasp_demo_index.csv")
print(len(index), "grasp demo GIFs in", OUT, "and copied to", figs, flush=True)
