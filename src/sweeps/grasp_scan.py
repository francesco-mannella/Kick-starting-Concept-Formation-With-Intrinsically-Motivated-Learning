import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/fmannella/Dropbox/Projects/current/Kick-starting-Sources/.claude/worktrees/l2-objects/src")
from params import Parameters  # noqa: E402
from SMMain import build_episode_dataset  # noqa: E402

W = "/home/fmannella/tmp/kss/designB_demos_work"
dataset, _ = build_episode_dataset(Parameters())
ctx = dataset.set_index("index")
NAMES = {1: "plate (still)", 2: "triangle (movable)", 3: "wedge (controllable)"}


def longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


rows = []
for seed in range(100, 110):
    p = f"{W}/seed_{seed}/parasite_{seed}/simulations/parasite_{seed}/storage/final_ckpt/eval/trajectory_df.csv"
    d = pd.read_csv(p)
    scols = [c for c in d.columns if c.startswith("s") and c[1:].isdigit()]
    scols = sorted(scols, key=lambda c: int(c[1:]))
    for (es, ep), g in d.groupby(["e_seed", "episode_id"]):
        g = g.sort_values("ts")
        t = g[scols].values > 0
        left, right = t[:, :20].any(1), t[:, 20:].any(1)
        anyc = t.any(1)
        both = left & right
        joints = g[["d1", "d2", "d3", "d4", "d5"]].values
        speed = np.r_[0, np.linalg.norm(np.diff(joints, axis=0), axis=1)]
        run = longest_run(anyc)
        # joint speed inside the longest contact run
        best = (0, 0, 0)
        cur = 0
        for i, v in enumerate(anyc):
            cur = cur + 1 if v else 0
            if cur > best[0]:
                best = (cur, i - cur + 1, i + 1)
        still = float(speed[best[1]:best[2]].mean()) if best[0] else np.nan
        rows.append(dict(seed=seed, e_seed=int(es), episode=int(ep), context=int(ctx.loc[int(ep), "context"]), steps=len(g), contact_steps=int(anyc.sum()),
                         longest_contact=run, bilateral_steps=int(both.sum()), longest_bilateral=longest_run(both), max_sensors=int(t.sum(1).max()),
                         speed_in_longest_contact=still, mean_speed=float(speed.mean())))
df = pd.DataFrame(rows)
df.to_csv("/home/fmannella/tmp/kss/grasp_scan.csv", index=False)
print(len(df), "episode runs over 10 seeds (", df.groupby("seed").size().iloc[0], "per seed )")
for k in (1, 3, 5, 10):
    print(f"  episodes with both fingers touching for >= {k} consecutive steps: {(df.longest_bilateral >= k).mean():.3f}  ({(df.longest_bilateral >= k).sum()})")
for k in (5, 10, 20, 40):
    print(f"  episodes with unbroken contact of >= {k} steps: {(df.longest_contact >= k).mean():.3f}  ({(df.longest_contact >= k).sum()})")
print("  mean steps in contact per episode: %.1f of %.0f" % (df.contact_steps.mean(), df.steps.mean()))
print("\nby object (context):")
for c, g in df.groupby("context"):
    print(f"  {NAMES[c]:22s} bilateral>=3: {(g.longest_bilateral >= 3).mean():.3f}  bilateral>=5: {(g.longest_bilateral >= 5).mean():.3f}  contact>=10: {(g.longest_contact >= 10).mean():.3f}  contact>=20: {(g.longest_contact >= 20).mean():.3f}  max sensors mean {g.max_sensors.mean():.1f}")
print("\nby seed (share of episodes with bilateral >= 5 / contact >= 20):")
print(df.groupby("seed").apply(lambda g: pd.Series(dict(bil5=(g.longest_bilateral >= 5).mean(), contact20=(g.longest_contact >= 20).mean(), maxrun=g.longest_contact.max()))).round(3).T.to_string())
top = df.sort_values(["longest_bilateral", "longest_contact"], ascending=False).head(8)
print("\ntop candidates (longest simultaneous two-finger contact):")
print(top[["seed", "e_seed", "episode", "context", "longest_bilateral", "bilateral_steps", "longest_contact", "max_sensors", "speed_in_longest_contact", "mean_speed"]].round(3).to_string(index=False))
