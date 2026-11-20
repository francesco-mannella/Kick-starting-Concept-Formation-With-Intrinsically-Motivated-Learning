import json
from pathlib import Path

import numpy as np
import pandas as pd

K = Path("/home/fmannella/tmp/kss")
A, B = "fig567_designB_greensmall_amp007", "fig567_designB_greensmall"
rng = np.random.default_rng(0)
dd = pd.read_csv(K / "mi_study3_discrete.csv")
dc = pd.read_csv(K / "mi_study3_continuous.csv")


def ci(x, n=4000):
    x = np.asarray(x, float)
    m = rng.choice(x, (n, len(x))).mean(1)
    return x.mean(), np.quantile(m, 0.025), np.quantile(m, 0.975)


def fmt(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]"


pd.set_option("display.width", 250)
train = {}
for sw in (A, B):
    t = pd.DataFrame([json.load(open(K / "mi_extract" / sw / f"seed_{s}.json")) for s in range(100, 110)])
    train[sw] = t.mean()
print("Training metric (calc_mi_metrics, plug-in, per time step) recomputed on 360 evaluation episodes, mean over 10 seeds:")
print(pd.DataFrame(train).loc[["mi_score_context_policy", "mi_score_context_touch", "mi_score_policy_touch", "mi_score_context_gripper", "mi_score_policy_gripper"]].round(3).to_string())

for level in ("segment", "step"):
    print(f"\n=== DISCRETE pairs, {level} level; permutation excess = raw - episode-block null (nats), mean over 10 seeds [95% bootstrap CI over seeds]")
    rows = []
    for pair in dd.pair.unique():
        for sw, tag in ((A, "amp0.07"), (B, "amp8")):
            s = dd[(dd.sweep == sw) & (dd.level == level) & (dd.pair == pair)].sort_values("seed")
            r = dict(pair=pair, sweep=tag, N=int(s.N.mean()), raw=s.raw.mean(), shrinkage=s.shrinkage.mean(), ami=s.ami.mean(), null=s.perm_null.mean(), excess=fmt(ci(s.perm_excess)), z=s.z.mean())
            if "cv_bound" in s and s.cv_bound.notna().any():
                r["cv_bound"] = s.cv_bound.mean()
            rows.append(r)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("paired difference amp0.07 - amp8 in permutation excess:")
    for pair in dd.pair.unique():
        a = dd[(dd.sweep == A) & (dd.level == level) & (dd.pair == pair)].sort_values("seed").perm_excess.values
        b = dd[(dd.sweep == B) & (dd.level == level) & (dd.pair == pair)].sort_values("seed").perm_excess.values
        print(f"  {pair:22s} {fmt(ci(a - b))}")

print("\n=== CONTINUOUS variables (segment level; PC5 of the executed action, 5 joint angles, 4 claw-segment touch sums); excess over the episode-block null, nats (dCor is unitless)")
rows = []
for (pair, method), g in dc.groupby(["pair", "method"]):
    for sw, tag in ((A, "amp0.07"), (B, "amp8")):
        s = g[g.sweep == sw].sort_values("seed")
        r = dict(pair=pair, method=method, sweep=tag, N=int(s.N.mean()), est=s.est.mean(), null=s.null.mean(), excess=fmt(ci(s.excess)), z=s.z.mean())
        if "cv_bound" in s and s.cv_bound.notna().any():
            r["cv_bound"] = s.cv_bound.mean()
        rows.append(r)
print(pd.DataFrame(rows).round(3).to_string(index=False))
