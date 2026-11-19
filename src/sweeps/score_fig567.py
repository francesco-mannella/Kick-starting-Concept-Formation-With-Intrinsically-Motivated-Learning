import sqlite3, sys
import numpy as np, pandas as pd
new = sys.argv[1]; old = sys.argv[2]
METRICS = [("alignment", "grid_comp_mean"), ("success", "eval_episode_success_rate"), ("MI ctx-policy", "mi_score_context_policy"),
           ("MI ctx-touch", "mi_score_context_touch"), ("touch freq", "eval_touch_freq")]
SERIES = [("agent", "parasite", ""), ("parasite", "parasite", "_par"), ("former", "former", "")]
WINDOWS = [(0, 100), (100, 400), (400, 1000), (1000, 1500)]
def load(path):
    con = sqlite3.connect(path); out = {}
    for sname, grp, suf in SERIES:
        for mname, key in METRICS:
            df = pd.read_sql("SELECT r.name, m.step, m.value FROM metrics m JOIN runs r ON r.id=m.run_id WHERE r.grp=? AND m.key=?", con, params=(grp, key + suf))
            if len(df): out[(sname, mname)] = df.assign(seed=df.name.str[-3:].astype(int))
    return out
A, B = load(new), load(old)
def win(df, lo, hi):
    d = df[(df.step >= lo) & (df.step < hi)]
    return d.groupby("seed").value.mean()
rows = []
for (s, m), df in A.items():
    for lo, hi in WINDOWS:
        a = win(df, lo, hi); b = win(B[(s, m)], lo, hi) if (s, m) in B else pd.Series(dtype=float)
        common = a.index.intersection(b.index)
        rows.append(dict(series=s, metric=m, epochs=f"{lo}-{hi}", new=f"{a.mean():.3f}+-{a.std():.3f}", old=f"{b.mean():.3f}+-{b.std():.3f}",
                         diff=round(float((a[common] - b[common]).mean()), 3) if len(common) else np.nan,
                         wins=f"{int((a[common] > b[common]).sum())}/{len(common)}", n=f"{len(a)}/{len(b)}"))
t = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(t.to_string(index=False))
print()
for s in ("agent", "former"):
    for lab, d in (("new", A), ("old", B)):
        df = d.get((s, "success"))
        if df is None: continue
        ps = win(df, 1000, 1500)
        print(f"{s} success epochs 1000-1500 per seed ({lab}):", " ".join(f"{v:.2f}" for v in ps.sort_index()), "| stuck (<0.3):", int((ps < 0.3).sum()))
