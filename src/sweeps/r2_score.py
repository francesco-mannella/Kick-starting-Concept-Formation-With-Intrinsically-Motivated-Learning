import sqlite3, json, glob
import numpy as np, pandas as pd
R = "/home/fmannella/tmp/kss/fig567_r2_"
CONDS = {"base": "no-fric/no-sal", "sal": "no-fric/sal", "fric": "fric/no-sal", "fricsal": "fric/sal"}
M = {"align": "grid_comp_mean", "succ": "eval_episode_success_rate", "MIpol": "mi_score_context_policy", "MItouch": "mi_score_context_touch", "touch": "eval_touch_freq"}
S = [("agent", "parasite", ""), ("parasite", "parasite", "_par"), ("former", "former", "")]
LO, HI = 1000, 1500

def load(c):
    con = sqlite3.connect(f"file:{R}{c}/runs.db?mode=ro", uri=True); out = {}
    for sn, g, suf in S:
        for mn, k in M.items():
            df = pd.read_sql("select r.name,m.value from metrics m join runs r on r.id=m.run_id where r.grp=? and m.key=? and m.step>=? and m.step<?", con, params=(g, k + suf, LO, HI))
            out[(sn, mn)] = df.groupby("name").value.mean().rename(lambda n: int(n[-3:]))
    return out

D = {c: load(c) for c in CONDS}
print(f"Epochs {LO}-{HI}, 10 seeds (100-109), mean +- sd\n")
rows = []
for (sn, mn) in D["base"]:
    rows.append({"series": sn, "metric": mn, **{CONDS[c]: f"{D[c][(sn, mn)].mean():.3f}+-{D[c][(sn, mn)].std():.3f}" for c in CONDS}})
print(pd.DataFrame(rows).to_string(index=False))
for sn in ("agent", "former"):
    for c in CONDS:
        v = D[c][(sn, "succ")].sort_index()
        print(f"{sn} success per seed {CONDS[c]}: " + " ".join(f"{x:.2f}" for x in v) + f" | stuck(<0.3): {(v < 0.3).sum()}")
print("\nPaired contrasts (new minus reference), z = mean/se over 10 seeds; * = |z|>2\n")
CONTRASTS = [("fric", "base", "friction effect, no saliency"), ("fricsal", "sal", "friction effect, with saliency"),
             ("sal", "base", "saliency effect, no friction"), ("fricsal", "fric", "saliency effect, with friction")]
for a, b, lab in CONTRASTS:
    print("==", lab)
    for k in D[a]:
        d = (D[a][k] - D[b][k]).dropna(); se = d.std() / np.sqrt(len(d)); z = d.mean() / se if se > 0 else np.nan
        print(f"  {k[0]:8s}{k[1]:8s} diff {d.mean():+.3f} se {se:.3f} z {z:+.1f}{' *' if abs(z) > 2 else ''}")
print("\nCounterfactual (agent, epoch 1499): success by condition, mean over seeds, with intact minus condition per seed (se)\n")
KINDS = ["intact", "swap", "shuffle_within_context", "shuffle_all_used", "constant_mean"]
for c in CONDS:
    per = {}
    for f in sorted(glob.glob(f"{R}{c}/stats/agent_*_counterfactual_by_context.csv")):
        t = pd.read_csv(f); per[int(f.split("agent_")[1][:3])] = t
    if not per:
        print(CONDS[c], ": no counterfactual files"); continue
    print("==", CONDS[c], f"({len(per)} seeds)")
    for metric in ("success", "touch", "hit2"):
        line = []
        for kd in KINDS:
            v = pd.Series({s: t[t.cond == kd][metric].mean() for s, t in per.items()})
            if kd == "intact": line.append(f"intact {v.mean():.3f}"); base = v
            else:
                d = base - v; line.append(f"{kd} {v.mean():.3f} (cost {d.mean():+.3f}+-{d.std() / np.sqrt(len(d)):.3f})")
        print(f"  {metric:8s}", " | ".join(line))
    for ctx in (1, 2, 3):
        base = pd.Series({s: t[(t.cond == "intact") & (t.ctx == ctx)].success.mean() for s, t in per.items()})
        cm = pd.Series({s: t[(t.cond == "constant_mean") & (t.ctx == ctx)].success.mean() for s, t in per.items()})
        d = base - cm
        print(f"  ctx {ctx} success intact {base.mean():.3f}, constant cost {d.mean():+.3f}+-{d.std() / np.sqrt(len(d)):.3f}")
