import glob
import json
import sqlite3

import numpy as np
import pandas as pd

R = "/home/fmannella/tmp/kss/pilot_pos_"
ARMS = {"ctrl": "original layout", "kind": "per-kind positions"}
SEEDS = range(300, 305)
LO, HI = 150, 300
M = {"align": "grid_comp_mean", "succ": "eval_episode_success_rate", "MIpol": "mi2_context_policy", "touch": "eval_touch_freq"}


def training(arm):
    con = sqlite3.connect(f"file:{R}{arm}/runs.db?mode=ro", uri=True)
    out = {}
    for mn, key in M.items():
        df = pd.read_sql("select r.name,m.value from metrics m join runs r on r.id=m.run_id where r.grp='parasite' and m.key=? and m.step>=? and m.step<?",
                         con, params=(key, LO, HI))
        out[mn] = df.groupby("name").value.mean().rename(lambda n: int(n[-3:]))
    return pd.DataFrame(out)


def static(arm):
    rows = {}
    for s in SEEDS:
        j = json.load(open(f"{R}{arm}/stats/agent_{s}_static.json"))
        rows[s] = {"eta2_ctx_all_excess": j["eta2_context_all"] - j["eta2_context_all_p95"],
                   "eta2_ctx_first_excess": j["eta2_context_first"] - j["eta2_context_first_p95"],
                   "eta2_goalproto_excess": j["eta2_context_goal_prototype"] - j["eta2_context_goal_prototype_p95"],
                   "eta2_posture_excess": j["eta2_posture"] - j["eta2_posture_p95"],
                   "mi_ctx_policy_excess": j["mi_context_policy_excess"],
                   "cos_same_minus_diff": j["cos_first_same_context"] - j["cos_first_diff_context"],
                   "nodes_used": j["nodes_used"]}
    return pd.DataFrame(rows).T


def counterfactual(arm):
    rows = {}
    for s in SEEDS:
        t = pd.read_csv(f"{R}{arm}/stats/agent_{s}_counterfactual_by_context.csv")
        row = {}
        for kind in ("swap", "shuffle_within_context", "shuffle_all_used", "constant_mean"):
            for ctx in (1, 2, 3, "all"):
                a = t[t.cond == "intact"] if ctx == "all" else t[(t.cond == "intact") & (t.ctx == ctx)]
                b = t[t.cond == kind] if ctx == "all" else t[(t.cond == kind) & (t.ctx == ctx)]
                row[f"cost_{kind}_ctx{ctx}"] = a.success.mean() - b.success.mean()
        row["intact_success"] = t[t.cond == "intact"].success.mean()
        row["hit2_cost_constant"] = t[t.cond == "intact"].hit2.mean() - t[t.cond == "constant_mean"].hit2.mean()
        rows[s] = row
    return pd.DataFrame(rows).T


tables = {name: {a: f(a) for a in ARMS} for name, f in (("training", training), ("static", static), ("counterfactual", counterfactual))}
pd.set_option("display.width", 220)
print(f"Pilot: 5 seeds (300-304), 300 epochs, friction on; training metrics = mean of epochs {LO}-{HI}; static and counterfactual at epoch 299\n")
for name, d in tables.items():
    both = pd.concat({ARMS[a]: d[a].mean() for a in ARMS}, axis=1)
    diff = d["kind"] - d["ctrl"]
    both["diff (kind - ctrl)"] = diff.mean()
    both["se"] = diff.std() / np.sqrt(len(diff))
    both["z"] = both["diff (kind - ctrl)"] / both["se"]
    both["*"] = np.where(both["z"].abs() > 2, "*", "")
    print(f"== {name}")
    print(both.round(3).to_string())
    print()
print("per-seed values of the main measures")
for key, tab in (("succ", "training"), ("touch", "training"), ("eta2_ctx_all_excess", "static"), ("mi_ctx_policy_excess", "static"), ("cost_constant_mean_ctxall", "counterfactual"), ("cost_swap_ctxall", "counterfactual")):
    print(f"{key:28s} ctrl", " ".join(f"{v:+.2f}" for v in tables[tab]["ctrl"][key].sort_index()), "| kind", " ".join(f"{v:+.2f}" for v in tables[tab]["kind"][key].sort_index()))
