import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

H = Path.home() / "tmp/kss"
NEW, REF, REF2 = H / "fig567_designB", H / "fig567_r2_fric", H / "fig567_r2_base"
LO, HI = 1000, 1500
M = {"align": "grid_comp_mean", "succ": "eval_episode_success_rate", "MIpol": "mi_score_context_policy", "MItouch": "mi_score_context_touch", "touch": "eval_touch_freq"}
S = [("agent", "parasite", ""), ("parasite", "parasite", "_par"), ("former", "former", "")]
SEEDS = range(100, 110)


def training(root):
    con = sqlite3.connect(f"file:{root}/runs.db?mode=ro", uri=True)
    out = {}
    for sn, g, suf in S:
        for mn, k in M.items():
            df = pd.read_sql("select r.name,m.value from metrics m join runs r on r.id=m.run_id where r.grp=? and m.key=? and m.step>=? and m.step<?", con, params=(g, k + suf, LO, HI))
            out[(sn, mn)] = df.groupby("name").value.mean().rename(lambda n: int(n[-3:]))
    return pd.DataFrame(out)


def static(root):
    rows = {}
    for s in SEEDS:
        j = json.load(open(root / "stats" / f"agent_{s}_static.json"))
        rows[s] = {"eta2_ctx_all_excess": j["eta2_context_all"] - j["eta2_context_all_p95"], "eta2_ctx_first_excess": j["eta2_context_first"] - j["eta2_context_first_p95"],
                   "eta2_goalproto_excess": j["eta2_context_goal_prototype"] - j["eta2_context_goal_prototype_p95"], "eta2_posture_excess": j["eta2_posture"] - j["eta2_posture_p95"],
                   "mi_ctx_policy_excess": j["mi_context_policy_excess"], "cos_same_minus_diff": j["cos_first_same_context"] - j["cos_first_diff_context"], "nodes_used": j["nodes_used"]}
    return pd.DataFrame(rows).T


def counterfactual(root):
    rows = {}
    for s in SEEDS:
        t = pd.read_csv(root / "stats" / f"agent_{s}_counterfactual_by_context.csv")
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


pd.set_option("display.width", 220)
print(f"Design B (plate still at (14,12), big triangle movable at (7,12), wedge controllable at (14,4)) vs the original layout, same seeds 100-109, champion + friction, 1500 epochs.")
print(f"Reference = fig567_r2_fric (original layout, friction, no saliency); no-friction original = fig567_r2_base. Training metrics: epochs {LO}-{HI}; policy statistics and counterfactuals: epoch 1499 agent.\n")
for name, fn in (("training", training), ("static", static), ("counterfactual", counterfactual)):
    new, ref = fn(NEW), fn(REF)
    try:
        ref2 = fn(REF2)
    except Exception:
        ref2 = None
    diff = new - ref
    table = pd.DataFrame({"design B": new.mean(), "original+friction": ref.mean()})
    if ref2 is not None:
        table["original, no friction"] = ref2.mean()
    table["B - orig"] = diff.mean()
    table["se"] = diff.std() / np.sqrt(len(diff))
    table["z"] = table["B - orig"] / table["se"]
    table["*"] = np.where(table["z"].abs() > 2, "*", "")
    print(f"== {name}  (* = |z| > 2, paired over 10 seeds)")
    print(table.round(3).to_string())
    print()
new_s, ref_s = training(NEW), training(REF)
for sn in ("agent", "former"):
    print(f"{sn} success per seed: design B " + " ".join(f"{v:.2f}" for v in new_s[(sn, 'succ')].sort_index()) + " | original " + " ".join(f"{v:.2f}" for v in ref_s[(sn, 'succ')].sort_index()))
print("stuck formers (success < 0.3): design B", int((new_s[("former", "succ")] < 0.3).sum()), "| original+friction", int((ref_s[("former", "succ")] < 0.3).sum()))
