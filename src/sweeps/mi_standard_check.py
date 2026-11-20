import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mi_standard  # noqa: E402

K = Path("/home/fmannella/tmp/kss")
study = pd.read_csv(K / "mi_study3_discrete.csv")
study = study[study.level == "segment"]
MAP = {"mi2_context_policy": "ctx-exec_policy", "mi2_context_touch": "ctx-touch", "mi2_policy_touch": "exec_policy-touch",
       "mi2_context_gripper": "ctx-grip", "mi2_policy_gripper": "exec_policy-grip"}
rows = []
for sw in ("fig567_designB_greensmall_amp007", "fig567_designB_greensmall"):
    for s in range(100, 110):
        v = np.load(K / "mi_extract" / sw / f"seed_{s}.npz")
        ep = v["ep"].astype(int)
        ss = np.zeros((len(ep), 40), np.float32)
        ss[:, ::10] = v["touch_groups"]
        res = mi_standard.layout_excess(ep, v["ended"], v["ctx"].astype(int), ep % 18, v["exec_policy"], ss, v["joints"])
        for key, pair in MAP.items():
            old = study[(study.sweep == sw) & (study.seed == s) & (study.pair == pair)].iloc[0]
            rows.append(dict(sweep=sw[-6:], seed=s, key=key, module=res[key], study=old.perm_excess, raw_module=res[key + "_raw"], raw_study=old.raw))
d = pd.DataFrame(rows)
d["diff"] = d.module - d.study
print("module vs study (layout-level permutation excess, segment level), per-seed absolute difference:")
print(d.groupby("key")[["module", "study", "diff"]].agg(["mean"]).round(3).to_string())
print("max |diff| over 100 values:", round(float(d["diff"].abs().max()), 3), "| max |raw diff|:", round(float((d.raw_module - d.raw_study).abs().max()), 6))
print("n_layouts, n_segments example:", res["mi2_n_layouts"], res["mi2_n_segments"])
