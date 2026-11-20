import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
K = Path("/home/fmannella/tmp/kss")
rng = np.random.default_rng(0)


def ent(c):
    p = np.bincount(c) / len(c)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def codes(x):
    return np.unique(x, return_inverse=True)[1]


def cv(c, F, layout, n_splits=6):
    ll = []
    for tri, tei in GroupKFold(n_splits).split(F, c, layout):
        clf = LogisticRegression(C=1.0, max_iter=400).fit(F[tri], c[tri])
        P = clf.predict_proba(F[tei])
        ll.append(-np.log(np.clip(P[np.arange(len(tei)), np.searchsorted(clf.classes_, c[tei])], 1e-9, 1)).mean())
    return ent(c) - float(np.mean(ll))


rows = []
for sw in ("fig567_designB_greensmall_amp007", "fig567_designB_greensmall"):
    for s in range(100, 110):
        v = dict(np.load(K / "mi_extract" / sw / f"seed_{s}.npz"))
        ep = v["ep"].astype(int)
        new = np.r_[True, v["ended"][:-1] | (ep[1:] != ep[:-1])]
        first = np.where(new)[0]
        e0, c0 = ep[first], codes(v["ctx"][first])
        layout = e0 % 18
        feats = {"exec_policy": np.eye(10)[v["exec_policy"][first]], "node": np.eye(100)[np.clip(v["node"][first], 0, 99)],
                 "touch": np.eye(5)[codes(v["touch"][first]) % 5],
                 "pol_pc": (v["pol_pc"][first] - v["pol_pc"][first].mean(0)) / (v["pol_pc"][first].std(0) + 1e-9)}
        for name, F in feats.items():
            shuffled = np.array([cv(rng.permutation(np.arange(3))[c0 % 3] if False else c0[rng.permutation(len(c0))], F, layout, 6) for _ in range(0)]) if False else None
            rows.append(dict(sweep=sw, seed=s, feature=name, cv_layout=cv(c0, F, layout), H_ctx=ent(c0)))
d = pd.DataFrame(rows)
print("Cross-validated information about context (nats), classifier tested on layouts it never saw (6 folds over the 18 layouts); H(context) = 1.10")
print(d.groupby(["feature", "sweep"]).cv_layout.agg(["mean", "std"]).round(3).to_string())
