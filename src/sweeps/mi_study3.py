import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from scipy.special import digamma
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import mutual_info_score as mi
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
K = Path("/home/fmannella/tmp/kss")
SWEEPS = ["fig567_designB_greensmall_amp007", "fig567_designB_greensmall"]
NPERM = 100
rng = np.random.default_rng(0)


def ent(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def codes(x):
    return np.unique(x, return_inverse=True)[1]


def shrink(counts):
    n = counts.sum()
    p = counts / n
    t = 1.0 / counts.size
    den = (n - 1) * ((t - p) ** 2).sum()
    lam = 1.0 if den == 0 else min(1.0, max(0.0, (1 - (p**2).sum()) / den))
    return lam * t + (1 - lam) * p


def shrink_mi(x, y):
    tab = np.zeros((x.max() + 1, y.max() + 1))
    np.add.at(tab, (x, y), 1)
    return ent(shrink(tab.sum(1))) + ent(shrink(tab.sum(0))) - ent(shrink(tab.ravel()))


def block_index(ep):
    eps = np.unique(ep)
    members = {e: np.where(ep == e)[0] for e in eps}
    n_layout = 18

    def draw():
        pi = rng.permutation(n_layout)
        out = np.empty(len(ep), int)
        for e in eps:
            src = members[int(pi[int(e) % n_layout] + n_layout * (int(e) // n_layout))]
            out[members[e]] = src[np.arange(len(members[e])) % len(src)]
        return out

    return draw


def ross_mi(x, c, k=5):
    n = len(c)
    radius = np.zeros(n)
    kc = np.zeros(n, int)
    cnt = np.zeros(n, int)
    for lab in np.unique(c):
        idx = np.where(c == lab)[0]
        if len(idx) < 2:
            continue
        kk = min(k, len(idx) - 1)
        d, _ = cKDTree(x[idx]).query(x[idx], k=kk + 1)
        radius[idx], kc[idx], cnt[idx] = d[:, -1], kk, len(idx)
    ok = cnt > 0
    m = cKDTree(x).query_ball_point(x[ok], r=np.nextafter(radius[ok], 0), return_length=True)
    return float(digamma(ok.sum()) + digamma(kc[ok]).mean() - digamma(cnt[ok]).mean() - digamma(np.maximum(m, 1)).mean())


def ksg_mi(x, y, k=5):
    n = len(x)
    z = np.hstack([x, y])
    d, _ = cKDTree(z).query(z, k=k + 1, p=np.inf)
    eps = np.nextafter(d[:, -1], 0)
    nx = cKDTree(x).query_ball_point(x, r=eps, p=np.inf, return_length=True) - 1
    ny = cKDTree(y).query_ball_point(y, r=eps, p=np.inf, return_length=True) - 1
    return float(digamma(k) + digamma(n) - (digamma(nx + 1) + digamma(ny + 1)).mean())


def dcor(x, y):
    def cen(v):
        a = squareform(pdist(v))
        return a - a.mean(0) - a.mean(1)[:, None] + a.mean()

    a, b = cen(x), cen(y)
    return float((a * b).mean() / np.sqrt((a * a).mean() * (b * b).mean()))


def zs(v):
    v = np.asarray(v, float)
    if v.ndim == 1:
        v = v[:, None]
    return (v - v.mean(0)) / (v.std(0) + 1e-9)


def cv_bound_disc(x, y, ep):
    X = np.eye(y.max() + 1)[y]
    return cv_generic(x, X, ep)


# the folds are grouped by episode, so repeats of a layout leak between train and test: use mi_cv_layout.py for the layout-grouped bound
def cv_generic(c, F, ep):
    ll = []
    for tri, tei in GroupKFold(5).split(F, c, ep):
        if len(set(c[tri])) < 2:
            continue
        clf = LogisticRegression(C=1.0, max_iter=300).fit(F[tri], c[tri])
        P = clf.predict_proba(F[tei])
        ll.append(-np.log(np.clip(P[np.arange(len(tei)), np.searchsorted(clf.classes_, c[tei])], 1e-9, 1)).mean())
    return ent(np.bincount(c) / len(c)) - float(np.mean(ll))


def segments(v):
    ep, ended = v["ep"], v["ended"]
    new = np.r_[True, ended[:-1] | (ep[1:] != ep[:-1])]
    first = np.where(new)[0]
    seg = np.cumsum(new) - 1
    cnt = np.bincount(seg)
    tg = np.add.reduceat(v["touch_groups"], first, axis=0)
    jm = np.add.reduceat(v["joints"], first, axis=0) / cnt[:, None]
    touch = tg.argmax(1) + 1
    touch[tg.sum(1) == 0] = 0
    return dict(ep=ep[first], ctx=v["ctx"][first], exec_policy=v["exec_policy"][first], node=v["node"][first], touch=touch,
                grip=KMeans(10, random_state=0).fit_predict(jm), pol_pc=v["pol_pc"][first], joints=jm, touch_groups=np.log1p(tg))


def disc_row(x, y, ep, cv):
    draw = block_index(ep)
    raw = mi(x, y)
    nl = np.array([mi(x, y[draw()]) for _ in range(NPERM)])
    r = dict(N=len(x), raw=raw, miller_madow=raw - (len(set(x)) - 1) * (len(set(y)) - 1) / (2 * len(x)), shrinkage=shrink_mi(x, y), ami=ami(x, y),
             perm_null=nl.mean(), perm_excess=raw - nl.mean(), z=(raw - nl.mean()) / (nl.std() + 1e-12))
    if cv:
        r["cv_bound"] = cv_bound_disc(x, y, ep)
    return r


def cont_row(kind, a, b, ep):
    draw = block_index(ep)
    fn = {"ross": ross_mi, "ksg": ksg_mi, "dcor": dcor}[kind]
    est = fn(a, b)
    nl = np.array([fn(a, b[draw()]) for _ in range(NPERM)])
    return dict(N=len(a), est=est, null=nl.mean(), excess=est - nl.mean(), z=(est - nl.mean()) / (nl.std() + 1e-12))


rows_d, rows_c = [], []
for sweep in SWEEPS:
    for s in range(100, 110):
        v = dict(np.load(K / "mi_extract" / sweep / f"seed_{s}.npz"))
        for level, d in (("step", v), ("segment", segments(v))):
            D = {k: codes(d[k]) for k in ("ctx", "exec_policy", "node", "touch", "grip")}
            for xn, yn in (("ctx", "exec_policy"), ("ctx", "node"), ("ctx", "touch"), ("ctx", "grip"), ("exec_policy", "touch"), ("node", "touch"), ("exec_policy", "grip"), ("node", "grip")):
                r = disc_row(D[xn], D[yn], d["ep"], cv=(xn == "ctx") and level == "segment")
                r.update(sweep=sweep, seed=s, level=level, pair=f"{xn}-{yn}")
                rows_d.append(r)
        sg = segments(v)
        P, J, T = (zs(sg[k]) + 1e-6 * rng.standard_normal((len(sg[k]), 1 if np.ndim(sg[k]) == 1 else np.shape(sg[k])[1])) for k in ("pol_pc", "joints", "touch_groups"))
        for name, X in (("pol_pc", P), ("joints", J), ("touch_groups", T)):
            r = cont_row("ross", X, codes(sg["ctx"]), sg["ep"])
            r.update(sweep=sweep, seed=s, pair=f"ctx-{name}", method="ross_knn")
            r["cv_bound"] = cv_generic(codes(sg["ctx"]), X, sg["ep"])
            rows_c.append(r)
        for xn, yn, X, Y in (("pol_pc", "touch_groups", P, T), ("pol_pc", "joints", P, J), ("touch_groups", "joints", T, J)):
            for kind in ("ksg", "dcor"):
                r = cont_row(kind, X, Y, sg["ep"])
                r.update(sweep=sweep, seed=s, pair=f"{xn}-{yn}", method=kind)
                rows_c.append(r)
        print("done", sweep, s, flush=True)
dd, dc = pd.DataFrame(rows_d), pd.DataFrame(rows_c)
dd.to_csv(K / "mi_study3_discrete.csv", index=False)
dc.to_csv(K / "mi_study3_continuous.csv", index=False)
print("saved csv")
