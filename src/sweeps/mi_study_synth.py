import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import mutual_info_score as mi
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
REPS, NPERM, N_EP = 200, 100, 36


def ent(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def shrink(counts):
    n = counts.sum()
    p = counts / n
    t = 1.0 / counts.size
    den = (n - 1) * ((t - p) ** 2).sum()
    lam = 1.0 if den == 0 else min(1.0, max(0.0, (1 - (p**2).sum()) / den))
    return lam * t + (1 - lam) * p


def shrink_mi(x, y):
    kx, ky = x.max() + 1, y.max() + 1
    tab = np.zeros((kx, ky))
    np.add.at(tab, (x, y), 1)
    return ent(shrink(tab.sum(1))) + ent(shrink(tab.sum(0))) - ent(shrink(tab.ravel()))


def mm_mi(x, y):
    return mi(x, y) - (len(set(x)) - 1) * (len(set(y)) - 1) / (2 * len(x))


def perm_null(ctx_ep, ep, y, n=NPERM):
    eps = np.unique(ep)
    lab = np.array([ctx_ep[e] for e in eps])
    out = np.empty(n)
    for i in range(n):
        m = dict(zip(eps, rng.permutation(lab)))
        out[i] = mi(np.array([m[e] for e in ep]), y)
    return out


def cv_bound(c, y, ep):
    X = np.eye(y.max() + 1)[y]
    ll = []
    for tri, tei in GroupKFold(5).split(X, c, ep):
        if len(set(c[tri])) < 2:
            continue
        clf = LogisticRegression(C=1.0, max_iter=300).fit(X[tri], c[tri])
        P = clf.predict_proba(X[tei])
        ll.append(-np.log(np.clip(P[np.arange(len(tei)), np.searchsorted(clf.classes_, c[tei])], 1e-9, 1)).mean())
    p = np.bincount(c) / len(c)
    return ent(p) - float(np.mean(ll))


def make_model(alpha, K, seed=1):
    r = np.random.default_rng(seed)
    d = r.dirichlet(np.full(K, 0.3), size=3)
    q = (1 - alpha) / K + alpha * d
    return q


def true_mi(q, rho):
    K = q.shape[1]
    pyc = rho / K + (1 - rho) * q
    py = pyc.mean(0)
    return float(sum((1 / 3) * (pyc[c] * np.log(pyc[c] / py)).sum() for c in range(3)))


def simulate(q, rho, rep=1):
    K = q.shape[1]
    ctx_ep = np.repeat([0, 1, 2], N_EP // 3)
    rng.shuffle(ctx_ep)
    ep, c, y = [], [], []
    for e in range(N_EP):
        mode = rng.integers(K)
        for _ in range(rng.integers(2, 5)):
            yy = mode if rng.random() < rho else rng.choice(K, p=q[ctx_ep[e]])
            for _ in range(rep):
                ep.append(e); c.append(ctx_ep[e]); y.append(yy)
    return dict(enumerate(ctx_ep)), np.array(ep), np.array(c), np.array(y)


def run(alpha, K, rho=0.5, rep=1, with_cv=True):
    q = make_model(alpha, K)
    tr = true_mi(q, rho)
    res = {k: [] for k in ("raw", "miller_madow", "shrinkage", "ami", "perm_excess", "cv_bound")}
    fp_ep = fp_iid = 0
    for _ in range(REPS):
        ctx_ep, ep, c, y = simulate(q, rho, rep)
        raw = mi(c, y)
        nl = perm_null(ctx_ep, ep, y)
        res["raw"].append(raw)
        res["miller_madow"].append(mm_mi(c, y))
        res["shrinkage"].append(shrink_mi(c, y))
        res["ami"].append(ami(c, y))
        res["perm_excess"].append(raw - nl.mean())
        if with_cv:
            res["cv_bound"].append(cv_bound(c, y, ep))
        fp_ep += raw > np.quantile(nl, 0.95)
        iid = np.array([mi(rng.permutation(c), y) for _ in range(NPERM)])
        fp_iid += raw > np.quantile(iid, 0.95)
    return tr, {k: np.array(v) for k, v in res.items() if v}, fp_ep / REPS, fp_iid / REPS, len(y)


print("Synthetic benchmark: context (3 classes, 36 episodes) vs a category with K levels; each episode has a preferred level (episode effect rho=0.5).")
print("bias = mean(estimate) - true MI; sd over 200 replicates; fp = fraction of replicates where raw MI exceeds the 95th percentile of the null (should be 0.05 when there is no dependence)\n")
for rep, label in ((1, "segment level (N ~ 108)"), (25, "time-step level (each segment repeated 25 times, N ~ 2700)")):
    print(f"==== {label}")
    for K in (5, 28):
        for alpha in (0.0, 0.3, 0.7):
            tr, res, fpe, fpi, n = run(alpha, K, rep=rep, with_cv=(rep == 1))
            line = f"K={K:2d} dependence={alpha:.1f} N={n:4d} true MI={tr:.3f} | "
            line += "  ".join(f"{k}: {v.mean() - tr:+.3f}±{v.std():.3f}" for k, v in res.items())
            line += f" | fp episode-null {fpe:.2f}, fp iid-null {fpi:.2f}"
            print(line, flush=True)
    print()
