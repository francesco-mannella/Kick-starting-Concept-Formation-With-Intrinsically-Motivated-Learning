import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

K = Path("/home/fmannella/tmp/kss")
ROOT = K / sys.argv[1]
SEEDS = range(100, 110)


def weights(root, s):
    return np.load(root / f"parasite_{s}/simulations/parasite_{s}/storage/001499/weights.npy", allow_pickle=True)[0]["policy"]


print(f"== postures of the 100 nodes (final glyph, {ROOT.name}); distances in simulator units (the arm is about 24 long)")
post, rows = {}, []
for s in SEEDS:
    d = pickle.load(open(ROOT / f"policy_maps/seed_{s}/policy_glyphs.pkl", "rb"))
    last = d["layers"][-1]
    X = np.array([g[last]["joints"].ravel() for g in d["glyphs"]])
    post[s] = X
    D = pdist(X)
    clusters = fcluster(linkage(X, "single"), 1.0, "distance")
    sizes = np.bincount(clusters)[1:]
    ev = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
    rows.append((s, D.mean(), np.quantile(D, 0.9), D.max(), len(sizes), sizes.max() / 100, ev[0] / ev.sum()))
print(f"{'seed':>5} {'mean node-node':>15} {'90th pct':>9} {'max':>6} {'clusters(<1)':>13} {'largest cluster':>16} {'PC1 share':>10}")
for r in rows:
    print(f"{r[0]:5d} {r[1]:15.2f} {r[2]:9.2f} {r[3]:6.2f} {r[4]:13d} {r[5]:16.2f} {r[6]:10.2f}")
within = np.mean([r[1] for r in rows])
M = np.array([post[s].mean(0) for s in SEEDS])
B = squareform(pdist(M))
between = B[np.triu_indices(len(M), 1)]
print(f"\nmean node-to-node distance inside a seed: {within:.2f}")
print(f"distance between the mean postures of two seeds: mean {between.mean():.2f}, min {between.min():.2f}, max {between.max():.2f}")
print(f"ratio between seeds / inside a seed: {between.mean() / within:.1f}")

print("\n== policy weights (375 x 100 per seed), all sweeps with seeds 100-109")
for name in ("fig567_designB_greensmall", "fig567_designB", "fig567_designB_fovea8", "fig567_r2_fric", "fig567_r2_base"):
    root = K / name
    share, cosine, resid, commons = [], [], [], []
    for s in SEEDS:
        W = weights(root, s)
        c = W.mean(1, keepdims=True)
        share.append(float((c ** 2).sum() / (W ** 2).sum(0).mean()))
        Wn = W / np.linalg.norm(W, axis=0)
        cosine.append(float((Wn.T @ Wn)[np.triu_indices(100, 1)].mean()))
        R = W - c
        Rn = R / (np.linalg.norm(R, axis=0) + 1e-9)
        resid.append(float((Rn.T @ Rn)[np.triu_indices(100, 1)].mean()))
        commons.append((c / np.linalg.norm(c)).ravel())
    C = np.array(commons)
    cs = (C @ C.T)[np.triu_indices(len(C), 1)]
    print(f"{name:28s} common vector share of the weights {np.mean(share):.2f}  node-node cosine {np.mean(cosine):.2f}  after removing the common vector {np.mean(resid):.2f}  cosine of the common vectors between seeds {cs.mean():.2f}")

side = 600
sheet = Image.new("RGB", (side * 5, side * 2 + 40), "white")
dr = ImageDraw.Draw(sheet)
for k, s in enumerate(SEEDS):
    im = Image.open(ROOT / f"policy_maps/seed_{s}/policy_map.png").convert("RGB").resize((side - 10, side - 10))
    sheet.paste(im, ((k % 5) * side + 5, (k // 5) * side + 45))
    dr.text(((k % 5) * side + 10, (k // 5) * side + 28), f"seed {s}", fill="black")
dr.text((10, 5), f"Policy maps of the 10 seeds ({ROOT.name}, active agent, epoch 1499): each tile = arm and gripper after the node's policy runs open loop", fill="black")
sheet.save(ROOT / "figures" / "policy_maps_overview.png")
print("overview saved")
