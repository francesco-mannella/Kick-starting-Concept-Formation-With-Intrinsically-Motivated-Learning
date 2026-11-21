import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

K = Path("/home/fmannella/tmp/kss")
CK = {
    "amp03": lambda s: K / f"satplan_probe/agent_amp03_eta0.05_00{s}/simulations/agent_amp03_eta0.05_00{s}/storage/000999",
    "amp07": lambda s: K / f"satplan_probe/agent_amp07_eta0.05_00{s}/simulations/agent_amp07_eta0.05_00{s}/storage/000999",
    "amp8": lambda s: K / f"fig567_champion_gate/parasite_00{s}/simulations/parasite_00{s}/storage/001499",
}
import re
LOG = {"amp03": lambda s: K / f"satplan_probe/agent_amp03_eta0.05_00{s}/sim.log", "amp07": lambda s: K / f"satplan_probe/agent_amp07_eta0.05_00{s}/sim.log", "amp8": lambda s: K / f"fig567_champion_gate/parasite_00{s}/sim.log"}
print("amplitude per group, from the runs' own logs:")
for g, f in LOG.items():
    txt = open(f(0)).read()
    print(f"  {g}: policy_params_amplitude = {re.search(r'policy_params_amplitude.: ([0-9.]+)', txt).group(1)}, output = {re.search(r'grip_out_fn.: .(\w+)', txt).group(1)}")

print(f"\n{'run':10s} {'mean node-node':>15} {'90th pct':>9} {'clusters(<1)':>13} {'largest':>8} {'PC1 share':>10} | weights: common share  cosine")
rows = []
for g, f in CK.items():
    for s in range(5):
        d = pickle.load(open(K / f"policymap_amp/{g}_{s}/policy_glyphs.pkl", "rb"))
        last = d["layers"][-1]
        X = np.array([n[last]["joints"].ravel() for n in d["glyphs"]])
        D = pdist(X)
        sizes = np.bincount(fcluster(linkage(X, "single"), 1.0, "distance"))[1:]
        ev = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
        W = np.load(f(s) / "weights.npy", allow_pickle=True)[0]["policy"]
        c = W.mean(1, keepdims=True)
        share = float((c ** 2).sum() / (W ** 2).sum(0).mean())
        Wn = W / np.linalg.norm(W, axis=0)
        cos = float((Wn.T @ Wn)[np.triu_indices(100, 1)].mean())
        rows.append(dict(group=g, seed=s, node_dist=D.mean(), p90=np.quantile(D, 0.9), clusters=len(sizes), largest=sizes.max() / 100, pc1=ev[0] / ev.sum(), share=share, cos=cos))
        print(f"{g}_{s:<5d} {D.mean():15.2f} {np.quantile(D, .9):9.2f} {len(sizes):13d} {sizes.max() / 100:8.2f} {ev[0] / ev.sum():10.2f} | {share:20.2f} {cos:8.2f}")
t = pd.DataFrame(rows)
print("\nmeans over 5 seeds:")
print(t.groupby("group")[["node_dist", "p90", "clusters", "largest", "pc1", "share", "cos"]].mean().round(2).to_string())

print("\nsaturation census of the same kind of runs (pre-tanh output of the actuator on the runs' own data; from the earlier census files)")
for name, pat in (("amplitude 0.03 (satplan03)", "sat_census_satplan03.csv"), ("amplitude 0.07 (satplan)", "sat_census_satplan.csv"), ("amplitude 8 (gate_long)", "sat_census_gate_long.csv")):
    p = K / "signal_analysis" / pat
    df = pd.read_csv(p)
    df = df[df.run.str.contains("amp07")] if pat == "sat_census_satplan.csv" else df
    print(f"  {name:26s} runs {len(df)}  median |pre-tanh| {df.med_pre.median():.1f}  share of outputs with |pre-tanh| > 3: {df.frac_gt3.mean():.2f}  > 1.5: {df.frac_gt15.mean():.2f}")

side = 520
sheet = Image.new("RGB", (side * 5, side * 3 + 40), "white")
dr = ImageDraw.Draw(sheet)
dr.text((10, 5), "Rows: amplitude 0.03, 0.07 (unsaturated, epoch 999) and 8 (saturated, epoch 1499); 5 seeds each; same 5-output gate base.", fill="black")
for r, g in enumerate(("amp03", "amp07", "amp8")):
    for s in range(5):
        im = Image.open(K / f"policymap_amp/{g}_{s}/policy_map.png").convert("RGB").resize((side - 8, side - 8))
        sheet.paste(im, (s * side + 4, r * side + 40))
        dr.text((s * side + 8, r * side + 26), f"{g} seed {s}", fill="black")
sheet.save(K / "fig567_designB_greensmall/figures/policy_maps_amp07_vs_amp8.png")
print("overview saved to the 8082 figures folder")
