#!/usr/bin/env python3
"""Rebuild the demo/episode GIFs of slides 45-47 from a Fig 5-7 sweep and the Slides request batches.

  python3 refresh_slides.py assets   --root SWEEP [--pick blue-cube=SEED,REP,EPISODE,GOAL ...] [--min-successes 1]
  python3 refresh_slides.py requests --ids ids.json

assets   : for blue-cube / red-triangle / green-stick picks the best episode of the sweep (or your --pick), renders the segment
           demo GIF (hi-res, 6 fps) and the episode GIF (zoomed, touch dots, 1 s pause at each success) into ./out/, and writes
           ./out/manifest.json and ./manifest.json (figure paths used for the panel crops).
           Needs SWEEP/demos/work/seed_N/... made by policy_demos.py (the evaluation of each seed, 20 reps).
requests : after uploading out/<obj>_demo.gif and out/<obj>_episode.gif to a link-readable Drive folder, put their file ids in
           ids.json {"45": {"demo": "...", "epi": "..."}, "46": {...}, "47": {...}} and run this; it writes out/slide45.json ...
           (Slides batchUpdate requests: delete the previous k2* images, recreate crops, frames, labels, episode GIF).
           Apply them with the Slides connector (update_presentation, requests = the file content, writeControl = current revision).
"""
import argparse, glob, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
HERE = Path(__file__).resolve().parent; OUT = HERE / "out"
SRC = HERE.parent
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MPLBACKEND": "agg", "SRC": str(SRC), "PYTHONPATH": f"{SRC}:{SRC.parent}/tools/box2dsim"}
OBJ = {"blue-cube": ("45", 1), "red-triangle": ("46", 2), "green-stick": ("47", 3)}
F1 = list(range(20, 40)); F2 = list(range(0, 20))

def storage(root, seed):
    n = f"parasite_{seed:03d}"; return Path(root) / "demos/work" / f"seed_{seed}" / n / "simulations" / n / "storage"

def select(root):
    rows = []
    for seed in range(10):
        ev = storage(root, seed) / "final_ckpt/eval"
        if not (ev / "segment_scores.csv").exists(): continue
        seg = pd.read_csv(ev / "segment_scores.csv"); tr = pd.read_csv(ev / "trajectory_df.csv")
        tr["rep"] = tr.e_seed.map({s: i for i, s in enumerate(sorted(tr.e_seed.unique()))})
        S = tr[[f"s{i}" for i in range(40)]].abs().values > 0.01
        tr["grasp"] = S[:, F1].any(1) & S[:, F2].any(1)
        g = tr.groupby(["rep", "episode_id"]).grasp.sum().rename("grasp_steps").reset_index()
        good = seg[seg.good].groupby(["rep", "episode"]).agg(n_good=("score", "size"), sum_score=("score", "sum")).reset_index()
        g = g.merge(good, left_on=["rep", "episode_id"], right_on=["rep", "episode"], how="left").drop(columns="episode").fillna(0); g["seed"] = seed; rows.append(g)
    d = pd.concat(rows); d["context"] = d.episode_id % 3 + 1; d["score"] = d.sum_score + 0.1 * d.grasp_steps
    return d.sort_values("score", ascending=False)

def run(cmd, cwd, log):
    subprocess.run(cmd, cwd=cwd, env=ENV, check=True, stdout=open(log, "w"), stderr=subprocess.STDOUT)

def successes(root, seed, rep, ep):
    ck = storage(root, seed) / "final_ckpt"; out = OUT / f"probe_{seed}_{rep}_{ep}"
    run([sys.executable, str(HERE / "episode_video2.py"), "--episode", str(ep), "--rep", str(rep), "--res", "0.3", "--out", str(out)], ck, OUT / "probe.log")
    import pickle; return int(np.sum(pickle.load(open(out / "run.pkl", "rb"))["out"]["matches"][0]))

def build(root, obj, seed, rep, ep, goal):
    s, ctx = OBJ[obj]; st = storage(root, seed); ck = st / "final_ckpt"; work = st / f"best_{obj}"; work.mkdir(exist_ok=True)
    for f in ("main.dump.npy", "weights.npy"): shutil.copy(ck / f, work / f)
    if not (work / "trajectory_df.csv").exists(): run([sys.executable, str(SRC / "run_evaluation.py"), "-n", "1", "-s", str(300 + rep), "-p"], work, work / "render.log")
    hi = work / "final_hi"; shutil.rmtree(hi, ignore_errors=True); (hi / "episodes").mkdir(parents=True)
    for g in (work / "episodes").glob("episode_*_0.gif"): shutil.copy(g, hi / "episodes" / g.name)
    tr = pd.read_csv(work / "trajectory_df.csv"); tr["Unnamed: 0"] = np.arange(len(tr)); tr.to_csv(hi / "trajectory_df.csv", index=False)
    shutil.copy(work / "weights.npy", hi / "weights.npy")
    for f in ("demos", "logs", "goal_frames"): (hi / f).mkdir()
    run([sys.executable, str(HERE / "run_hi.py"), "-e", str(ep), "-r", "0", "-g", str(goal)], hi, OUT / f"{obj}_hi.log")
    gif = glob.glob(str(hi / "demos" / f"e{ep:02d}_g{goal:02d}_r0_*.gif"))[0]; png = glob.glob(str(hi / "demos" / f"final_e{ep:02d}_g{goal:02d}_r0_*.png"))[0]
    im = Image.open(gif); fr = []
    step = 1 if im.n_frames <= 30 else 2
    for i in range(0, im.n_frames, step): im.seek(i); fr.append(im.convert("RGB").copy())
    dur = [160 * step] * (len(fr) - 1) + [2000]; fr[0].save(OUT / f"{obj}_demo.gif", save_all=True, append_images=fr[1:], duration=dur, loop=0)
    ev = OUT / f"{obj}_ev"; run([sys.executable, str(HERE / "episode_video2.py"), "--episode", str(ep), "--rep", str(rep), "--res", "4", "--out", str(ev)], ck, OUT / f"{obj}_video.log")
    run([sys.executable, str(HERE / "compose.py"), "--run", str(ev / "run.pkl"), "--out", str(OUT / f"{obj}_episode.mp4"), "--title", f"{obj} seed{seed} rep{rep} ep{ep}"], HERE, OUT / f"{obj}_compose.log")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(OUT / f"{obj}_episode.mp4"), "-vf", "fps=12,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4", str(OUT / f"{obj}_episode.gif")], check=True)
    for f in (OUT / f"{obj}_demo.gif", OUT / f"{obj}_episode.gif"): assert f.stat().st_size < 10e6, f"{f.name} over the 10 MB Drive upload limit"
    return png

def assets(a):
    OUT.mkdir(exist_ok=True); d = select(a.root); picks = {}
    for p in a.pick or []: o, v = p.split("="); picks[o] = [int(x) for x in v.split(",")]
    manifest = {}; fig = {}
    for obj, (s, ctx) in OBJ.items():
        if obj in picks: seed, rep, ep, goal = picks[obj]
        else:
            for _, r in d[d.context == ctx].iterrows():
                seed, rep, ep = int(r.seed), int(r.rep), int(r.episode_id)
                if a.min_successes and successes(a.root, seed, rep, ep) < a.min_successes: continue
                seg = pd.read_csv(storage(a.root, seed) / "final_ckpt/eval/segment_scores.csv"); seg = seg[(seg.rep == rep) & (seg.episode == ep) & seg.good]
                if len(seg) == 0: continue
                goal = int(seg.sort_values("score").iloc[-1].goal); break
        print("building", obj, seed, rep, ep, goal, flush=True)
        fig[s] = {"figure": str(build(a.root, obj, seed, rep, ep, goal))}; manifest[obj] = dict(slide=s, seed=seed, rep=rep, episode=ep, goal=goal)
    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1); json.dump(fig, open(HERE / "manifest.json", "w"), indent=1)
    print("done. Upload out/*_demo.gif and out/*_episode.gif to the Drive folder, fill ids.json, then run: refresh_slides.py requests --ids ids.json")

def requests_(a):
    ids = json.load(open(a.ids)); state = HERE / "state.json"; old = json.load(open(state))["prefix"] if state.exists() else "k2"
    new = "r" + str(int(time.time()))[-7:]; subprocess.run([sys.executable, "layout.py"], cwd=HERE, check=True)
    for s in ("45", "46", "47"):
        env = {**os.environ, "REFRESH": "1", "PREFIX_OLD": old, "PREFIX_NEW": new, "DEMO_ID": ids[s]["demo"], "EPI_ID": ids[s]["epi"]}
        out = subprocess.run([sys.executable, "gen2.py", s], cwd=HERE, env=env, capture_output=True, text=True, check=True).stdout
        (OUT / f"slide{s}.json").write_text(out); print("wrote", OUT / f"slide{s}.json")
    json.dump({"prefix": new}, open(state, "w")); print("object-id prefix of the images now on the slides after you apply the batches:", new)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sp = p.add_subparsers(dest="cmd", required=True)
    a_ = sp.add_parser("assets"); a_.add_argument("--root", required=True); a_.add_argument("--pick", action="append"); a_.add_argument("--min-successes", type=int, default=1)
    r_ = sp.add_parser("requests"); r_.add_argument("--ids", required=True)
    a = p.parse_args(); (assets if a.cmd == "assets" else requests_)(a)
