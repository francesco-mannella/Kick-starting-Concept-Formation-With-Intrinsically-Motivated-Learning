#!/usr/bin/env python3
"""Rebuild the demo (all segments of the episode in one GIF) and episode GIFs of slides 45-47 from a Fig 5-7 sweep and the Slides request batches.

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

def sequence_gif(hi, ep, goals, out, sync):
    """All segments of the episode in one GIF on the episode video's own timeline (one frame per control step, same pauses at the same
    successes), so the two GIFs stay in sync when they loop together. Per-frame palette noise is suppressed and only the cropped panels are kept."""
    import tempfile
    from PIL import Image
    n, succ, hold = sync["n"], set(sync["succ"]), sync["hold_frames"]
    df = pd.read_csv(hi / "trajectory_df.csv"); g = df[df.episode_id == ep].goal_id.values
    runs, s = [], 0
    for i in range(1, len(g) + 1):
        if i == len(g) or g[i] != g[s]: runs.append((int(g[s]), i - s)); s = i
    offset = n - sum(r[1] for r in runs)
    seg_frames = {}
    for goal in goals:
        f = glob.glob(str(hi / "demos" / f"e{ep:02d}_g{goal:02d}_r0_*.gif"))
        if not f: continue
        im = Image.open(f[0]); fr = []
        for i in range(im.n_frames): im.seek(i); fr.append(im.convert("RGB").resize((1200, 1600), Image.LANCZOS))
        seg_frames[goal] = fr
    first = seg_frames[runs[0][0]][0]; step_frame = []; last = first
    for t in range(n):
        k = t - offset; frame = None
        if k >= 0:
            acc = 0
            for goal, length in runs:
                if k < acc + length:
                    fr = seg_frames.get(goal); frame = fr[min(k - acc, len(fr) - 1)] if fr else None; break
                acc += length
        step_frame.append(frame if frame is not None else last); last = step_frame[-1]
    used = [(.04, .04, .27, .10), (.03, .235, .32, .485), (.03, .515, .995, .745), (.04, .805, .97, .995)]
    tmp = Path(tempfile.mkdtemp()); k = 0; prev = None
    for t, fr in enumerate(step_frame):
        a = np.array(fr); mask = np.full(a.shape, 255, np.uint8)
        for x0, y0, x1, y1 in used:
            b = (int(y0 * 1600), int(y1 * 1600), int(x0 * 1200), int(x1 * 1200)); mask[b[0]:b[1], b[2]:b[3]] = a[b[0]:b[1], b[2]:b[3]]
        cur = mask.astype(np.int16)
        if prev is not None: cur = np.where((np.abs(cur - prev).max(-1) <= 45)[..., None], prev, cur)
        prev = cur
        for _ in range(hold if t in succ else 1):
            Image.fromarray(cur.astype(np.uint8)).save(tmp / f"f_{k:04d}.png"); k += 1
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(sync["gif_fps"]), "-i", str(tmp / "f_%04d.png"), "-vf",
                    "split[a][b];[a]palettegen=max_colors=128:stats_mode=full[p];[b][p]paletteuse=dither=none:diff_mode=rectangle", "-loop", "0", str(out)], check=True)
    if shutil.which("gifsicle"): subprocess.run(["gifsicle", "-O3", str(out), "-o", str(out)], check=False, stderr=subprocess.DEVNULL)
    shutil.rmtree(tmp)
    return k

def build(root, obj, seed, rep, ep, goal):
    s, ctx = OBJ[obj]; st = storage(root, seed); ck = st / "final_ckpt"; work = st / f"best_{obj}"; work.mkdir(exist_ok=True)
    for f in ("main.dump.npy", "weights.npy"): shutil.copy(ck / f, work / f)
    if not (work / "trajectory_df.csv").exists(): run([sys.executable, str(SRC / "run_evaluation.py"), "-n", "1", "-s", str(300 + rep), "-p"], work, work / "render.log")
    hi = work / "final_hi"; shutil.rmtree(hi, ignore_errors=True); (hi / "episodes").mkdir(parents=True)
    for g in (work / "episodes").glob("episode_*_0.gif"): shutil.copy(g, hi / "episodes" / g.name)
    tr = pd.read_csv(work / "trajectory_df.csv"); tr["Unnamed: 0"] = np.arange(len(tr)); tr.to_csv(hi / "trajectory_df.csv", index=False)
    shutil.copy(work / "weights.npy", hi / "weights.npy")
    for f in ("demos", "logs", "goal_frames"): (hi / f).mkdir()
    seg = pd.read_csv(ck / "eval/segment_scores.csv"); goals = sorted(int(x) for x in seg[(seg.rep == rep) & (seg.episode == ep)].goal)
    procs = [subprocess.Popen([sys.executable, str(HERE / "run_hi.py"), "-e", str(ep), "-r", "0", "-g", str(g)], cwd=hi, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT) for g in goals]
    for p in procs: p.wait()
    png = glob.glob(str(hi / "demos" / f"final_e{ep:02d}_g{goals[0]:02d}_r0_*.png"))[0]
    ev = OUT / f"{obj}_ev"; run([sys.executable, str(HERE / "episode_video2.py"), "--episode", str(ep), "--rep", str(rep), "--res", "4", "--out", str(ev)], ck, OUT / f"{obj}_video.log")
    run([sys.executable, str(HERE / "compose.py"), "--run", str(ev / "run.pkl"), "--out", str(OUT / f"{obj}_episode.mp4"), "--title", "", "--gif", str(OUT / f"{obj}_episode.gif"), "--gif-width", "600"], HERE, OUT / f"{obj}_compose.log")
    nfr = sequence_gif(hi, ep, goals, OUT / f"{obj}_demo.gif", json.load(open(ev / "sync.json")))
    def _dur(p):
        im = Image.open(p); t = 0
        for i in range(im.n_frames): im.seek(i); t += im.info.get("duration", 0)
        return t
    print(obj, "GIF total duration ms: demo", _dur(OUT / f"{obj}_demo.gif"), "episode", _dur(OUT / f"{obj}_episode.gif"), flush=True)
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
    ids = json.load(open(a.ids)); state = HERE / "state.json"; st_ = json.load(open(state)) if state.exists() else {"prefix": "k2"}; old = st_["prefix"]; old_has_traj = st_.get("traj", True)
    new = "r" + str(int(time.time()))[-7:]; subprocess.run([sys.executable, "layout.py"], cwd=HERE, check=True)
    for s in ("45", "46", "47"):
        w, h = Image.open(OUT / f"{ {'45': 'blue-cube', '46': 'red-triangle', '47': 'green-stick'}[s] }_episode.gif").size
        dw, dh = Image.open(OUT / f"{ {'45': 'blue-cube', '46': 'red-triangle', '47': 'green-stick'}[s] }_demo.gif").size
        env = {**os.environ, "DEMO_BASE": f"{dw * 25},{dh * 25}", "HAS_TRAJ": "1" if old_has_traj else "0", "EPI_AR": str(w / h), "REFRESH": "1", "PREFIX_OLD": old, "PREFIX_NEW": new, "DEMO_ID": ids[s]["demo"], "EPI_ID": ids[s]["epi"]}
        out = subprocess.run([sys.executable, "gen2.py", s], cwd=HERE, env=env, capture_output=True, text=True, check=True).stdout
        (OUT / f"slide{s}.json").write_text(out); print("wrote", OUT / f"slide{s}.json")
    json.dump({"prefix": new, "traj": False}, open(state, "w")); print("object-id prefix of the images now on the slides after you apply the batches:", new)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sp = p.add_subparsers(dest="cmd", required=True)
    a_ = sp.add_parser("assets"); a_.add_argument("--root", required=True); a_.add_argument("--pick", action="append"); a_.add_argument("--min-successes", type=int, default=1)
    r_ = sp.add_parser("requests"); r_.add_argument("--ids", required=True)
    a = p.parse_args(); (assets if a.cmd == "assets" else requests_)(a)
