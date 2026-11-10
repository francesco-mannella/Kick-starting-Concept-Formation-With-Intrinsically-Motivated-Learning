import argparse, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
import box2dsim.envs.mkvideo as mk
import box2dsim.envs.Simulator as Sim
from SMMain import Main
from SMEnv import SMEnv
from storage import StorageManager

p = argparse.ArgumentParser()
p.add_argument("--episode", type=int, required=True); p.add_argument("--rep", type=int, required=True)
p.add_argument("--eval-seed", type=int, default=300); p.add_argument("--res", type=float, default=4.0)
p.add_argument("--out", required=True)
a = p.parse_args()
torch.set_default_device("cpu")
calls = []
def grab(self, name=None, dirname=None):
    if name and f"episode_{a.episode}_" in Path(name).name:
        calls.append((name, [f.copy() for f in self.frames]))
mk.vidManager.mk_video = grab
_o = Sim.TestPlotterOneEye.__init__
def zoomed(self, *args, **kw):
    kw["xlim"] = [-6, 28]; kw["ylim"] = [-6, 28]; _o(self, *args, **kw)
Sim.TestPlotterOneEye.__init__ = zoomed
from matplotlib.patches import Circle
_os = Sim.TestPlotterOneEye.onStep
def on_step(self):
    _os(self)
    for c in getattr(self, "_dots", []): c.remove()
    self._dots = []
    obs = getattr(self.env, "observation", None)
    if obs is None: return
    for i in np.where(np.asarray(obs["TOUCH_SENSORS"]) > 0)[0]:
        x, y = self.env.sim.bodies[f"s{i + 1}"].worldCenter
        c = Circle((x, y), 0.55, fc=(1, 0.1, 0.1), ec=(0.5, 0, 0), lw=0.8, zorder=10); self.ax.add_patch(c); self._dots.append(c)
Sim.TestPlotterOneEye.onStep = on_step
_e = SMEnv.__init__
def env_init(self, *args, **kw):
    if kw.get("render_params") is None: kw["render_params"] = {"resolution_prop": a.res, "duration": 100}
    _e(self, *args, **kw)
SMEnv.__init__ = env_init
runs = []
_r = Main.run_episodes
def run_episodes(self, *args, **kw):
    out = _r(self, *args, **kw); runs.append(out); return out
Main.run_episodes = run_episodes
main = np.load("main.dump.npy", allow_pickle=True)[0]
orig_dir = (Path.cwd() / ".." / ".." / ".." / "..").resolve()
exp_name = (Path.cwd() / ".." / "..").resolve().stem
main.sm = StorageManager(exp_name, orig_dir)
main.params.use_wandb = False
Path(a.out, "ev").mkdir(parents=True, exist_ok=True)
goal_counts, traj = main.evaluation_episodes(epoch=main.epoch, suffix="_0", render_path=str(Path(a.out) / "ev"), save_stats=False,
                                             n_episodes=main.params.tests, render="offline", e_seed=a.eval_seed + a.rep)
traj.to_csv(Path(a.out) / "traj_check.csv", index=False)
matches = runs[0][0]
fr = calls[0][1]
pickle.dump(dict(out=dict(matches=matches[a.episode:a.episode + 1], n_frames=[len(fr)]),
                 frames=[[np.array(f.convert("RGB")) for f in fr]]), open(Path(a.out) / "run.pkl", "wb"))
print("calls", [(Path(n).name, len(f)) for n, f in calls], "matches", np.where(matches[a.episode])[0].tolist())
