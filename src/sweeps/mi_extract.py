import argparse
import contextlib
import io
import json
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import policy_stats as ps
from SMMain import Main

_ = Main
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--storage", required=True)
parser.add_argument("--epoch", type=int, default=1499)
parser.add_argument("--eseed", type=int, default=7)
parser.add_argument("--out", required=True)
args = parser.parse_args()

main, _d = ps.load(args.storage, args.epoch, "data.npy")
captured = {}
orig = main.calc_mi_metrics


def wrap(contexts, params_ind, controller_out, policy_ended):
    res = orig(contexts, params_ind, controller_out, policy_ended)
    p, md = main.params, controller_out.model_data
    mask = np.ones(policy_ended.shape, bool)
    mask[:, : p.drop_first_n_steps + p.policy_selection_steps] = False
    a = md["batch_a"]
    uniq = a[policy_ended].reshape(-1, a.shape[-1])
    km = KMeans(10, random_state=0).fit(uniq)
    act = a[mask].reshape(-1, a.shape[-1])
    pca = PCA(5).fit(uniq)
    ep = np.repeat(np.arange(mask.shape[0])[:, None], mask.shape[1], 1)[mask]
    gp = md["g_p"][mask]
    ss = md["batch_ss"][mask]
    jt = md["batch_p"][mask].reshape(len(ep), -1)
    tc = ss.argmax(-1) // 10 + 1
    tc[ss.sum(-1) == 0] = 0
    gk = KMeans(10, random_state=0).fit(jt)
    captured.update(
        ep=ep, ctx=np.repeat(np.asarray(contexts)[:, None], mask.shape[1], 1)[mask], ended=policy_ended[mask],
        exec_policy=km.predict(act), node=(np.round(gp[:, 0]) * 10 + np.round(gp[:, 1])).astype(int), touch=tc, grip=gk.predict(jt),
        pol_pc=pca.transform(act).astype(np.float32), joints=jt.astype(np.float32),
        touch_groups=ss.reshape(len(ep), 4, 10).sum(-1).astype(np.float32), metrics=res)
    return res


main.calc_mi_metrics = wrap
with contextlib.redirect_stdout(io.StringIO()):
    main.evaluation_episodes(orig_controller=main.controller, epoch=0, n_episodes=main.params.evaluation_episodes, render=None, save_stats=False, e_seed=args.eseed)
m = captured.pop("metrics")
np.savez_compressed(args.out + ".npz", **captured)
json.dump({k: float(v) for k, v in m.items()}, open(args.out + ".json", "w"))
print("saved", args.out, len(captured["ep"]), "steps", len(np.unique(captured["ep"])), "episodes")
