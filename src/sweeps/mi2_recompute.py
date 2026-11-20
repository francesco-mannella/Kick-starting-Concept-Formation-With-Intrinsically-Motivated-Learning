import argparse
import contextlib
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

sys.path.append(str(Path(__file__).resolve().parents[1]))
import mi_standard  # noqa: E402  (appended last: the sweep's own checkout provides SMMain)
from SMMain import Main  # noqa: E402

_ = Main
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--storage", required=True)
parser.add_argument("--epoch", type=int, required=True)
parser.add_argument("--series", choices=["agent", "par"], default="agent")
parser.add_argument("--out", required=True)
args = parser.parse_args()

main = np.load(Path(args.storage) / f"{args.epoch:06d}" / "main.dump.npy", allow_pickle=True)[0]
main.params.use_wandb = False
controller = main.controller_par if args.series == "par" else main.controller
captured = {}


def wrap(contexts, params_ind, controller_out, policy_ended):
    p, md = main.params, controller_out.model_data
    mask = np.ones(policy_ended.shape, bool)
    mask[:, : p.drop_first_n_steps + p.policy_selection_steps] = False
    a = md["batch_a"]
    uniq = a[policy_ended].reshape(-1, a.shape[-1])
    policies_ind = KMeans(10, random_state=0).fit(uniq).predict(a[mask].reshape(-1, a.shape[-1]))
    contexts = np.asarray(contexts)
    joints = md["batch_p"][mask]
    joints = joints.reshape(-1, joints.shape[-1])
    layout = np.repeat((np.asarray(params_ind) * 10 + contexts.astype(int))[:, None], mask.shape[1], axis=1)[mask]
    episode = np.repeat(np.arange(mask.shape[0])[:, None], mask.shape[1], axis=1)[mask]
    ctx = np.repeat(contexts[:, None], mask.shape[1], axis=1)[mask]
    captured.update(mi_standard.layout_excess(episode, policy_ended[mask], ctx, layout, policies_ind, md["batch_ss"][mask], joints))
    return {}


main.calc_mi_metrics = wrap
with contextlib.redirect_stdout(io.StringIO()):
    main.evaluation_episodes(orig_controller=controller, epoch=0, n_episodes=main.params.evaluation_episodes, render=None, save_stats=False)
suffix = "_par" if args.series == "par" else ""
json.dump({f"{k}{suffix}": float(v) for k, v in captured.items()}, open(args.out, "w"))
