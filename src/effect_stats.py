#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from policy_stats import eta2, load, node_effects, segments, unit_null  # noqa: E402
from SMMain import Main  # noqa: E402  (pickled dumps look up Main in __main__)

_ = Main


def parse_arguments():
    parser = argparse.ArgumentParser(description="Effect-space differentiation: eta2 of the open-loop arm configuration of the executed nodes explained by context")
    parser.add_argument("--storage", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--controller", default="controller")
    parser.add_argument("--n-perm", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def segment_end_postures(data, episodes):
    actions, postures, touch = data["a"], data["p"], np.linalg.norm(data["ss"], axis=-1) > 1e-6
    rows = []
    for ep in range(actions.shape[0]):
        starts, last = [], None
        for t in range(actions.shape[1]):
            v = actions[ep, t]
            if not v.any() or (last is not None and np.allclose(v, last)):
                continue
            starts.append(t)
            last = v
        active = np.where(actions[ep].any(-1))[0]
        if not len(starts):
            continue
        ends = [s - 1 for s in starts[1:]] + [active[-1]]
        for s, e in zip(starts, ends):
            contact = touch[ep, s : e + 1].any()
            rows.append((ep, episodes.context[ep % len(episodes)], contact, np.abs(postures[ep, e])))
    return rows


if __name__ == "__main__":
    args = parse_arguments()
    rng = np.random.default_rng(args.seed)
    main, data = load(args.storage, args.epoch, "data.npy")
    _, label = segments(data, main.episode_dataset)
    effects = node_effects(main, args.controller)[label.node.values]
    units = (str(args.epoch) + "_" + label.ep.astype(str)).values
    contexts = label.ctx.values
    null, p95 = unit_null(effects, units, contexts, rng, args.n_perm)
    out = {"segments": len(label), "eta2_context_effect": float(eta2(effects, contexts)), "eta2_context_effect_null": null, "eta2_context_effect_p95": p95}
    out["excess_over_p95"] = out["eta2_context_effect"] - p95
    rows = segment_end_postures(data, main.episode_dataset)
    post = np.array([r[3] for r in rows])
    ctx = np.array([r[1] for r in rows])
    contact = np.array([r[2] for r in rows])
    unit = np.array([str(args.epoch) + "_" + str(r[0]) for r in rows])
    for name, sel in (("all", np.ones(len(rows), bool)), ("contact", contact)):
        if sel.sum() < 12 or len(np.unique(ctx[sel])) < 3:
            continue
        x, c, u = post[sel], ctx[sel], unit[sel]
        n, p = unit_null(x, u, c, rng, args.n_perm)
        out[f"eta2_closedloop_{name}"] = float(eta2(x, c))
        out[f"eta2_closedloop_{name}_null"] = n
        out[f"eta2_closedloop_{name}_p95"] = p
        out[f"closedloop_{name}_excess"] = out[f"eta2_closedloop_{name}"] - p
        out[f"closedloop_{name}_segments"] = int(sel.sum())
    Path(args.out).write_text(json.dumps(out))
    print(json.dumps(out))
