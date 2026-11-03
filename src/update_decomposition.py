import numpy as np


def common_share(rows):
    return float((rows.mean(0) ** 2).sum() / (rows ** 2).sum(1).mean())


def decompose_update(phi, goals, policies, side=10, spread_frac=0.1):
    n_events, n_nodes = phi.shape
    goal_node = goals.argmax(1)
    xy = np.stack((np.arange(n_nodes) // side, np.arange(n_nodes) % side), 1)
    dist = np.linalg.norm(xy[None] - xy[goal_node][:, None], axis=-1)
    total = phi.sum()
    out = {
        "upd_events": float(n_events),
        "upd_goal_nodes": float(len(np.unique(goal_node))),
        "upd_own_share": float(phi[np.arange(n_events), goal_node].sum() / total),
        "upd_weighted_distance": float((phi * dist).sum() / total),
        "upd_spread_nodes": float((phi >= spread_frac * phi.max(1, keepdims=True)).sum(1).mean()),
    }
    means = [policies[goal_node == g].mean(0) for g in np.unique(goal_node) if (goal_node == g).sum() >= 2]
    out["upd_event_common_share"] = common_share(np.stack(means)) if len(means) >= 2 else float("nan")
    out["upd_event_common_raw"] = common_share(policies) if n_events >= 2 else float("nan")
    return out
