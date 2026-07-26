"""Evaluation utilities: multi-sample minADE/minFDE and geometry-based buckets.

The released data provides only the GT trajectory, so scenario buckets are derived
from its geometry rather than from labels. This is the honest proxy for "where do
the worst cases hide": a config that is free on straight cruising but breaks on
turns or stops is exactly the failure mode a mean over clips would hide.
"""

import numpy as np

DT = 6.4 / 64  # seconds per future step


def gt_geometry(gt_xy):
    """Speed profile and net heading change from a GT xy path (T, 2)."""
    vel = np.diff(gt_xy, axis=0) / DT  # (T-1, 2)
    speed = np.linalg.norm(vel, axis=1)  # (T-1,)
    moving = speed > 0.5  # m/s, below this heading is undefined
    if moving.sum() >= 2:
        head = np.arctan2(vel[moving, 1], vel[moving, 0])
        net_turn = np.abs(np.rad2deg(np.angle(np.exp(1j * (head[-1] - head[0])))))
    else:
        net_turn = 0.0
    k = max(len(speed) // 8, 1)
    v0 = float(speed[:k].mean())
    v_end = float(speed[-k:].mean())
    return {"v0": v0, "v_end": v_end, "v_mean": float(speed.mean()),
            "net_turn_deg": float(net_turn)}


def bucket(gt_xy):
    """Assign a GT path to one scenario bucket (priority: stop > turn > accel > cruise)."""
    g = gt_geometry(gt_xy)
    if g["v_end"] < 0.5 or g["v_end"] < 0.5 * g["v0"]:
        return "decel_stop"
    if g["net_turn_deg"] >= 30.0:
        return "turn"
    if g["v_end"] > 1.5 * max(g["v0"], 0.5):
        return "accel"
    return "cruise"


def ade_fde(pred_xy, gt_xy):
    """Per-sample ADE (mean L2 over steps) and FDE (endpoint L2). pred_xy (K, T, 2)."""
    err = np.linalg.norm(pred_xy - gt_xy[None], axis=2)  # (K, T)
    ade = err.mean(1)  # (K,)
    fde = err[:, -1]  # (K,)
    return ade, fde


def min_metrics(pred_xy, gt_xy):
    """Multi-sample minADE / minFDE from K predicted paths (K, T, 2)."""
    ade, fde = ade_fde(pred_xy, gt_xy)
    return float(ade.min()), float(fde.min())


def paired_bootstrap_ci(delta, n_boot=10000, seed=0, alpha=0.05):
    """Bootstrap CI for the mean of a paired difference vector."""
    d = np.asarray(delta, dtype=float)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi)
