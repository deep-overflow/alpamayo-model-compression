"""Evaluation utilities: multi-sample minADE/minFDE and geometry-based buckets.

The released data provides only the GT trajectory, so scenario buckets are derived
from its geometry rather than from labels. This is the honest proxy for "where do
the worst cases hide": a config that is free on straight cruising but breaks on
turns or stops is exactly the failure mode a mean over clips would hide.
"""

from itertools import groupby

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


def coc_degenerate(text):
    """Degeneracy heuristics for one generated CoC string.

    `degenerate` uses exactly the analyze_alpasim.coc_stats thresholds so open-loop rates
    measured here stay comparable to the closed-loop degeneracy rates already reported.

    Those thresholds miss one observed failure mode: a filler run like "//------------"
    counts as a single long word, so it keeps unique_ratio high and non-ascii low.
    `degenerate_strict` adds that case (a >=10-character run of one character) and is
    reported alongside rather than folded in, so both scales stay interpretable.
    """
    s = str(text).strip()
    words = s.split()
    uniq = len(set(words)) / max(len(words), 1)
    nonascii = sum(ord(ch) > 127 for ch in s) / max(len(s), 1)
    empty = len(s) == 0
    soup = (not empty) and (nonascii > 0.05 or uniq < 0.5 or len(s) > 300)
    run = max((len(list(g)) for _, g in groupby(s.replace(" ", ""))), default=0)
    return {"empty": bool(empty), "soup": bool(soup), "degenerate": bool(empty or soup),
            "degenerate_strict": bool(empty or soup or run >= 10),
            "len": len(s), "unique_ratio": float(uniq), "nonascii": float(nonascii),
            "char_run": int(run)}


def paired_bootstrap_ci(delta, n_boot=10000, seed=0, alpha=0.05):
    """Bootstrap CI for the mean of a paired difference vector."""
    d = np.asarray(delta, dtype=float)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi)
