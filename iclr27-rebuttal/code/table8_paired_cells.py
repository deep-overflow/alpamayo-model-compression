"""Paper Table 8 (paired differences from MoOP in closed-loop driving, 95% CI): every cell, Origin150 and Hard100.

Reuses cvpr27/code/closedloop_ci.py as a module -- the same runs, loader (Hard100 = paper convention, 100 scenes,
aborted rollout = score 0 and no event), per-scene means, scene-cluster bootstrap and RNG order as its main(), so the
cells already in the table come out identical. Sign as in the caption: scene score = MoOP - method; offroad and
at-fault rates = method - MoOP (positive favours MoOP). Pooled ("Combined") = set-stratified, as in closedloop_ci.py.

  .venv/bin/python iclr27-rebuttal/code/table8_paired_cells.py [--seed 0] [--draws 100000]
writes outputs/iclr27-rebuttal/table8_paired/{cells_seed<k>.json, table_seed<k>.md}
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "cvpr27" / "code"))
import closedloop_ci as cc  # noqa: E402

ROWS = [("Baseline", "baseline"), ("Wanda", "wanda"), ("LLM-Pruner", "lp_r50"), ("Týr-the-Pruner", "tyr"),
        ("FM-only", "traj"), ("CE-only", "coc"), ("MoOP + expert", "dualexp")]
OUT = REPO / "outputs" / "iclr27-rebuttal" / "table8_paired"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=100_000)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    per = {}
    for set_ in cc.SETS:  # same order of RNG draws as closedloop_ci.main()
        data = {k: cc.load(set_, c, "paper") for k, c in cc.ARMS}
        scenes = sorted({x["scene"] for x in data["baseline"]})
        si = {s: i for i, s in enumerate(scenes)}
        for k, v in data.items():
            assert sorted({x["scene"] for x in v}) == scenes and len(v) == 2 * len(scenes), (set_, k)
            for m in cc.METRICS:
                acc, cnt = np.zeros(len(scenes)), np.zeros(len(scenes))
                for x in v:
                    acc[si[x["scene"]]] += x[m]
                    cnt[si[x["scene"]]] += 1
                per[set_, k, m] = acc / cnt  # (S,)
        rng.integers(0, len(scenes), (args.draws, len(scenes)))  # the per-arm resample main() draws here
    n = {s: len(per[s, "baseline", "score"]) for s in cc.SETS}
    idx = {s: rng.integers(0, n[s], (args.draws, n[s])) for s in cc.SETS}  # the shared paired resample

    cells = {}
    for label, k in ROWS:
        for m in ("score", "offroad", "at_fault"):
            sign = 1.0 if m == "score" else -1.0  # score: MoOP - method; rates: method - MoOP
            d = {s: sign * (per[s, "dual", m] - per[s, k, m]) for s in cc.SETS}
            for s in cc.SETS:
                b = d[s][idx[s]].mean(1)  # (B,)
                lo, hi = np.percentile(b, [2.5, 97.5])
                cells[label, s, m] = (float(d[s].mean()), float(lo), float(hi))
            bp = sum(d[s][idx[s]].sum(1) for s in cc.SETS) / sum(n.values())
            lo, hi = np.percentile(bp, [2.5, 97.5])
            cells[label, "pooled", m] = (float(np.concatenate([d[s] for s in cc.SETS]).mean()), float(lo), float(hi))

    f = lambda c: f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]{'*' if c[1] > 0 or c[2] < 0 else ''}"  # noqa: E731
    lines = [f"seed {args.seed}, {args.draws:,} draws; * = CI excludes 0",
             "| Method | O150 score | O150 offroad | O150 at-fault | H100 score | H100 offroad | H100 at-fault | Combined score |",
             "|---|---|---|---|---|---|---|---|"]
    for label, _ in ROWS:
        lines.append(f"| {label} | " + " | ".join(f(cells[label, s, m]) for s, m in (
            ("origin150", "score"), ("origin150", "offroad"), ("origin150", "at_fault"), ("hard100", "score"),
            ("hard100", "offroad"), ("hard100", "at_fault"), ("pooled", "score"))) + " |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"cells_seed{args.seed}.json").write_text(json.dumps(
        {"seed": args.seed, "draws": args.draws, "cells": {"|".join(k): v for k, v in cells.items()}}, indent=1))
    (OUT / f"table_seed{args.seed}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
