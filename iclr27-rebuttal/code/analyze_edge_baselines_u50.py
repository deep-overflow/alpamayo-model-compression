"""Edge u50: the ICLR paper baselines (Tyr, LLM-Pruner, Wanda) next to dense / MoOP / FM-only / CE-only
(plans/2026-11-15_a2-edge-baselines-u50.md).

Per set (Test500, OOD-Val250, OOD-train1178): minADE@6 / minFDE@6 (analyze_a100.metric) at 1.6 / 3.2 / 6.4 s, mean and median;
CoC readouts; paired arm - dense and MoOP - arm per clip with ci_openloop_lingoqa.summarise (200,000 draws, seed 0 / 1).
Arms that are not finished on a set are skipped (--partial allows incomplete arms on their finished clips only).

Usage: .venv/bin/python iclr27-rebuttal/code/analyze_edge_baselines_u50.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO / "cvpr27" / "code"))
from analyze_a100 import HORIZONS, load_rows, metric
from ci_openloop_lingoqa import summarise

OUT = REPO / "outputs" / "iclr27-rebuttal"
SETS = {"test500": REPO / "outputs" / "eval_sets" / "test_500.parquet", "oodval": OUT / "edge_moop" / "ood_val_250.parquet",
        "oodtrain": OUT / "edge_moop" / "ood_train_1178.parquet"}
ARMS = {"dense": "dense", "moop": "moop_u50", "moop_e": "moop_em75_u50", "fm_only": "fm_only_u50", "ce_only": "ce_only_u50",
        "tyr": "tyr_u50", "lp": "lp_u50", "wanda": "wanda_u50"}
LABEL = {"dense": "Dense", "moop": "MoOP", "moop_e": "MoOP+E75", "fm_only": "FM-only", "ce_only": "CE-only", "tyr": "Tyr",
         "lp": "LLM-Pruner", "wanda": "Wanda"}
BASE = ("fm_only", "ce_only", "tyr", "lp", "wanda")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="iclr27-rebuttal/edge_baselines_u50_analysis")
    args = ap.parse_args()
    out = REPO / "outputs" / args.exp_id
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for s, p in SETS.items():
        ids = list(pd.read_parquet(p)["clip_id"])
        rows = {a: load_rows(OUT / f"edge_{s}_{d}", "edge") for a, d in ARMS.items()}
        arms = [a for a in ARMS if all(c in rows[a] for c in ids)]
        absolute, paired = {}, {}
        for k in ("ade", "fde"):
            for h in HORIZONS:
                for a in arms:
                    absolute[f"{a}|{k}|{h}"] = [metric(rows[a][c], k, h) for c in ids]
                for a in arms:
                    if a != "dense" and "dense" in arms:
                        paired[f"{a}|dense|{k}|{h}"] = list(np.subtract(absolute[f"{a}|{k}|{h}"], absolute[f"dense|{k}|{h}"]))
                    if a in BASE and "moop" in arms:
                        paired[f"moop|{a}|{k}|{h}"] = list(np.subtract(absolute[f"moop|{k}|{h}"], absolute[f"{a}|{k}|{h}"]))
                    if a in BASE + ("moop",) and "moop_e" in arms:  # MoOP + expert MLP 75 %
                        paired[f"moop_e|{a}|{k}|{h}"] = list(np.subtract(absolute[f"moop_e|{k}|{h}"],
                                                                         absolute[f"{a}|{k}|{h}"]))
        coc = {a: {"degenerate": float(np.mean([rows[a][c]["coc_degenerate"] for c in ids])),
                   "nominal": float(np.mean([rows[a][c]["nominal"] for c in ids])),
                   "len": float(np.mean([rows[a][c]["gen_len"] for c in ids])),
                   "no_anchor": int(sum(not rows[a][c]["has_eos"] for c in ids))} for a in arms}
        res[s] = {"n": len(ids), "arms": arms, "absolute": summarise(absolute), "paired": summarise(paired), "coc": coc}
        print(f"  {s}: {arms}", flush=True)
    L = ["Edge u50: dense / MoOP / MoOP+E75 / FM-only / CE-only / Tyr / LLM-Pruner / Wanda -- minADE@6 / minFDE@6 (m); Ada", ""]
    for s, r in res.items():
        L += [f"== {s} ({r['n']} clips)",
              f"{'arm':<11}" + "".join(f"{f'{k.upper()} {h}':>17}" for k in ("ade", "fde") for h in HORIZONS)
              + f"{'degen':>8}{'len':>6}{'no-eos':>7}"]
        for a in r["arms"]:
            ab = r["absolute"]
            L.append(f"{LABEL[a]:<11}" + "".join(f"{ab[f'{a}|{k}|{h}']['mean']:>9.3f} ({ab[f'{a}|{k}|{h}']['median']:.3f})"
                                              for k in ("ade", "fde") for h in HORIZONS)
                     + f"{r['coc'][a]['degenerate']:>8.3f}{r['coc'][a]['len']:>6.1f}{r['coc'][a]['no_anchor']:>7d}")
        L.append("paired (mean [95% CI], * = CI excludes 0):")
        for key, e in r["paired"].items():
            if key.endswith(("|ade|6.4s", "|fde|6.4s", "|ade|1.6s", "|ade|3.2s")):
                st = "*" if e["ci"][0] > 0 or e["ci"][1] < 0 else ""
                L.append(f"  {key:<30} {e['mean']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]{st} p {e['p']:.3g}")
        L.append("")
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    (out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
