"""Pick the KD checkpoint on val500 (plan D10-2; selection set decided 2026-10-25).

val500 = eval_sets/indist_500 (official val, distribution-matched like test_2000), run_baseline
protocol (k 8, seed 42, max_gen 256). Being the selection set, val500 is no longer reported;
the report uses test_2000 and OOD-Val262. Epoch 0 is the zero-shot MT-W model (its stored
val500 rows), epochs 1..E the train_kd checkpoints. Rule: the epoch checkpoint with the lowest
mean minADE@6; epoch 0 is a reference only, never selected.
Writes outputs/minitron/<exp>_select/{summary.json, summary.txt, selected.txt}.

Usage:
  .venv/bin/python minitron/code/select_ckpt.py --exp mtw24_kd --zs-arm mtw24_zs
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from compare_openloop import boot_ci, rows

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="mtw24_kd")
    ap.add_argument("--zs-arm", default="mtw24_zs")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    arms = {0: rows(f"minitron/{args.zs_arm}_indist")}
    arms.update({e: rows(f"minitron/{args.exp}_val_e{e}") for e in range(1, args.epochs + 1)})
    ids = sorted(set.intersection(*(set(r) for r in arms.values())))
    assert len(ids) == 500, f"expected the 500 val500 clips in every run, got {len(ids)}"
    at6 = {e: np.array([min(r[c]["ade_rollout_k"][:6]) for c in ids]) for e, r in arms.items()}
    fde6 = {e: np.array([min(r[c]["fde_rollout_k"][:6]) for c in ids]) for e, r in arms.items()}
    rec = {}
    for e in arms:
        d = at6[e] - at6[0]
        lo, hi = boot_ci(d) if e else (0.0, 0.0)
        rec[e] = {"minADE6_mean": float(at6[e].mean()), "minADE6_median": float(np.median(at6[e])),
                  "minFDE6_mean": float(fde6[e].mean()),
                  "coc_degenerate": float(np.mean([bool(arms[e][c].get("coc_degenerate", False))
                                                   for c in ids])),
                  "vs_e0_mean": float(d.mean()), "vs_e0_ci95": [float(lo), float(hi)]}
    best = min(range(1, args.epochs + 1), key=lambda e: rec[e]["minADE6_mean"])
    sel = f"outputs/minitron/{args.exp}/ckpt_epoch{best}"
    out = REPO / "outputs" / "minitron" / f"{args.exp}_select"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(
        {"selection_set": "val500 (eval_sets/indist_500)", "n_clips": len(ids),
         "rule": "lowest mean minADE@6, epochs >= 1", "selected_epoch": best, "selected": sel,
         "epochs": rec}, indent=2))
    lines = [f"selection on val500, n={len(ids)}; rule: lowest mean minADE@6 among epochs >= 1",
             "epoch  minADE@6 mean  median  minFDE@6  CoC degen  vs e0 [95% CI]"]
    for e, r in rec.items():
        lines.append(f"{e:5d}  {r['minADE6_mean']:13.4f}  {r['minADE6_median']:6.4f}  "
                     f"{r['minFDE6_mean']:8.3f}  {r['coc_degenerate']:9.1%}  "
                     f"{r['vs_e0_mean']:+.4f} [{r['vs_e0_ci95'][0]:+.4f}, {r['vs_e0_ci95'][1]:+.4f}]")
    lines.append(f"selected: epoch {best} -> {sel}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "selected.txt").write_text(sel + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
