"""Pick the KD checkpoint on val500 (plan D10-2; selection set decided 2026-10-25).

val500 = eval_sets/indist_500 (official val, distribution-matched like test_2000), run_baseline
protocol (k 8, seed 42, max_gen 256). Being the selection set, val500 is no longer reported;
the report uses test_2000 and OOD-Val262. "zs" is the zero-shot MT-W model (its stored val500
rows), the others the train_kd checkpoints named by --ckpts (ckpt_epochN rows live in
<exp>_val_eN, ckpt_qNN rows in <exp>_val_qNN). Rule: the checkpoint with the lowest mean
minADE@6; zs is a reference only, never selected.
Writes outputs/minitron/<exp>_select/{summary.json, summary.txt, selected.txt}.

Usage:
  .venv/bin/python minitron/code/select_ckpt.py --exp mtw24_kd --zs-arm mtw24_zs
  .venv/bin/python minitron/code/select_ckpt.py --exp mtw24_kd2 \\
      --ckpts ckpt_q25 ckpt_q50 ckpt_q75 ckpt_q100
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
    ap.add_argument("--ckpts", nargs="+", default=["ckpt_epoch1", "ckpt_epoch2", "ckpt_epoch3"])
    args = ap.parse_args()
    tag = {c: c.removeprefix("ckpt_").replace("epoch", "e") for c in args.ckpts}
    arms = {"zs": rows(f"minitron/{args.zs_arm}_indist")}
    arms.update({c: rows(f"minitron/{args.exp}_val_{tag[c]}") for c in args.ckpts})
    ids = sorted(set.intersection(*(set(r) for r in arms.values())))
    assert len(ids) == 500, f"expected the 500 val500 clips in every run, got {len(ids)}"
    at6 = {c: np.array([min(r[i]["ade_rollout_k"][:6]) for i in ids]) for c, r in arms.items()}
    fde6 = {c: np.array([min(r[i]["fde_rollout_k"][:6]) for i in ids]) for c, r in arms.items()}
    rec = {}
    for c in arms:
        d = at6[c] - at6["zs"]
        lo, hi = boot_ci(d) if c != "zs" else (0.0, 0.0)
        rec[c] = {"minADE6_mean": float(at6[c].mean()), "minADE6_median": float(np.median(at6[c])),
                  "minFDE6_mean": float(fde6[c].mean()),
                  "coc_degenerate": float(np.mean([bool(arms[c][i].get("coc_degenerate", False))
                                                   for i in ids])),
                  "vs_zs_mean": float(d.mean()), "vs_zs_ci95": [float(lo), float(hi)]}
    best = min(args.ckpts, key=lambda c: rec[c]["minADE6_mean"])
    sel = f"outputs/minitron/{args.exp}/{best}"
    out = REPO / "outputs" / "minitron" / f"{args.exp}_select"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(
        {"selection_set": "val500 (eval_sets/indist_500)", "n_clips": len(ids),
         "rule": "lowest mean minADE@6 among the checkpoints (zs is a reference)",
         "selected_ckpt": best, "selected": sel, "ckpts": rec}, indent=2))
    lines = [f"selection on val500, n={len(ids)}; rule: lowest mean minADE@6 among the checkpoints",
             "checkpoint    minADE@6 mean  median  minFDE@6  CoC degen  vs zs [95% CI]"]
    for c, r in rec.items():
        lines.append(f"{c:12s}  {r['minADE6_mean']:13.4f}  {r['minADE6_median']:6.4f}  "
                     f"{r['minFDE6_mean']:8.3f}  {r['coc_degenerate']:9.1%}  "
                     f"{r['vs_zs_mean']:+.4f} [{r['vs_zs_ci95'][0]:+.4f}, {r['vs_zs_ci95'][1]:+.4f}]")
    lines.append(f"selected: {best} -> {sel}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "selected.txt").write_text(sel + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
