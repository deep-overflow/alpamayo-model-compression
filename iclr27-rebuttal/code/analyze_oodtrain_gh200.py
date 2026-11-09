"""GH200 OOD-train open loop + reference-CoC NLL for the ICLR27 arms.

Plan: iclr27-rebuttal/plans/2026-11-09_oodtrain-iclr27-arms-gh200.md. Rows come from
iclr27-rebuttal/code/pull_oodtrain_gh200.sh (outputs/iclr27-rebuttal/gh200/).

  --stage gate   G1: GH200 dense shard 0 vs A100 dense on the same clips: |mean delta 6.4 s minADE@6| <= 0.02 m,
                 CoC degeneration side by side, GPU names and model revision from the configs
  --stage table  each arm: minADE@6 / minFDE@6 at 1.6 / 3.2 / 6.4 s (mean, median), CoC degeneration, reference-CoC
                 NLL; paired deltas vs dense and vs MoOP (6.4 s minADE@6 / minFDE@6, NLL; 200,000 bootstrap draws);
                 per-cluster 6.4 s minADE@6; dense and MoOP against their A100 OOD-train rows (a sanity check only,
                 the two GPU types are never merged into one table)

--root runs the table on another site's rows; `--root outputs/cvpr27/a100` reproduces the two ICLR27 arms that A100
has (dense 0.922, MoOP 1.017 at 6.4 s, cvpr27/reports/2026-11-03_oodtrain-coc-quality.md).
Usage (repo root): .venv/bin/python iclr27-rebuttal/code/analyze_oodtrain_gh200.py --stage table
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "cvpr27" / "code"))
from analyze_a100 import HORIZONS, boot_ci, load_rows, metric

ARMS = [("dense", "baseline", "Dense"), ("moop", "slim_dual_u40_v2", "MoOP"),
        ("moop_em93p75", "slim_dualexp_u40_em93p75", "MoOP + expert"), ("fm_only", "slim_traj_u40_v2", "FM-only"),
        ("ce_only", "slim_coc_u40_v2", "CE-only"), ("wanda", "slim_wanda_u40_v2", "Wanda"),
        ("lp_r50", "slim_lp_r50", "LLM-Pruner"), ("tyr_u40_r", "slim_tyr_u40_r", "Tyr")]
TAG = {k: t for k, t, _ in ARMS}
LABEL = {k: lab for k, _, lab in ARMS}
A100 = REPO / "outputs" / "cvpr27" / "a100"
G1_TOL = 0.02


def ids_and_clusters():
    man = pd.read_parquet(REPO / "outputs" / "eval_sets" / "ood_train.parquet")
    return list(man.clip_id), dict(zip(man.clip_id, man.cluster))


def nll_rows(root, arm):
    d = root / f"nll_{arm}_oodtrain"
    return load_rows(d, TAG[arm]) if d.exists() else {}


def paired(r, base, ids, nll_r=None, nll_b=None):
    """arm - base on the clips both have: 6.4 s minADE@6 / minFDE@6 and reference-CoC NLL (mean, CI, median)."""
    both = [c for c in ids if c in r and c in base]
    out = {"n": len(both)}
    for kind in ("ade", "fde"):
        d = np.array([metric(r[c], kind) - metric(base[c], kind) for c in both])  # (N,)
        out[f"min{kind.upper()}6"] = {"mean": float(d.mean()), "ci": boot_ci(d), "median": float(np.median(d))}
    if nll_r and nll_b:
        bn = [c for c in ids if c in nll_r and c in nll_b]
        d = np.array([nll_r[c]["nll_gtcoc"] - nll_b[c]["nll_gtcoc"] for c in bn])  # (N,)
        out["nll"] = {"n": len(bn), "mean": float(d.mean()), "ci": boot_ci(d), "median": float(np.median(d))}
    return out


def fmt(p, d=3):
    star = "*" if p["ci"][0] > 0 or p["ci"][1] < 0 else ""
    return f"{p['mean']:+.{d}f} [{p['ci'][0]:+.{d}f}, {p['ci'][1]:+.{d}f}]{star}"


def gate(root):
    gh = load_rows(root / "dense_oodtrain", "baseline")
    s0 = root / "dense_oodtrain" / "baseline_s0of4.json"
    if not s0.exists():
        print(f"G1: {s0} not there yet")
        return 2
    clips = [r["clip_id"] for r in json.loads(s0.read_text())]
    a = load_rows(A100 / "dense_oodtrain", "baseline")
    both = [c for c in clips if c in a]
    d = np.array([metric(gh[c], "ade") - metric(a[c], "ade") for c in both])  # (N,)
    deg_g = np.mean([gh[c]["coc_degenerate"] for c in both])
    deg_a = np.mean([a[c]["coc_degenerate"] for c in both])
    same = np.mean([gh[c]["gen_coc"] == a[c]["gen_coc"] for c in both])
    cfg = json.loads((root / "dense_oodtrain" / "config.json").read_text())
    ok = abs(d.mean()) <= G1_TOL
    print(f"G1 dense shard 0, {len(both)} clips: GH200 - A100 6.4 s minADE@6 mean {d.mean():+.4f} m"
          f" (median {np.median(d):+.4f}), |mean| <= {G1_TOL} -> {'PASS' if ok else 'FAIL'}")
    print(f"   CoC degeneration GH200 {deg_g:.1%} vs A100 {deg_a:.1%}; same CoC text on {same:.1%} of clips")
    print(f"   GH200 config: gpu {cfg.get('gpu')}, model revision {cfg.get('model_revision', '')[:7]}")
    return 0 if ok else 1


def table(root):
    ids, cluster = ids_and_clusters()
    rows = {k: load_rows(root / f"{k}_oodtrain", TAG[k]) for k, _, _ in ARMS if (root / f"{k}_oodtrain").exists()}
    nll = {k: nll_rows(root, k) for k in rows}
    res, lines = {"root": str(root), "arms": {}, "paired": {}}, []
    lines.append(f"# OOD-train open loop, {res['root']} (manifest ood_train, {len(ids)} clips)")
    lines.append("\n| arm | clips | minADE@6 1.6 / 3.2 / 6.4 s | minFDE@6 1.6 / 3.2 / 6.4 s | median minADE@6 6.4 s"
                 " | CoC degeneration | reference-CoC NLL |")
    lines.append("|---|---|---|---|---|---|---|")
    for k, r in rows.items():
        cs = [c for c in ids if c in r]
        e = {"n": len(cs), "coc_degenerate": float(np.mean([r[c]["coc_degenerate"] for c in cs]))}
        for h in HORIZONS:
            for kind in ("ade", "fde"):
                v = np.array([metric(r[c], kind, h) for c in cs])  # (N,)
                e[f"min{kind.upper()}6_{h}"], e[f"min{kind.upper()}6_{h}_median"] = float(v.mean()), float(np.median(v))
        nc = [c for c in ids if c in nll[k]]
        e["nll_n"] = len(nc)
        e["nll"] = float(np.mean([nll[k][c]["nll_gtcoc"] for c in nc])) if nc else None
        res["arms"][k] = e
        lines.append(f"| {LABEL[k]} | {e['n']} | " + " / ".join(f"{e[f'minADE6_{h}']:.3f}" for h in HORIZONS) + " | "
                     + " / ".join(f"{e[f'minFDE6_{h}']:.3f}" for h in HORIZONS)
                     + f" | {e['minADE6_6.4s_median']:.3f} | {e['coc_degenerate']:.1%} | "
                     + (f"{e['nll']:.3f}" if e["nll"] is not None else "—") + " |")
    for ref in ("dense", "moop"):
        if ref not in rows:
            continue
        lines.append(f"\n| arm − {LABEL[ref]} | clips | Δ minADE@6 6.4 s [95% CI] | median | Δ minFDE@6 6.4 s [95% CI]"
                     " | Δ reference-CoC NLL [95% CI] |")
        lines.append("|---|---|---|---|---|---|")
        for k, r in rows.items():
            if k == ref:
                continue
            p = res["paired"][f"{k}-{ref}"] = paired(r, rows[ref], ids, nll[k], nll[ref])
            lines.append(f"| {LABEL[k]} | {p['n']} | {fmt(p['minADE6'])} | {p['minADE6']['median']:+.3f} | "
                         f"{fmt(p['minFDE6'])} | " + (fmt(p["nll"]) if "nll" in p else "—") + " |")
    groups = sorted(set(cluster.values()))
    lines.append("\n6.4 s minADE@6 by OOD cluster (clips in parentheses):\n")
    lines.append("| arm | " + " | ".join(f"{g} ({sum(1 for c in ids if cluster[c] == g)})" for g in groups) + " |")
    lines.append("|---|" + "---|" * len(groups))
    res["clusters"] = {}
    for k, r in rows.items():
        cm = {g: float(np.mean([metric(r[c], "ade") for c in ids if c in r and cluster[c] == g])) for g in groups}
        res["clusters"][k] = cm
        lines.append(f"| {LABEL[k]} | " + " | ".join(f"{cm[g]:.3f}" for g in groups) + " |")
    if root.resolve() != A100.resolve():
        lines.append("\nCross-site sanity check (GH200 − A100, same clips; never merged):")
        res["cross_site"] = {}
        for k in ("dense", "moop"):
            if k in rows and (A100 / f"{k}_oodtrain").exists():
                a = load_rows(A100 / f"{k}_oodtrain", TAG[k])
                p = res["cross_site"][k] = paired(rows[k], a, ids, nll[k], nll_rows(A100, k))
                same = np.mean([rows[k][c]["gen_coc"] == a[c]["gen_coc"] for c in ids if c in rows[k] and c in a])
                lines.append(f"  {LABEL[k]}: {p['n']} clips, Δ minADE@6 6.4 s {fmt(p['minADE6'], 4)}, same CoC text "
                             f"{same:.1%}" + (f", Δ NLL {fmt(p['nll'], 4)}" if "nll" in p else ""))
    dest = root / "analysis"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(res, indent=1))
    (dest / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {dest}/metrics.json, table.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gate", "table"], default="table")
    ap.add_argument("--root", default="outputs/iclr27-rebuttal/gh200")
    args = ap.parse_args()
    root = REPO / args.root
    sys.exit(gate(root) if args.stage == "gate" else table(root))


if __name__ == "__main__":
    main()
