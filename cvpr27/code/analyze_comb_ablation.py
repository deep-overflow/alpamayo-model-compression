"""Combination-rule ablation table (cvpr27/plans/2026-11-04_combination-rule-ablation.md), A100 rows.

Absolute minADE@6 / minFDE@6 at 1.6 / 3.2 / 6.4 s, CoC-degenerate share and LingoQA for MoOP l0keep, FM-only,
CE-only and the three rules; paired deltas vs MoOP l0keep (bootstrap 200k; LingoQA resamples the 100 segments).
Writes outputs/cvpr27/comb_ablation/{metrics.json, table.md, config.json}.

  .venv/bin/python cvpr27/code/analyze_comb_ablation.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import analyze_a100 as az

A = az.REPO / "outputs" / "cvpr27" / "a100"
ARMS = [("moop_l0keep", "MoOP l0keep (rank max)"), ("rank_mean", "rank-mean"), ("raw_mean", "raw mean"),
        ("raw_max", "raw max"), ("fm_only_l0keep", "FM-only"), ("ce_only_l0keep", "CE-only")]
REF = "moop_l0keep"
H = ["1.6s", "3.2s", "6.4s"]


def star(lo, hi):
    return "*" if lo > 0 or hi < 0 else ""


def main():
    res, md = {}, ["# Combination-rule ablation (A100, u40, l0keep)", ""]
    for s in ("test2000", "oodval"):
        rows = {a: az.load_rows(A / f"{a}_{s}", az.ARMS[a]) for a, _ in ARMS}
        ids = sorted(set.intersection(*(set(r) for r in rows.values())))
        md += [f"## {s} (n={len(ids)})", "",
               ("| rule | minADE@6 1.6 / 3.2 / 6.4 s | minFDE@6 1.6 / 3.2 / 6.4 s | CoC degenerate | "
                "d minADE@6 6.4 s vs MoOP l0keep | d minFDE@6 6.4 s |"), "|---|---|---|---|---|---|"]
        for a, name in ARMS:
            e = {"n": len(ids)}
            for k in ("ade", "fde"):
                for h in H:
                    v = np.array([az.metric(rows[a][c], k, h) for c in ids])
                    e[f"min{k.upper()}6_{h}"] = float(v.mean())
                    e[f"min{k.upper()}6_{h}_median"] = float(np.median(v))
                    if a != REF:
                        d = v - np.array([az.metric(rows[REF][c], k, h) for c in ids])
                        lo, hi = az.boot_ci(d)
                        e[f"d_min{k.upper()}6_{h}"] = {"mean": float(d.mean()), "ci": [lo, hi],
                                                        "median": float(np.median(d))}
            e["coc_degenerate"] = float(np.mean([rows[a][c]["coc_degenerate"] for c in ids]))
            res[f"{s}/{a}"] = e
            ab = " / ".join(f"{e[f'minADE6_{h}']:.3f}" for h in H)
            fb = " / ".join(f"{e[f'minFDE6_{h}']:.3f}" for h in H)
            if a == REF:
                da = df = "--"
            else:
                x, y = e["d_minADE6_6.4s"], e["d_minFDE6_6.4s"]
                da = f"{x['mean']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]{star(*x['ci'])}"
                df = f"{y['mean']:+.3f} [{y['ci'][0]:+.3f}, {y['ci'][1]:+.3f}]{star(*y['ci'])}"
            md.append(f"| {name} | {ab} | {fb} | {e['coc_degenerate']:.1%} | {da} | {df} |")
        md.append("")
    # LingoQA
    lq = {a: az.lingo(A, a) for a, _ in ARMS}
    md += ["## LingoQA (500 questions, Lingo-Judge)", "", "| rule | accuracy | d vs MoOP l0keep (segment bootstrap) |",
           "|---|---|---|"]
    base = lq[REF]
    for a, name in ARMS:
        v = lq[a]
        e = {"accuracy": v["accuracy"]}
        if a != REF:
            qs = sorted(set(v["rows"]) & set(base["rows"]))
            d = [float(v["rows"][q]["correct"]) - float(base["rows"][q]["correct"]) for q in qs]
            seg = [v["rows"][q]["segment_id"] for q in qs]
            lo, hi = az.cluster_boot_ci(d, seg)
            e["d_vs_ref"] = {"mean": float(np.mean(d)), "ci": [lo, hi], "n": len(qs)}
            dd = f"{100 * np.mean(d):+.1f} pp [{100 * lo:+.1f}, {100 * hi:+.1f}]{star(lo, hi)}"
        else:
            dd = "--"
        res[f"lingoqa/{a}"] = e
        md.append(f"| {name} | {100 * v['accuracy']:.1f}% | {dd} |")
    dst = az.REPO / "outputs" / "cvpr27" / "comb_ablation"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "metrics.json").write_text(json.dumps(res, indent=2))
    (dst / "table.md").write_text("\n".join(md) + "\n")
    (dst / "config.json").write_text(json.dumps({"arms": ARMS, "ref": REF, "n_boot": az.N_BOOT, "k": az.K,
                                                 "plan": "cvpr27/plans/2026-11-04_combination-rule-ablation.md"},
                                                indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
