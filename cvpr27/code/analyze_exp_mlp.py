"""MoOP l0keep + expert MLP-only ladder (cvpr27/plans/2026-11-05_moop-l0keep-expert-mlp.md), A100 rows.

Absolute minADE@6 / minFDE@6 at 1.6 / 3.2 / 6.4 s on test2000, OOD-val and OOD-train; paired deltas vs MoOP l0keep at
every horizon (bootstrap 200k); share of clips whose generated CoC is identical to MoOP l0keep's (the VLM is the
same, so this checks that only the expert differs); bf16 weight memory saved.
Writes outputs/cvpr27/exp_mlp/{metrics.json, table.md, config.json}.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import analyze_a100 as az

A = az.REPO / "outputs" / "cvpr27" / "a100"
REF = "moop_l0keep"
ARMS = [(REF, "MoOP l0keep (expert full)", 8256), ("l0keep_em75", "+ expert MLP 75%", 2064),
        ("l0keep_em87p5", "+ expert MLP 87.5%", 1032), ("l0keep_em93p75", "+ expert MLP 93.75%", 516)]
H = ["1.6s", "3.2s", "6.4s"]


def star(lo, hi):
    return "*" if lo > 0 or hi < 0 else ""


def main():
    res, md = {}, ["# MoOP l0keep + expert MLP-only (A100)", ""]
    md += ["| arm | expert MLP kept / layer | removed params | bf16 weights saved vs MoOP l0keep |", "|---|---|---|---|"]
    for a, name, keep in ARMS:
        extra = 36 * (8256 - keep) * 3 * 2048
        md.append(f"| {name} | {keep} | {2657447936 + extra:,} | {2 * extra / 1e9:.2f} GB |")
    md.append("")
    for s in ("test2000", "oodval", "oodtrain"):
        rows = {a: az.load_rows(A / f"{a}_{s}", az.ARMS[a]) for a, _, _ in ARMS}
        ids = sorted(set.intersection(*(set(r) for r in rows.values())))
        md += [f"## {s} (n={len(ids)})", "",
               ("| arm | minADE@6 1.6 / 3.2 / 6.4 s | minFDE@6 1.6 / 3.2 / 6.4 s | same CoC as MoOP l0keep | "
                "d minADE@6 1.6 / 3.2 / 6.4 s | d minFDE@6 6.4 s |"), "|---|---|---|---|---|---|"]
        for a, name, _ in ARMS:
            e = {"n": len(ids)}
            for k in ("ade", "fde"):
                for h in H:
                    v = np.array([az.metric(rows[a][c], k, h) for c in ids])
                    e[f"min{k.upper()}6_{h}"] = float(v.mean())
                    if a != REF:
                        d = v - np.array([az.metric(rows[REF][c], k, h) for c in ids])
                        lo, hi = az.boot_ci(d)
                        e[f"d_min{k.upper()}6_{h}"] = {"mean": float(d.mean()), "ci": [lo, hi],
                                                        "median": float(np.median(d))}
            e["same_coc"] = float(np.mean([rows[a][c]["gen_coc"] == rows[REF][c]["gen_coc"] for c in ids]))
            e["coc_degenerate"] = float(np.mean([rows[a][c]["coc_degenerate"] for c in ids]))
            res[f"{s}/{a}"] = e
            ab = " / ".join(f"{e[f'minADE6_{h}']:.3f}" for h in H)
            fb = " / ".join(f"{e[f'minFDE6_{h}']:.3f}" for h in H)
            if a == REF:
                da = df = "--"
            else:
                da = " / ".join(f"{e[f'd_minADE6_{h}']['mean']:+.3f}{star(*e[f'd_minADE6_{h}']['ci'])}" for h in H)
                y = e["d_minFDE6_6.4s"]
                df = f"{y['mean']:+.3f} [{y['ci'][0]:+.3f}, {y['ci'][1]:+.3f}]{star(*y['ci'])}"
                x = e["d_minADE6_6.4s"]
                da += f" ([{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}] at 6.4 s)"
            md.append(f"| {name} | {ab} | {fb} | {e['same_coc']:.1%} | {da} | {df} |")
        md.append("")
    dst = az.REPO / "outputs" / "cvpr27" / "exp_mlp"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "metrics.json").write_text(json.dumps(res, indent=2))
    (dst / "table.md").write_text("\n".join(md) + "\n")
    (dst / "config.json").write_text(json.dumps({"arms": ARMS, "ref": REF, "n_boot": az.N_BOOT, "k": az.K,
                                                 "plan": "cvpr27/plans/2026-11-05_moop-l0keep-expert-mlp.md"},
                                                indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
