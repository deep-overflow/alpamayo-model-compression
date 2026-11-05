"""Every A100 arm at (or built on) MoOP's u40 budget: absolute minADE@6 / minFDE@6 at 1.6 / 3.2 / 6.4 s on test2000,
OOD-val and OOD-train (where run), CoC-degenerate share, LingoQA, and paired deltas vs MoOP l0keep for all six metrics
(bootstrap 200k; LingoQA resamples the 100 segments). Writes outputs/cvpr27/u40_summary/{metrics.json, table.md}.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import analyze_a100 as az

A = az.REPO / "outputs" / "cvpr27" / "a100"
REF = "moop_l0keep"
GROUPS = [
    ("Reference", [("dense", "dense (no pruning)"), ("moop", "MoOP (shipped, layer 0 cut)"),
                   ("moop_l0keep", "MoOP l0keep")]),
    ("External baselines (same budget)", [("lp_m24", "LLM-Pruner (budget-matched)"), ("lp_r50", "LLM-Pruner r=0.5 (24.99%)"),
                                          ("tyr_off", "Tyr official"), ("tyr_u40_r", "Tyr simplified port"),
                                          ("tyr_vt", "Tyr official, Hessian V:T 1:1"), ("wandasp", "Wanda-sp")]),
    ("Criterion ablation", [("fm_only_l0keep", "FM-only"), ("ce_only_l0keep", "CE-only"),
                            ("rank_mean", "rank-mean"), ("raw_mean", "raw mean"), ("raw_max", "raw max")]),
    ("Extra cuts on MoOP l0keep", [("l0keep_l14m95", "+ VLM L1-4 MLP to 95% mass"),
                                   ("l0keep_l14m90", "+ VLM L1-4 MLP to 90% mass"),
                                   ("l0keep_l14m80", "+ VLM L1-4 MLP to 80% mass"),
                                   ("l0keep_em75", "+ expert MLP 75%"), ("l0keep_em87p5", "+ expert MLP 87.5%"),
                                   ("l0keep_em93p75", "+ expert MLP 93.75%")]),
]
H = ["1.6s", "3.2s", "6.4s"]
COLS = [(k, h) for k in ("ade", "fde") for h in H]
SETS = [("test2000", 2000), ("oodval", 262), ("oodtrain", 1271)]


def star(lo, hi):
    return "*" if lo > 0 or hi < 0 else ""


def main():
    res = {}
    for s, n in SETS:
        ids = az.manifest_ids(az.REPO / "outputs", s)
        ref = az.load_rows(A / f"{REF}_{s}", az.ARMS[REF])
        for _, arms in GROUPS:
            for a, _ in arms:
                rows = az.load_rows(A / f"{a}_{s}", az.ARMS[a])
                have = [c for c in ids if c in rows and c in ref]
                if len(have) < n:
                    continue
                e = {"n": len(have), "coc_degenerate": float(np.mean([rows[c]["coc_degenerate"] for c in have]))}
                for k, h in COLS:
                    v = np.array([az.metric(rows[c], k, h) for c in have])
                    e[f"{k}_{h}"] = float(v.mean())
                    if a != REF:
                        d = v - np.array([az.metric(ref[c], k, h) for c in have])
                        lo, hi = az.boot_ci(d)
                        e[f"d_{k}_{h}"] = [float(d.mean()), lo, hi]
                res[f"{s}/{a}"] = e
    base = az.lingo(A, REF)
    for _, arms in GROUPS:
        for a, _ in arms:
            v = az.lingo(A, a)
            if v is None:
                continue
            e = {"accuracy": v["accuracy"]}
            if a != REF:
                qs = sorted(set(v["rows"]) & set(base["rows"]))
                d = [float(v["rows"][q]["correct"]) - float(base["rows"][q]["correct"]) for q in qs]
                lo, hi = az.cluster_boot_ci(d, [v["rows"][q]["segment_id"] for q in qs])
                e["d"] = [float(np.mean(d)), lo, hi]
            res[f"lingoqa/{a}"] = e

    md = ["# A100 arms at MoOP's u40 budget", ""]
    for s, n in SETS:
        md += [f"## {s} (n={n})", "", "| group | arm | minADE@6 1.6 s | 3.2 s | 6.4 s | minFDE@6 1.6 s | 3.2 s | 6.4 s | CoC degenerate |",
               "|---|---|---|---|---|---|---|---|---|"]
        for g, arms in GROUPS:
            for a, name in arms:
                e = res.get(f"{s}/{a}")
                if e is None:
                    continue
                md.append(f"| {g} | {name} | " + " | ".join(f"{e[f'{k}_{h}']:.3f}" for k, h in COLS)
                          + f" | {e['coc_degenerate']:.1%} |")
        md += ["", f"### {s}: paired delta vs MoOP l0keep (mean [95% CI]; * = CI excludes 0)", "",
               "| arm | d minADE@6 1.6 s | 3.2 s | 6.4 s | d minFDE@6 1.6 s | 3.2 s | 6.4 s |", "|---|---|---|---|---|---|---|"]
        for _, arms in GROUPS:
            for a, name in arms:
                e = res.get(f"{s}/{a}")
                if e is None or a == REF:
                    continue
                md.append(f"| {name} | " + " | ".join(
                    f"{e[f'd_{k}_{h}'][0]:+.3f} [{e[f'd_{k}_{h}'][1]:+.3f}, {e[f'd_{k}_{h}'][2]:+.3f}]{star(*e[f'd_{k}_{h}'][1:])}"
                    for k, h in COLS) + " |")
        md.append("")
    md += ["## LingoQA (500 questions)", "", "| arm | accuracy | d vs MoOP l0keep |", "|---|---|---|"]
    for _, arms in GROUPS:
        for a, name in arms:
            e = res.get(f"lingoqa/{a}")
            if e is None:
                md.append(f"| {name} | = MoOP l0keep (VLM unchanged) | -- |")
                continue
            d = "--" if "d" not in e else f"{100 * e['d'][0]:+.1f} pp [{100 * e['d'][1]:+.1f}, {100 * e['d'][2]:+.1f}]{star(*e['d'][1:])}"
            md.append(f"| {name} | {100 * e['accuracy']:.1f}% | {d} |")
    dst = az.REPO / "outputs" / "cvpr27" / "u40_summary"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "metrics.json").write_text(json.dumps(res, indent=2))
    (dst / "table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
