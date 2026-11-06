"""Ratio sweep (cvpr27/plans/2026-11-03_baselines-ratio-sweep.md, OOD-train in §8), A100 rows.

Absolute minADE@6 / minFDE@6 at 1.6 / 3.2 / 6.4 s on test_2000, OOD-Val and OOD-train, CoC-degenerate share and LingoQA
for every method at u35/u40/u45/u50,
and paired deltas vs MoOP l0keep at the same budget for all six metrics (bootstrap 200k; LingoQA resamples the 100
segments). The u40 column uses the arms already on the A100 table (lp_m24, tyr_off, wandasp, fm/ce_only_l0keep).
Writes outputs/cvpr27/ratio_sweep/{metrics.json, table.md, config.json} and the curve figure.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import analyze_a100 as az

A = az.REPO / "outputs" / "cvpr27" / "a100"
RATIOS = [35, 40, 45, 50]
REMOVED = {35: 20.92, 40: 23.99, 45: 26.85, 50: 29.98}
# method -> arm per ratio
METHODS = {
    "MoOP l0keep": {35: "l0keep_u35", 40: "moop_l0keep", 45: "l0keep_u45", 50: "l0keep_u50"},
    "FM-only": {35: "fm_only_u35", 40: "fm_only_l0keep", 45: "fm_only_u45", 50: "fm_only_u50"},
    "CE-only": {35: "ce_only_u35", 40: "ce_only_l0keep", 45: "ce_only_u45", 50: "ce_only_u50"},
    "Tyr": {35: "tyr_u35", 40: "tyr_off", 45: "tyr_u45", 50: "tyr_u50"},
    "LLM-Pruner": {35: "lp_u35", 40: "lp_m24", 45: "lp_u45", 50: "lp_u50"},
    "Wanda-sp": {35: "wandasp_u35", 40: "wandasp", 45: "wandasp_u45", 50: "wandasp_u50"},
}
H = ["1.6s", "3.2s", "6.4s"]
COLS = [(k, h) for k in ("ade", "fde") for h in H]


def star(lo, hi):
    return "*" if lo > 0 or hi < 0 else ""


def main():
    res = {}
    for s in ("test2000", "oodval", "oodtrain"):
        ids = az.manifest_ids(az.REPO / "outputs", s)
        for r in RATIOS:
            ref = az.load_rows(A / f"{METHODS['MoOP l0keep'][r]}_{s}", az.ARMS[METHODS["MoOP l0keep"][r]])
            for m, arms in METHODS.items():
                a = arms[r]
                rows = az.load_rows(A / f"{a}_{s}", az.ARMS[a])
                have = [c for c in ids if c in rows and c in ref]
                e = {"arm": a, "n": len(have),
                     "coc_degenerate": float(np.mean([rows[c]["coc_degenerate"] for c in have]))}
                for k, h in COLS:
                    v = np.array([az.metric(rows[c], k, h) for c in have])
                    e[f"{k}_{h}"] = float(v.mean())
                    if m != "MoOP l0keep":
                        d = v - np.array([az.metric(ref[c], k, h) for c in have])
                        lo, hi = az.boot_ci(d)
                        e[f"d_{k}_{h}"] = [float(d.mean()), lo, hi]
                res[f"{s}/u{r}/{m}"] = e
    for r in RATIOS:
        base = az.lingo(A, METHODS["MoOP l0keep"][r])
        for m, arms in METHODS.items():
            v = az.lingo(A, arms[r])
            e = {"accuracy": v["accuracy"]}
            if m != "MoOP l0keep":
                qs = sorted(set(v["rows"]) & set(base["rows"]))
                d = [float(v["rows"][q]["correct"]) - float(base["rows"][q]["correct"]) for q in qs]
                lo, hi = az.cluster_boot_ci(d, [v["rows"][q]["segment_id"] for q in qs])
                e["d"] = [float(np.mean(d)), lo, hi]
            res[f"lingoqa/u{r}/{m}"] = e

    md = ["# Ratio sweep (A100): MoOP l0keep vs FM-only / CE-only / Tyr / LLM-Pruner / Wanda-sp", ""]
    for s, n in (("test2000", 2000), ("oodval", 262), ("oodtrain", 1271)):
        short = sorted({res[f"{s}/u{r}/{m}"]["n"] for r in RATIOS for m in METHODS} - {n})
        if short:  # an arm with missing rows is compared on the clips it has; say so instead of hiding it
            md += [f"> {s}: some arms have fewer than {n} clips (n = {short}); see metrics.json", ""]
        md += [f"## {s} (n={n}) -- absolute", "",
               "| ratio (removed) | method | minADE@6 1.6 s | 3.2 s | 6.4 s | minFDE@6 1.6 s | 3.2 s | 6.4 s | CoC degenerate |",
               "|---|---|---|---|---|---|---|---|---|"]
        for r in RATIOS:
            for m in METHODS:
                e = res[f"{s}/u{r}/{m}"]
                vals = " | ".join(f"{e[f'{k}_{h}']:.3f}" for k, h in COLS)
                md.append(f"| u{r} ({REMOVED[r]:.2f}%) | {m} | {vals} | {e['coc_degenerate']:.1%} |")
        md += ["", f"## {s} -- paired delta vs MoOP l0keep at the same ratio (mean [95% CI]; * = CI excludes 0)", "",
               "| ratio | method | d minADE@6 1.6 s | 3.2 s | 6.4 s | d minFDE@6 1.6 s | 3.2 s | 6.4 s |",
               "|---|---|---|---|---|---|---|---|"]
        for r in RATIOS:
            for m in METHODS:
                if m == "MoOP l0keep":
                    continue
                e = res[f"{s}/u{r}/{m}"]
                cells = " | ".join(f"{e[f'd_{k}_{h}'][0]:+.3f} [{e[f'd_{k}_{h}'][1]:+.3f}, {e[f'd_{k}_{h}'][2]:+.3f}]"
                                   f"{star(*e[f'd_{k}_{h}'][1:])}" for k, h in COLS)
                md.append(f"| u{r} | {m} | {cells} |")
        md.append("")
    md += ["## LingoQA (500 questions; dense 72.6%)", "", "| ratio | " + " | ".join(METHODS) + " |",
           "|" + "---|" * (len(METHODS) + 1)]
    for r in RATIOS:
        cells = []
        for m in METHODS:
            e = res[f"lingoqa/u{r}/{m}"]
            c = f"{100 * e['accuracy']:.1f}%"
            if "d" in e:
                c += f" ({100 * e['d'][0]:+.1f}{star(*e['d'][1:])})"
            cells.append(c)
        md.append(f"| u{r} | " + " | ".join(cells) + " |")
    md.append("")
    dst = az.REPO / "outputs" / "cvpr27" / "ratio_sweep"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "metrics.json").write_text(json.dumps(res, indent=2))
    (dst / "table.md").write_text("\n".join(md) + "\n")
    (dst / "config.json").write_text(json.dumps({"methods": METHODS, "removed_pct": REMOVED, "n_boot": az.N_BOOT,
                                                 "k": az.K, "ref": "MoOP l0keep at the same ratio"}, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
