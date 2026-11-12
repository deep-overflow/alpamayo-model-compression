"""Paper Table 14 ("Combination rules with 95% confidence intervals"): LingoQA, Test500 and OOD-Val262 for the four rules under
the paper's pre-l0keep recipe (layer 0 cut like every layer, importance_v2, u40), all Ada.

  max-rank (MoOP)  the paper's rows: dual_u40_v2_pred_test / _pred_oodval, lingo_vqa_slim_dual_u40_v2
  mean-rank        dualsum_u40_v2: Test500 from iclr27-rebuttal/comb_pre_l0keep/rank_mean_test500 when complete, else the
                   September run dualsum_u40_v2_test (same build, same card type); OOD-Val262 and LingoQA from comb_pre_l0keep
  max-raw / mean-raw  rawabsmean_u40_v2 / rawabsavg_u40_v2 from comb_pre_l0keep (run_comb_pre_l0keep.sh)

Each cell is the same estimator as the max-rank cells already in the paper (ci_openloop_lingoqa): open loop = mean minADE@6
6.4 s with a 200,000-draw clip bootstrap (seed 0); LingoQA = Lingo-Judge accuracy with a 200,000-draw bootstrap over the 100
segments. Missing runs print [x]. Also written: all horizons (minADE@6 / minFDE@6 at 1.6 / 3.2 / 6.4 s) and paired rule -
max-rank differences, for the report.
Usage (repo root): .venv/bin/python iclr27-rebuttal/code/table14_comb_rules.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "cvpr27" / "code"))
sys.path.insert(0, str(REPO / "iclr27-rebuttal" / "code"))
from analyze_a100 import HORIZONS, load_rows, metric
from ci_openloop_lingoqa import fmt_ci, lingo_block, star, summarise

OUT = REPO / "outputs"
C = "iclr27-rebuttal/comb_pre_l0keep"
SETS = {"test500": ("test_500.parquet", 500), "oodval262": ("ood_val.parquet", 262),
        "oodtrain1271": ("ood_train.parquet", 1271)}   # OOD-train added 2026-11-13 (run_comb_oodtrain.sh, Ada)
# row -> (tag, {set: [run dirs, first complete wins]}, LingoQA dir)
ROWS = {
    "Max-rank (MoOP)": ("slim_dual_u40_v2", {"test500": ["dual_u40_v2_pred_test"], "oodval262": ["dual_u40_v2_pred_oodval"],
                                             "oodtrain1271": [f"{C}/max_rank_oodtrain1271"]},
                        "lingo_vqa_slim_dual_u40_v2"),
    "Mean-rank": ("slim_dualsum_u40_v2", {"test500": [f"{C}/rank_mean_test500", "dualsum_u40_v2_test"],
                                          "oodval262": [f"{C}/rank_mean_oodval262"],
                                          "oodtrain1271": [f"{C}/rank_mean_oodtrain1271"]}, f"{C}/lingo_vqa_rank_mean"),
    "Max-raw": ("slim_rawabsmean_u40_v2", {"test500": [f"{C}/raw_max_test500"], "oodval262": [f"{C}/raw_max_oodval262"],
                                           "oodtrain1271": [f"{C}/raw_max_oodtrain1271"]}, f"{C}/lingo_vqa_raw_max"),
    "Mean-raw": ("slim_rawabsavg_u40_v2", {"test500": [f"{C}/raw_mean_test500"], "oodval262": [f"{C}/raw_mean_oodval262"],
                                           "oodtrain1271": [f"{C}/raw_mean_oodtrain1271"]}, f"{C}/lingo_vqa_raw_mean"),
}
REF = "Max-rank (MoOP)"


def gpu_of(d):
    p = OUT / d / "config.json"
    return json.loads(p.read_text()).get("gpu") if p.exists() else None


def main():
    used, ol = {}, {}
    for set_name, (man, n) in SETS.items():
        ids = list(pd.read_parquet(OUT / "eval_sets" / man)["clip_id"])
        rows = {}
        for row, (tag, runs, _) in ROWS.items():
            for d in runs[set_name]:
                r = load_rows(OUT / d, tag) if (OUT / d).exists() else {}
                if all(c in r for c in ids):
                    assert "Ada" in str(gpu_of(d)), (d, gpu_of(d))
                    rows[row], used[f"{row}|{set_name}"] = r, d
                    break
        series = {f"abs|{row}|{k}|{h}": [metric(r[c], k, h) for c in ids]
                  for row, r in rows.items() for k in ("ade", "fde") for h in HORIZONS}
        series.update({f"d|{row}|{k}|{h}": [metric(r[c], k, h) - metric(rows[REF][c], k, h) for c in ids]
                       for row, r in rows.items() if row != REF for k in ("ade", "fde") for h in HORIZONS})
        ol[set_name] = {"n": n, "rows": list(rows), "stats": summarise(series) if series else {}}

    ldirs = {row: OUT / v[2] for row, v in ROWS.items() if (OUT / v[2] / "scored.json").exists()}
    for row, d in ldirs.items():
        assert "Ada" in str(json.loads((d / "config.json").read_text()).get("gpu")), d
        used[f"{row}|lingoqa"] = str(d.relative_to(OUT))
    lq = lingo_block(ldirs, [REF]) if REF in ldirs else {"stats": {}}
    for k, v in list(lq["stats"].items()):   # rule - rank-max, renamed to the open-loop keys
        if k.endswith(f"|{REF}"):
            lq["stats"][f"d|{k.split('|')[0]}"] = v

    def cell_ol(row, set_name, kind="ade"):
        e = ol[set_name]["stats"].get(f"abs|{row}|{kind}|6.4s")
        return ("[x]", "[x]") if e is None else (fmt_ci(e), f"{e['mean']:.3f} {{\\scriptsize [{e['ci'][0]:.3f}, {e['ci'][1]:.3f}]}}")

    def cell_lq(row):
        e = lq["stats"].get(f"{row}|abs")
        if e is None:
            return "[x]", "[x]"
        m, (lo, hi) = e["mean"] * 100, (e["ci"][0] * 100, e["ci"][1] * 100)
        return f"{m:.1f} [{lo:.1f}, {hi:.1f}]", f"{m:.1f} {{\\scriptsize [{lo:.1f}, {hi:.1f}]}}"

    md = ["# Table 14 -- combination rules with 95% confidence intervals (pre-l0keep recipe, Ada)", "",
          ("| Rule | LingoQA (%) | Test500 minADE@6 (m) | OOD-Val262 minADE@6 (m) | OOD-train1271 minADE@6 (m) |"),
          "|---|---|---|---|---|"]
    tex, tex4, tex_fde = [], [], []
    for row in ROWS:
        a, b, c, d = cell_lq(row), cell_ol(row, "test500"), cell_ol(row, "oodval262"), cell_ol(row, "oodtrain1271")
        bf, cf = cell_ol(row, "test500", "fde"), cell_ol(row, "oodval262", "fde")
        md.append(f"| {row} | {a[0]} | {b[0]} | {c[0]} | {d[0]} |")
        tex.append(f"{row} & {a[1]} & {b[1]} & {c[1]} \\\\")
        tex4.append(f"{row} & {a[1]} & {b[1]} & {c[1]} & {d[1]} \\\\")
        tex_fde.append(f"{row} & {a[1]} & {b[1]} & {bf[1]} & {c[1]} & {cf[1]} \\\\")
    md += ["", "LaTeX rows, the paper's layout (LingoQA | Test500 minADE@6, minFDE@6 | OOD-Val262 minADE@6, minFDE@6; 6.4 s):",
           "", "```", *tex_fde, "```", "",
           "LaTeX rows, three columns (LingoQA | Test500 | OOD-Val262 minADE@6):", "", "```", *tex, "```", "",
           "LaTeX rows with OOD-train as a fourth column:", "", "```", *tex4, "```", "",
           "## Paired difference rule - max-rank (same clips / questions; * = CI excludes 0)", "",
           "| Rule | LingoQA (pp) | Test500 dminADE@6 6.4 s | OOD-Val262 dminADE@6 6.4 s | OOD-train1271 dminADE@6 6.4 s |",
           "|---|---|---|---|---|"]
    for row in ROWS:
        if row == REF:
            continue
        e = lq["stats"].get(f"d|{row}")
        cl = (fmt_ci(e, nd=1, scale=100, sign=True) + star(e)) if e else "[x]"
        cs = []
        for s in SETS:
            d = ol[s]["stats"].get(f"d|{row}|ade|6.4s")
            cs.append(fmt_ci(d, sign=True) + star(d) if d else "[x]")
        md.append(f"| {row} | {cl} | " + " | ".join(cs) + " |")
    for s, blk in ol.items():
        md += ["", f"## {s}: all horizons (mean [95% clip-bootstrap CI])", "",
               "| Rule | minADE@6 1.6 s | 3.2 s | 6.4 s | minFDE@6 1.6 s | 3.2 s | 6.4 s |", "|---|---|---|---|---|---|---|"]
        for row in blk["rows"]:
            md.append(f"| {row} | " + " | ".join(fmt_ci(blk["stats"][f"abs|{row}|{k}|{h}"]) for k in ("ade", "fde")
                                                  for h in HORIZONS) + " |")
        md += ["", "| Rule - max-rank | dADE 1.6 s | 3.2 s | 6.4 s | dFDE 1.6 s | 3.2 s | 6.4 s |", "|---|---|---|---|---|---|---|"]
        for row in blk["rows"]:
            if row != REF:
                md.append(f"| {row} | " + " | ".join(fmt_ci(blk["stats"][f"d|{row}|{k}|{h}"], sign=True)
                                                      + star(blk["stats"][f"d|{row}|{k}|{h}"])
                                                      for k in ("ade", "fde") for h in HORIZONS) + " |")
    md += ["", "Runs used:", "", *[f"- {k}: `{v}`" for k, v in used.items()]]

    dst = OUT / "iclr27-rebuttal" / "table14_comb_rules"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "config.json").write_text(json.dumps({"rows": {k: {"tag": v[0], "runs": v[1], "lingoqa": v[2]} for k, v in ROWS.items()},
                                                 "runs_used": used, "n_boot": 200_000, "seed": 0,
                                                 "plan": "iclr27-rebuttal/plans/2026-11-08_comb-rules-pre-l0keep-test500.md"},
                                                indent=2))
    (dst / "metrics.json").write_text(json.dumps({"open_loop": ol, "lingoqa": lq}, indent=2))
    (dst / "summary.txt").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
