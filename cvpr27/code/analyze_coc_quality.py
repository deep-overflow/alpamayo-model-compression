"""OOD CoC quality on the A100 rows: reference-CoC NLL and Claude grades, per arm and set.

cvpr27/plans/2026-11-02_oodtrain-coc-quality.md.

  --stage gate    G1: A100 dense / MoOP OOD-val nll_gtcoc vs the stored Ada rows (baseline_pred_oodval,
                  dual_u40_v2_pred_oodval). PASS = all 262 clips, mean per-clip |d| <= 0.02, |mean d| <= 0.01.
  --stage table   nll_gtcoc and grade columns for every arm/set present, paired deltas vs dense and MoOP
                  (bootstrap 200k, analyze_a100.boot_ci), per-cluster scores, grade-vs-minADE@6 rank correlation,
                  the judge's repeat agreement and the mismatched-reference chance floor.

  .venv/bin/python cvpr27/code/analyze_coc_quality.py --stage table [--judge-exp ood_coc_v1]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import analyze_a100 as az

OUT = REPO / "outputs"
A100 = OUT / "cvpr27" / "a100"
ARMS = ["dense", "moop", "moop_l0keep", "lp_m24", "tyr_off", "wandasp", "tyr_vt"]
SETS = {"oodval": ("ood_val.parquet", 262), "oodtrain": ("ood_train.parquet", 1271)}
REFS = ["dense", "moop"]
GRADE_COLS = ["score", "action", "cause", "contradiction", "coherent"]


def rows_of(d, key):
    """clip_id -> row from every *_s*of*.json shard under d (last write wins)."""
    rows = {}
    for p in sorted(d.glob("*_s*of*.json")):
        for r in json.loads(p.read_text()):
            rows[r["clip_id"]] = r if key is None else r[key]
    return rows


def gate():
    ref_dirs = {"dense": "baseline_pred_oodval", "moop": "dual_u40_v2_pred_oodval"}
    ok_all = True
    lines = ["G1: A100 nll_gtcoc (run_nll_gtcoc.py) vs stored Ada rows (run_baseline TF path), OOD-val"]
    for arm, ref in ref_dirs.items():
        a = rows_of(A100 / f"nll_{arm}_oodval", "nll_gtcoc")
        b = {c: r["nll_gtcoc"] for c, r in rows_of(OUT / ref, None).items() if "nll_gtcoc" in r}
        ids = sorted(set(a) & set(b))
        d = np.array([a[c] - b[c] for c in ids])
        ok = len(ids) == 262 and np.abs(d).mean() <= 0.02 and abs(d.mean()) <= 0.01
        ok_all &= bool(ok)
        lines.append(f"  {arm:5s} n={len(ids)}  A100 {np.mean([a[c] for c in ids]):.4f}  Ada "
                     f"{np.mean([b[c] for c in ids]):.4f}  mean d {d.mean():+.4f}  mean|d| {np.abs(d).mean():.4f}  "
                     f"max|d| {np.abs(d).max():.4f}  -> {'PASS' if ok else 'FAIL'}")
    lines.append(f"G1 {'PASS' if ok_all else 'FAIL'}")
    print("\n".join(lines))
    dst = OUT / "cvpr27" / "coc_quality"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "gate.txt").write_text("\n".join(lines) + "\n")
    return ok_all


def paired(a, b, ids):
    d = np.array([a[c] - b[c] for c in ids], dtype=float)
    lo, hi = az.boot_ci(d)
    return {"n": len(ids), "delta": float(d.mean()), "ci": [lo, hi], "median": float(np.median(d))}


def fmt_p(p):
    star = "*" if p["ci"][0] > 0 or p["ci"][1] < 0 else ""
    return f"{p['delta']:+.3f} [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]{star}"


def table(judge_exp):
    grades = None
    gpath = OUT / "cvpr27" / "coc_judge" / judge_exp / "grades.parquet"
    if gpath.exists():
        g = pd.read_parquet(gpath)
        grades = g[g.status == "ok"]
    res, md = {}, ["# OOD CoC quality (A100 rows)", ""]
    for s, (manifest, n_exp) in SETS.items():
        ids_all = list(pd.read_parquet(OUT / "eval_sets" / manifest).clip_id)
        cluster = dict(pd.read_parquet(OUT / "eval_sets" / manifest)[["clip_id", "cluster"]].values)
        nll = {a: rows_of(A100 / f"nll_{a}_{s}", "nll_gtcoc") for a in ARMS}
        ol = {a: rows_of(A100 / f"{a}_{s}", None) for a in ARMS}
        sc = {}
        if grades is not None:
            for a in ARMS + ["ctrl_mismatch"]:
                gg = grades[(grades.set == s) & (grades.arm == a) & (grades.rep == 0)]
                if len(gg):
                    sc[a] = gg.set_index("clip_id")
        md += [f"## {s}", "", ("| arm | n NLL | nll_gtcoc | n graded | score | score>=4 | action=2 | cause=2 | "
                               "contradiction | coherent | CoC degenerate | gen len |"), "|" + "---|" * 12]
        for a in ARMS + ["ctrl_mismatch"]:
            e = {}
            if nll.get(a):
                v = [nll[a][c] for c in ids_all if c in nll[a]]
                e.update({"nll_n": len(v), "nll_gtcoc": float(np.mean(v)), "nll_complete": len(v) == n_exp})
            if a in sc:
                t = sc[a]
                e.update({"graded_n": len(t), "score": float(t.score.mean()), "score_ge4": float((t.score >= 4).mean()),
                          "action_match": float((t.action == 2).mean()), "cause_match": float((t.cause == 2).mean()),
                          "contradiction": float(t.contradiction.mean()), "coherent": float(t.coherent.mean())})
            if ol.get(a):
                have = [c for c in ids_all if c in ol[a]]
                e.update({"ol_n": len(have),
                          "coc_degenerate": float(np.mean([ol[a][c]["coc_degenerate"] for c in have])),
                          "gen_len": float(np.mean([ol[a][c]["gen_len"] for c in have]))})
            if not e:
                continue
            res[f"{s}/{a}"] = e

            def f(k, spec, e=e):
                return format(e[k], spec) if k in e else "--"
            md.append(f"| {a} | {e.get('nll_n', '--')} | {f('nll_gtcoc', '.3f')} | {e.get('graded_n', '--')} | "
                      f"{f('score', '.2f')} | {f('score_ge4', '.1%')} | {f('action_match', '.1%')} | "
                      f"{f('cause_match', '.1%')} | {f('contradiction', '.1%')} | {f('coherent', '.1%')} | "
                      f"{f('coc_degenerate', '.1%')} | {f('gen_len', '.1f')} |")

        md += ["", "Paired deltas (mean [95% bootstrap CI]; * = CI excludes 0). NLL: lower is better; score: higher.",
               "", "| arm | vs | n | d nll_gtcoc | n | d score |", "|---|---|---|---|---|---|"]
        for a in ARMS:
            for ref in REFS:
                if a == ref or a in REFS[: REFS.index(ref)]:
                    continue
                row, key = [], f"{s}/{a}/{ref}"
                res[key] = {}
                if nll.get(a) and nll.get(ref):
                    ids = [c for c in ids_all if c in nll[a] and c in nll[ref]]
                    res[key]["nll"] = p = paired(nll[a], nll[ref], ids)
                    row += [str(p["n"]), fmt_p(p)]
                else:
                    row += ["--", "--"]
                if a in sc and ref in sc:
                    sa, sr = sc[a].score.to_dict(), sc[ref].score.to_dict()
                    ids = [c for c in ids_all if c in sa and c in sr]
                    res[key]["score"] = p = paired(sa, sr, ids)
                    row += [str(p["n"]), fmt_p(p)]
                else:
                    row += ["--", "--"]
                if row != ["--"] * 4:
                    md.append(f"| {a} | {ref} | " + " | ".join(row) + " |")

        if sc:
            # per-cluster mean score, and the clip-level rank correlation of score with minADE@6
            md += ["", "Mean score by cluster:", ""]
            clusters = sorted({cluster[c] for c in ids_all})
            arms_g = [a for a in ARMS if a in sc]
            md += ["| cluster | n | " + " | ".join(arms_g) + " |", "|" + "---|" * (len(arms_g) + 2)]
            for cl in clusters:
                cids = [c for c in ids_all if cluster[c] == cl]
                cells = [f"{sc[a].score.reindex(cids).mean():.2f}" for a in arms_g]
                md.append(f"| {cl} | {len(cids)} | " + " | ".join(cells) + " |")
            md += ["", "Spearman rho (clip level) of score vs minADE@6 (6.4 s):", ""]
            for a in arms_g:
                if not ol.get(a):
                    continue
                cids = [c for c in sc[a].index if c in ol[a]]
                x = pd.Series([sc[a].score[c] for c in cids], dtype=float)
                y = pd.Series([az.metric(ol[a][c], "ade") for c in cids])
                rho = float(x.corr(y, method="spearman"))
                res[f"{s}/{a}"]["rho_score_minade6"] = rho
                md.append(f"- {a}: rho {rho:+.3f} (n={len(cids)})")
        md.append("")

    if grades is not None:
        reps = grades.pivot_table(index=["arm", "set", "clip_id"], columns="rep", values="score", aggfunc="first")
        if 1 in reps.columns:
            reps = reps.dropna()
            res["judge_repeat"] = {"n": len(reps), "same": float((reps[0] == reps[1]).mean()),
                                   "within_1": float(((reps[0] - reps[1]).abs() <= 1).mean())}
            md.append(f"Judge repeat ({len(reps)} pairs): same score {res['judge_repeat']['same']:.1%}, "
                      f"within 1 {res['judge_repeat']['within_1']:.1%}")
    dst = OUT / "cvpr27" / "coc_quality"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "metrics.json").write_text(json.dumps(res, indent=2))
    (dst / "table.md").write_text("\n".join(md) + "\n")
    (dst / "config.json").write_text(json.dumps({"arms": ARMS, "sets": SETS, "refs": REFS, "judge_exp": judge_exp,
                                                 "n_boot": az.N_BOOT}, indent=2))
    print("\n".join(md))
    print(f"-> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["gate", "table"])
    ap.add_argument("--judge-exp", default="ood_coc_v1")
    args = ap.parse_args()
    if args.stage == "gate":
        sys.exit(0 if gate() else 1)
    table(args.judge_exp)


if __name__ == "__main__":
    main()
