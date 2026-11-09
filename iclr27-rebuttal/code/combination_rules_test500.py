"""VLM importance combination rules on Test500: rank-max (= MoOP l0keep), rank-mean, raw-mean, raw-max.

A100 (all four rules): outputs/cvpr27/a100/<arm>_test2000 restricted to the 500 test_500 clips (test_500 is a subset of
test_2000). Every rule uses the same importance (importance_v2, calib_100), removes the same parameter count, keeps the
32 layer-0 Q heads and leaves the expert uncut (cvpr27/reports/2026-11-05_combination-rule-ablation.md); the full
test_2000 means are printed as a check against that report.

Ada (2026-10-15 runs, outputs/rawmax_compare; rank-mean and raw-mean were never run on Ada with l0keep):
  rank-max   swap_ship_L0fullM_500 (VLM / expert masks identical to dual_u40_l0keep_v2)
  raw-max    rawmean (FM score = |mean over steps|, importance_v2) and rawabsmean (mean of |per step|, the A100 raw-max
             recipe, but on the Ada-measured importance_v2_ada, so it pairs with dual_ada = rank-max on importance_v2_ada)
  lnorm      max over scores divided by their per-layer means

Every arm: minADE@6 / minFDE@6 at 1.6 / 3.2 / 6.4 s (mean, clip-bootstrap CI, median), CoC degeneration; paired
differences vs dense and vs the matching rank-max (200,000 draws, seed 0, seed 1 re-check), from ci_openloop_lingoqa.
Usage (repo root): .venv/bin/python iclr27-rebuttal/code/combination_rules_test500.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "cvpr27" / "code"))
sys.path.insert(0, str(REPO / "iclr27-rebuttal" / "code"))
from analyze_a100 import HORIZONS, N_BOOT, K, load_rows, metric
from ci_openloop_lingoqa import flips, fmt_ci, pfmt, star, summarise

OUT = REPO / "outputs"
A100 = OUT / "cvpr27" / "a100"
KINDS = ("ade", "fde")

# key, label, run dir, run tag
A100_ARMS = [
    ("dense", "Dense", "dense_test2000", "baseline"),
    ("rank_max", "rank-max (MoOP l0keep)", "moop_l0keep_test2000", "slim_dual_u40_l0keep_v2"),
    ("rank_mean", "rank-mean", "rank_mean_test2000", "slim_dualsum_u40_l0keep_v2"),
    ("raw_mean", "raw-mean", "raw_mean_test2000", "slim_rawabsavg_u40_l0keep_v2"),
    ("raw_max", "raw-max", "raw_max_test2000", "slim_rawabsmean_u40_l0keep_v2"),
    ("fm_only", "FM-only l0keep", "fm_only_l0keep_test2000", "slim_trajlastcoc_u40_l0keep_v2"),
    ("ce_only", "CE-only l0keep", "ce_only_l0keep_test2000", "slim_coc_u40_l0keep_v2"),
]
A100_PAIRS = [(a, r) for a, *_ in A100_ARMS[1:] for r in ("dense", "rank_max") if a != r]
# 2026-11-05 report, test_2000 6.4 s minADE@6
A100_CHECK = {"rank_max": 0.948, "rank_mean": 0.953, "raw_mean": 1.004, "raw_max": 1.000, "fm_only": 0.982,
              "ce_only": 1.033}

ADA_ARMS = [
    ("dense", "Dense", "baseline_pred_test", "baseline"),
    ("rank_max", "rank-max (MoOP l0keep)", "swap_ship_L0fullM_500", "slim_swap_ship_L0fullM"),
    ("raw_max_meanstep", "raw-max, FM = |step mean|", "rawmean_u40_l0keep_v2_test", "slim_rawmean_u40_l0keep_v2"),
    ("lnorm", "layer-normalised max", "lnorm_u40_l0keep_v2_test", "slim_lnorm_u40_l0keep_v2"),
    ("rank_max_ada", "rank-max, Ada importance", "dual_ada_u40_l0keep_v2_test", "slim_dual_ada_u40_l0keep_v2"),
    ("raw_max_ada", "raw-max, Ada importance", "rawabsmean_u40_l0keep_v2_test", "slim_rawabsmean_u40_l0keep_v2"),
]
ADA_PAIRS = [(a, "dense") for a, *_ in ADA_ARMS[1:]] + [
    ("raw_max_meanstep", "rank_max"), ("lnorm", "rank_max"), ("rank_max_ada", "rank_max"),
    ("raw_max_ada", "rank_max_ada")]
# outputs/rawmax_compare/summary.txt, test500 6.4 s minADE@6
ADA_CHECK = {"rank_max": 0.9557, "raw_max_meanstep": 0.9789, "lnorm": 0.9524, "rank_max_ada": 0.9515,
             "raw_max_ada": 0.9672}


def block(rows, ids, pairs):
    """rows {arm: {clip: row}} -> absolute and paired summaries on ids (every arm must have every clip)."""
    for a, r in rows.items():
        missing = [c for c in ids if c not in r]
        assert not missing, f"{a}: {len(missing)} clips missing"
    absolute = {f"{a}|{k}|{h}": [metric(r[c], k, h) for c in ids] for a, r in rows.items() for k in KINDS
                for h in HORIZONS}  # (N,) each
    paired = {f"{a}|{ref}|{k}|{h}": [metric(rows[a][c], k, h) - metric(rows[ref][c], k, h) for c in ids]
              for a, ref in pairs for k in KINDS for h in HORIZONS}  # (N,) each
    coc = {a: float(np.mean([r[c]["coc_degenerate"] for c in ids])) for a, r in rows.items()}
    return {"n": len(ids), "coc_degenerate": coc, "absolute": summarise(absolute), "paired": summarise(paired)}


def md_block(blk, arms, pairs, title):
    A, P = blk["absolute"], blk["paired"]
    lab = {a: lab for a, lab, *_ in arms}
    md = [f"### {title}", "", f"n = {blk['n']} clips; mean [95% clip-bootstrap CI]; median 6.4 s.", "",
          ("| arm | minADE@6 1.6 s | 3.2 s | 6.4 s | minFDE@6 1.6 s | 3.2 s | 6.4 s | median ADE / FDE 6.4 s "
           "| CoC degenerate |"), "|---|---|---|---|---|---|---|---|---|"]
    for a, *_ in arms:
        cells = [fmt_ci(A[f"{a}|{k}|{h}"]) for k in KINDS for h in HORIZONS]
        md.append(f"| {lab[a]} | " + " | ".join(cells) + f" | {A[f'{a}|ade|6.4s']['median']:.3f} / "
                  f"{A[f'{a}|fde|6.4s']['median']:.3f} | {blk['coc_degenerate'][a]:.1%} |")
    md += ["", "Paired arm - reference (mean [95% CI], * = CI excludes 0; median and bootstrap p at 6.4 s):", "",
           ("| arm | vs | dADE 1.6 s | 3.2 s | 6.4 s | dFDE 1.6 s | 3.2 s | 6.4 s | median dADE / dFDE 6.4 s "
            "| p ADE / FDE 6.4 s |"), "|---|---|---|---|---|---|---|---|---|---|"]
    for a, ref in pairs:
        cells = [fmt_ci(P[f"{a}|{ref}|{k}|{h}"], sign=True) + star(P[f"{a}|{ref}|{k}|{h}"]) for k in KINDS
                 for h in HORIZONS]
        ea, ef = P[f"{a}|{ref}|ade|6.4s"], P[f"{a}|{ref}|fde|6.4s"]
        md.append(f"| {lab[a]} | {lab[ref]} | " + " | ".join(cells)
                  + f" | {ea['median']:+.3f} / {ef['median']:+.3f} | {pfmt(ea['p'])} / {pfmt(ef['p'])} |")
    return [*md, ""]


def split_deltas(rows, ids, ids2000):
    """A100 arm - rank-max at 6.4 s on Test500, on the other test_2000 clips, and on all of test_2000."""
    rest = [c for c in ids2000 if c not in set(ids)]
    res, md = {}, ["### A100: arm - rank-max at 6.4 s on Test500 vs the rest of test_2000", "",
                   ("| arm | minADE@6 Test500 (500) | other 1,500 | test_2000 | minFDE@6 Test500 | other 1,500 "
                    "| test_2000 |"), "|---|---|---|---|---|---|---|"]
    for a in ("rank_mean", "raw_mean", "raw_max", "fm_only", "ce_only"):
        cells = {}
        for name, cs in (("test500", ids), ("rest1500", rest), ("test2000", ids2000)):
            series = {f"{k}": [metric(rows[a][c], k) - metric(rows["rank_max"][c], k) for c in cs]
                      for k in KINDS}  # (N,) each
            cells[name] = summarise(series)
        res[a] = cells
        lab = {x: lab for x, lab, *_ in A100_ARMS}[a]
        md.append(f"| {lab} | " + " | ".join(fmt_ci(cells[s][k], sign=True) + star(cells[s][k])
                                            for k in KINDS for s in ("test500", "rest1500", "test2000")) + " |")
    return res, [*md, ""]


def main():
    dst = OUT / "iclr27-rebuttal" / "combination_rules_test500"
    dst.mkdir(parents=True, exist_ok=True)
    ids = list(pd.read_parquet(OUT / "eval_sets" / "test_500.parquet")["clip_id"])
    ids2000 = list(pd.read_parquet(OUT / "eval_sets" / "test_2000.parquet")["clip_id"])
    assert set(ids) <= set(ids2000)

    gpus, check = {}, []
    a100_rows = {}
    for a, _, d, tag in A100_ARMS:
        gpus[f"a100/{d}"] = json.loads((A100 / d / "config.json").read_text()).get("gpu")
        a100_rows[a] = load_rows(A100 / d, tag)
        if a in A100_CHECK:
            m = float(np.mean([metric(a100_rows[a][c], "ade") for c in ids2000]))
            check.append(f"A100 {a:17s} test_2000 minADE@6 6.4 s {m:.4f} (11-05 report {A100_CHECK[a]:.3f}) "
                         f"{'ok' if abs(m - A100_CHECK[a]) <= 5e-4 + 1e-9 else 'DIFF'}")
    ada_rows = {}
    for a, _, d, tag in ADA_ARMS:
        gpus[f"ada/{d}"] = json.loads((OUT / d / "config.json").read_text()).get("gpu")
        ada_rows[a] = load_rows(OUT / d, tag)
        if a in ADA_CHECK:
            m = float(np.mean([metric(ada_rows[a][c], "ade") for c in ids]))
            check.append(f"Ada  {a:17s} test500   minADE@6 6.4 s {m:.4f} (10-15 summary {ADA_CHECK[a]:.4f}) "
                         f"{'ok' if abs(m - ADA_CHECK[a]) <= 5e-5 + 1e-9 else 'DIFF'}")
    assert all("A100" in str(g) for k, g in gpus.items() if k.startswith("a100/")), gpus
    assert all("Ada" in str(g) for k, g in gpus.items() if k.startswith("ada/")), gpus

    a100 = block(a100_rows, ids, A100_PAIRS)
    ada = block(ada_rows, ids, ADA_PAIRS)
    split, split_md = split_deltas(a100_rows, ids, ids2000)
    flipped = {"a100": flips(a100["paired"]), "ada": flips(ada["paired"]),
               "a100_split": [f"{a}/{s}/{k}" for a, v in split.items() for s, e in v.items() for k in flips(e)]}

    md = ["# VLM importance combination rules on Test500", "",
          (f"minADE@6 / minFDE@6 = min over the first {K} of 8 samples, rollout only. Bootstrap {N_BOOT:,} draws, "
           "seed 0, clip-level; paired = per-clip arm minus reference. A100 and Ada are separate tables."), ""]
    md += md_block(a100, A100_ARMS, A100_PAIRS, "A100 (test_500 clips of the test_2000 runs) -- all four rules")
    md += split_md
    md += md_block(ada, ADA_ARMS, ADA_PAIRS, "Ada (2026-10-15 runs) -- rank-max and raw-max variants only")
    md += ["## Checks", "", "```", *check, "```", "",
           "Seed check (paired cells whose 'excludes 0' call flips between seed 0 and seed 1):", ""]
    md += [f"- {k}: {v if v else 'none'}" for k, v in flipped.items()]

    (dst / "config.json").write_text(json.dumps({
        "a100_arms": {a: {"label": lab, "dir": f"cvpr27/a100/{d}", "tag": t} for a, lab, d, t in A100_ARMS},
        "ada_arms": {a: {"label": lab, "dir": d, "tag": t} for a, lab, d, t in ADA_ARMS},
        "a100_pairs": A100_PAIRS, "ada_pairs": ADA_PAIRS, "clips": "test_500 (A100: subset of test_2000)",
        "gpus": gpus, "k": K, "horizons": HORIZONS, "n_boot": N_BOOT, "seeds": [0, 1]}, indent=2))
    (dst / "metrics.json").write_text(json.dumps({"a100": a100, "a100_split_vs_rank_max": split, "ada": ada,
                                                  "checks": check, "seed_flips": flipped}, indent=2))
    (dst / "summary.txt").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
