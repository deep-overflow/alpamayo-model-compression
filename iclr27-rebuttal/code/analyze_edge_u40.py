"""Alpamayo 2 Edge at u40: dense / MoOP / FM-only / CE-only on Test500, OOD-Val250 (open loop) and LingoQA (plans/2026-11-14_a2-edge-moop-u40.md).

Readouts, with the machinery of ci_openloop_lingoqa.py (200,000 bootstrap draws, seed 0, shared draws across arms; a seed-1 rerun
of every CI is stored too):
  open loop  minADE@6 / minFDE@6 (first 6 of 8 samples, analyze_a100.metric) at 1.6 / 3.2 / 6.4 s: mean with a clip CI, median;
             CoC degeneration, <NOMINAL> rate, CoCs without the anchor, generated length; paired arm - dense, MoOP - FM-only,
             MoOP - CE-only, FM-only - CE-only per clip (mean, 95% CI, two-sided bootstrap p)
  LingoQA    Lingo-Judge accuracy over the 500 val questions (scored.json from score_lingo_vqa.py); CIs resample the 100
             segments; the same paired deltas, and each arm vs the always-"yes" floor
--partial uses the clips / questions every arm has so far (progress check only).

Usage: .venv/bin/python iclr27-rebuttal/code/analyze_edge_u40.py [--partial]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO / "cvpr27" / "code"))
from analyze_a100 import HORIZONS, load_rows, metric
from ci_openloop_lingoqa import FLOOR, lingo_rows, summarise

OUT = REPO / "outputs" / "iclr27-rebuttal"
ARMS = ("dense", "moop", "fm_only", "ce_only")
LABEL = {"dense": "Dense", "moop": "MoOP", "fm_only": "FM-only", "ce_only": "CE-only"}
SETS = {"test500": REPO / "outputs" / "eval_sets" / "test_500.parquet", "oodval": OUT / "edge_moop" / "ood_val_250.parquet"}
PAIRS = [("moop", "dense"), ("fm_only", "dense"), ("ce_only", "dense"), ("moop", "fm_only"), ("moop", "ce_only"),
         ("fm_only", "ce_only")]
KINDS = ("ade", "fde")
# --preset l0keep (plans/2026-11-14_a2-edge-l0keep-u40.md): Test500 only, the three _l0keep arms next to dense and the plain arms
L0_ARMS = ARMS + ("moop_l0keep", "fm_only_l0keep", "ce_only_l0keep")
L0_LABEL = {**LABEL, "moop_l0keep": "MoOP-l0k", "fm_only_l0keep": "FM-l0k", "ce_only_l0keep": "CE-l0k"}
L0_PAIRS = ([(f"{a}_l0keep", a) for a in ("moop", "fm_only", "ce_only")]
            + [(f"{a}_l0keep", "dense") for a in ("moop", "fm_only", "ce_only")]
            + [("moop_l0keep", "fm_only_l0keep"), ("moop_l0keep", "ce_only_l0keep"), ("fm_only_l0keep", "ce_only_l0keep")])
# --preset qcut (plans/2026-11-15_a2-edge-qcut-u40.md): q2 / q4 arms (fewer Q heads, more MLP channels, same budget), all sets
QC_ARMS = ARMS + tuple(f"{a}_q{n}" for n in (2, 4) for a in ("moop", "fm_only", "ce_only"))
QC_LABEL = {**LABEL, **{f"{a}_q{n}": f"{LABEL[a]}-q{n}" for n in (2, 4) for a in ("moop", "fm_only", "ce_only")}}
QC_PAIRS = [pair for n in (2, 4) for pair in
            [(f"{a}_q{n}", a) for a in ("moop", "fm_only", "ce_only")]
            + [(f"{a}_q{n}", "dense") for a in ("moop", "fm_only", "ce_only")]
            + [(f"moop_q{n}", f"fm_only_q{n}"), (f"moop_q{n}", f"ce_only_q{n}"), (f"fm_only_q{n}", f"ce_only_q{n}")]]


def open_loop(set_name, partial):
    ids = list(pd.read_parquet(SETS[set_name])["clip_id"])
    runs = {a: load_rows(OUT / f"edge_{set_name}_{a}", "edge") for a in ARMS}
    if partial:
        ids = [c for c in ids if all(c in runs[a] for a in ARMS)]
    else:
        for a in ARMS:
            missing = [c for c in ids if c not in runs[a]]
            assert not missing, f"{set_name}/{a}: {len(missing)} clips missing"
    absolute, paired = {}, {}
    for kind in KINDS:
        for h in HORIZONS:
            for a in ARMS:
                absolute[f"{a}|{kind}|{h}"] = [metric(runs[a][c], kind, h) for c in ids]
            for a, b in PAIRS:
                paired[f"{a}|{b}|{kind}|{h}"] = [metric(runs[a][c], kind, h) - metric(runs[b][c], kind, h) for c in ids]
    coc = {a: {"degenerate": float(np.mean([runs[a][c]["coc_degenerate"] for c in ids])),
               "nominal": float(np.mean([runs[a][c]["nominal"] for c in ids])),
               "no_anchor": int(sum(not runs[a][c]["has_eos"] for c in ids)),
               "gen_len_mean": float(np.mean([runs[a][c]["gen_len"] for c in ids]))} for a in ARMS}
    buckets = pd.Series([runs["dense"][c]["bucket"] for c in ids]).value_counts().to_dict()
    by_bucket = {}
    for bk in buckets:
        sel = [c for c in ids if runs["dense"][c]["bucket"] == bk]
        by_bucket[bk] = {a: float(np.mean([metric(runs[a][c], "ade", "6.4s") for c in sel])) for a in ARMS}
    return {"n": len(ids), "absolute": summarise(absolute), "paired": summarise(paired), "coc": coc,
            "buckets": {k: int(v) for k, v in buckets.items()}, "ade6_by_bucket_mean": by_bucket}


def lingo(partial):
    rows = {a: lingo_rows(OUT / f"edge_lingo_{a}") for a in ARMS}
    floor = lingo_rows(REPO / "outputs" / FLOOR)
    qs = sorted(set.intersection(*(set(r) for r in rows.values())) & set(floor))
    if not partial:
        assert len(qs) == 500, f"LingoQA: {len(qs)} common questions"
    seg = [floor[q]["segment_id"] for q in qs]
    acc = {a: [float(rows[a][q]["correct"]) for q in qs] for a in ARMS}
    yes = [float(floor[q]["correct"]) for q in qs]
    series = {f"{a}|abs": acc[a] for a in ARMS}
    for a, b in PAIRS:
        series[f"{a}|{b}"] = list(np.subtract(acc[a], acc[b]))
    for a in ARMS:
        series[f"{a}|floor_yes"] = list(np.subtract(acc[a], yes))
    def text(r):  # an empty answer comes back from predictions.csv as NaN; it is scored as given (judged wrong)
        p = r.get("prediction")
        return p if isinstance(p, str) else ""

    words = {a: float(np.median([len(text(rows[a][q]).split()) for q in qs])) for a in ARMS}
    empty = {a: int(sum(not text(rows[a][q]).strip() for q in qs)) for a in ARMS}
    return {"n_questions": len(qs), "n_segments": len(set(seg)), "floor_yes_accuracy": float(np.mean(yes)),
            "answer_words_median": words, "empty_answers": empty, "stats": summarise(series, groups=seg)}


def ci(e, scale=1.0, nd=3):
    lo, hi = e["ci"]
    s = "*" if (lo > 0 or hi < 0) else ""
    return f"{e['mean'] * scale:+.{nd}f} [{lo * scale:+.{nd}f}, {hi * scale:+.{nd}f}]{s} p={e['p']:.3g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--exp-id", default="iclr27-rebuttal/edge_u40_analysis")
    ap.add_argument("--skip-lingo", action="store_true")
    ap.add_argument("--preset", default="u40", choices=["u40", "l0keep", "qcut"])
    args = ap.parse_args()
    global ARMS, LABEL, PAIRS, SETS
    if args.preset == "l0keep":
        ARMS, LABEL, PAIRS = L0_ARMS, L0_LABEL, L0_PAIRS
        SETS = {"test500": SETS["test500"]}
        args.skip_lingo = True
    elif args.preset == "qcut":
        ARMS, LABEL, PAIRS = QC_ARMS, QC_LABEL, QC_PAIRS
    if args.preset != "u40" and args.exp_id == "iclr27-rebuttal/edge_u40_analysis":
        args.exp_id = f"iclr27-rebuttal/edge_u40_{args.preset}_analysis"
    out = REPO / "outputs" / args.exp_id
    out.mkdir(parents=True, exist_ok=True)
    res = {s: open_loop(s, args.partial) for s in SETS}
    lq = None if args.skip_lingo else lingo(args.partial)

    title = (f"Alpamayo 2 Edge u40{'' if args.preset == 'u40' else ' ' + args.preset} (Ada RTX 5880, cards 4-7)"
             f"{' -- PARTIAL' if args.partial else ''}")
    L = [title, ""]
    for s, r in res.items():
        L += [f"== {s} ({r['n']} clips) open loop, minADE@6 / minFDE@6 [m]: mean (median)",
              f"{'arm':<9}" + "".join(f"{f'{k}@6 {h}':>20}" for k in ("ADE", "FDE") for h in HORIZONS)
              + f"{'degen':>7}{'nominal':>9}{'no-anchor':>10}{'len':>6}"]
        for a in ARMS:
            ab = r["absolute"]
            L.append(f"{LABEL[a]:<9}" + "".join(
                f"{ab[f'{a}|{k}|{h}']['mean']:>12.3f} ({ab[f'{a}|{k}|{h}']['median']:.3f})" for k in KINDS for h in HORIZONS)
                + f"{r['coc'][a]['degenerate']:>7.3f}{r['coc'][a]['nominal']:>9.2f}{r['coc'][a]['no_anchor']:>10d}"
                f"{r['coc'][a]['gen_len_mean']:>6.1f}")
        L.append("paired deltas (mean [95% CI], * = CI excludes 0):")
        for a, b in PAIRS:
            L.append(f"  {LABEL[a]} - {LABEL[b]}:")
            for k in KINDS:
                L.append("    " + k.upper() + "@6  " + " | ".join(
                    f"{h} {ci(r['paired'][f'{a}|{b}|{k}|{h}'])}" for h in HORIZONS))
        L += [f"  buckets {r['buckets']}", "  minADE@6 6.4 s by bucket: "
              + json.dumps({k: {a: round(v, 3) for a, v in d.items()} for k, d in r["ade6_by_bucket_mean"].items()}), ""]
    if lq:
        st = lq["stats"]
        L += [(f"== LingoQA ({lq['n_questions']} questions, {lq['n_segments']} segments; always-yes floor "
               f"{100 * lq['floor_yes_accuracy']:.1f}%)")]
        for a in ARMS:
            e = st[f"{a}|abs"]
            L.append(f"{LABEL[a]:<9}{100 * e['mean']:6.1f}%  [{100 * e['ci'][0]:.1f}, {100 * e['ci'][1]:.1f}]  "
                     f"median words {lq['answer_words_median'][a]:.0f}  vs floor {ci(st[f'{a}|floor_yes'], 100, 1)} pp")
        for a, b in PAIRS:
            L.append(f"  {LABEL[a]} - {LABEL[b]}: {ci(st[f'{a}|{b}'], 100, 1)} pp")
    (out / "metrics.json").write_text(json.dumps({"partial": args.partial, "open_loop": res, "lingoqa": lq}, indent=2))
    (out / "config.json").write_text(json.dumps({
        "plan": "iclr27-rebuttal/plans/2026-11-14_a2-edge-moop-u40.md", "arms": ARMS, "sets": {k: str(v) for k, v in SETS.items()},
        "metric": "analyze_a100.metric (min over the first 6 of 8 samples)", "bootstrap": "ci_openloop_lingoqa.summarise",
        "lingo_floor": FLOOR}, indent=2))
    (out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
