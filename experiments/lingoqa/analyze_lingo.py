"""Stage 3: paired comparison of arms on the LingoQA reasoning probe.

Statistics are clustered by segment. One CoC serves the 5 questions of its segment, so
a degenerate CoC breaks 5 answers at once: the 500 questions are not 500 independent
trials, they are 100 clusters of 5. Resampling questions would understate the variance.

  primary    paired bootstrap resampling *segments* -- the cluster analogue of
             eval_lib.paired_bootstrap_ci, which resamples independent units.
  secondary  McNemar over the per-question pairs, reported with the note that it is
             anti-conservative here because it ignores the clustering.

Pre-registered gates (see plans/, stated before the runs):
  - the probe is informative iff `coc` beats `blind` with a cluster CI excluding 0.
    If it does not, the CoC carries no question-relevant signal and no arm comparison
    is reported.
  - an arm retains reasoning iff its paired delta vs the comparison arm has a 95%
    cluster CI containing 0 (equivalence, not merely p > 0.05).

Usage:
  python experiments/lingoqa/analyze_lingo.py --runs lingo_pilot_blind lingo_judge_...
  python experiments/lingoqa/analyze_lingo.py --runs ... --baseline lingo_pilot_blind
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(Path(__file__).parent))

import lingo_lib as ll


def load_run(exp_id):
    """Per-question verdicts, from either track.

    The probe's Stage 2 writes `rows.json` already carrying `correct`; the
    standard-protocol VQA runner writes generations to `rows.json` and the judge's
    verdicts to `scored.json`. Preferring `scored.json` lets both tracks share this
    analysis, and the cluster structure is the same either way -- 5 questions per
    segment, so segments are still the resampling unit.
    """
    d = REPO / "outputs" / exp_id
    scored = d / "scored.json"
    rows = json.loads((scored if scored.exists() else d / "rows.json").read_text())
    cfg = json.loads((d / "config.json").read_text())
    return {r["question_id"] + "|" + r["segment_id"]: r for r in rows}, cfg


def cluster_bootstrap(delta_by_seg, n_boot=10000, seed=0, alpha=0.05):
    """CI for the mean paired delta, resampling segments (clusters) with replacement.

    delta_by_seg: {segment_id: array of per-question deltas within that segment}
    The statistic is the question-weighted mean, so a resample that draws a segment
    twice counts all of its questions twice -- which is the point.
    """
    segs = list(delta_by_seg)
    per = [np.asarray(delta_by_seg[s], dtype=float) for s in segs]
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(segs), size=(n_boot, len(segs)))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = [per[i] for i in idx[b]]
        boots[b] = np.concatenate(pick).mean()
    obs = np.concatenate(per).mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(obs), float(lo), float(hi)


def mcnemar(a_correct, b_correct):
    """Exact McNemar on paired binary outcomes. Anti-conservative under clustering."""
    from scipy.stats import binomtest
    b = int(np.sum(a_correct & ~b_correct))   # a right, b wrong
    c = int(np.sum(~a_correct & b_correct))
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(b, b + c, 0.5).pvalue)


def compare(name_a, run_a, name_b, run_b, n_boot, seed):
    """Paired comparison of two runs over their shared questions."""
    keys = sorted(set(run_a) & set(run_b))
    a = np.array([run_a[k]["correct"] for k in keys], dtype=bool)
    b = np.array([run_b[k]["correct"] for k in keys], dtype=bool)
    by_seg = {}
    for k, d in zip(keys, a.astype(float) - b.astype(float)):
        by_seg.setdefault(run_a[k]["segment_id"], []).append(d)
    obs, lo, hi = cluster_bootstrap(by_seg, n_boot=n_boot, seed=seed)
    nb, nc, p = mcnemar(a, b)
    return {"a": name_a, "b": name_b, "n_questions": len(keys), "n_segments": len(by_seg),
            "acc_a": float(a.mean()), "acc_b": float(b.mean()),
            "delta": obs, "ci_lo": lo, "ci_hi": hi, "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "mcnemar_b": nb, "mcnemar_c": nc, "mcnemar_p": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="exp_ids of Stage 2 runs")
    ap.add_argument("--baseline", default=None,
                    help="exp_id every other run is compared against (default: first)")
    ap.add_argument("--exp-id", default="lingo_analysis")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    runs, cfgs = {}, {}
    for r in args.runs:
        runs[r], cfgs[r] = load_run(r)
    base = args.baseline or args.runs[0]

    lines = ["LingoQA reasoning probe -- paired analysis",
             f"clustered by segment; {args.n_boot} bootstrap resamples over segments", ""]
    lines.append(f"{'run':<34}{'condition':<10}{'n_q':>6}{'n_seg':>7}{'acc':>9}")
    per_run = {}
    for r in args.runs:
        rows = runs[r]
        acc = float(np.mean([v["correct"] for v in rows.values()]))
        nseg = len({v["segment_id"] for v in rows.values()})
        per_run[r] = {"accuracy": acc, "n_questions": len(rows), "n_segments": nseg,
                      "condition": cfgs[r].get("condition") or cfgs[r].get("style")}
        lines.append(f"{r:<34}{cfgs[r].get('condition') or cfgs[r].get('style') or '-'!s:<12}{len(rows):>6}{nseg:>7}"
                     f"{acc * 100:>8.1f}%")

    lines += ["", f"paired deltas vs {base} (positive = run is better)", ""]
    lines.append(f"{'run':<34}{'delta':>9}{'95% CI':>20}{'McNemar p':>12}{'b/c':>10}")
    comps = []
    for r in args.runs:
        if r == base:
            continue
        c = compare(r, runs[r], base, runs[base], args.n_boot, args.seed)
        comps.append(c)
        ci = "[{:+.2f}, {:+.2f}]".format(c["ci_lo"] * 100, c["ci_hi"] * 100)
        bc = "{}/{}".format(c["mcnemar_b"], c["mcnemar_c"])
        lines.append(f"{r:<34}{c['delta'] * 100:>+8.2f}pp{ci:>20}"
                     f"{c['mcnemar_p']:>12.4f}{bc:>10}")

    summary = "\n".join(lines) + "\n"
    out_dir = REPO / "outputs" / args.exp_id
    ll.write_outputs(out_dir, {"runs": args.runs, "baseline": base, "n_boot": args.n_boot,
                               "seed": args.seed,
                               "statistic": "question-weighted mean, segments resampled"},
                     {"per_run": per_run, "comparisons": comps}, summary)
    print(summary)


if __name__ == "__main__":
    main()
