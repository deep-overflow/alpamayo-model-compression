"""Alpamayo 2 Super on LingoQA (500 val questions, Lingo-Judge): dense vs MoOP / FM-only / CE-only per allocation.

Plan: iclr27-rebuttal/plans/2026-11-13_a2-lingoqa.md. Rows come from run_a2_lingoqa.py + experiments/lingoqa/score_lingo_vqa.py
(scored.json in each run dir). Statistics are the paper's Table 4 ones (iclr27-rebuttal/code/ci_openloop_lingoqa.lingo_block):
accuracy with a 95% CI that resamples the 100 segments (the 5 questions of a segment share a clip), paired deltas vs MoOP and vs
dense, and vs the always-"yes" floor. Allocations whose runs are missing are skipped.

Usage: .venv/bin/python iclr27-rebuttal/code/analyze_a2_lingoqa.py [--exp-id iclr27-rebuttal/a2_lingoqa_analysis]
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "iclr27-rebuttal" / "code"))
from ci_openloop_lingoqa import lingo_block

L = "iclr27-rebuttal/a2_lingo"
ALLOCS = {"u40": "", "u50": "_u50", "u50-q8": "_u50_q8"}  # mask-file suffix per allocation
CRIT = (("MoOP", "moop"), ("FM-only", "fm_only"), ("CE-only", "ce_only"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="iclr27-rebuttal/a2_lingoqa_analysis")
    args = ap.parse_args()
    out = REPO / "outputs" / args.exp_id
    out.mkdir(parents=True, exist_ok=True)
    dense = REPO / "outputs" / f"{L}_dense"
    res, lines = {}, []
    for alloc, sfx in ALLOCS.items():
        dirs = {"dense": dense, **{lab: REPO / "outputs" / f"{L}_{key}{sfx}" for lab, key in CRIT}}
        if not all((d / "scored.json").exists() for d in dirs.values()):
            continue
        blk = lingo_block(dirs, ["MoOP", "dense"])
        res[alloc] = blk
        st = blk["stats"]
        lines += ["", (f"{alloc}: LingoQA accuracy ({blk['n_questions']} questions, {blk['n_segments']} segments; always-yes "
                      f"floor {blk['floor_yes_accuracy'] * 100:.1f}%)"),
                  f"  {'arm':<9}{'acc %':>7}  {'95% CI':<15}{'d vs MoOP pp [CI] p':<34}{'d vs dense pp [CI] p':<34}"]
        for k in dirs:
            a = st[f"{k}|abs"]
            s = f"  {k:<9}{a['mean'] * 100:>7.1f}  [{a['ci'][0] * 100:.1f}, {a['ci'][1] * 100:.1f}]   "
            for ref in ("MoOP", "dense"):
                e = st.get(f"{k}|{ref}")
                s += (f"{e['mean'] * 100:+.1f} [{e['ci'][0] * 100:+.1f}, {e['ci'][1] * 100:+.1f}] p={e['p']:.2g}".ljust(34)
                      if e else "—".ljust(34))
            lines.append(s)
    (out / "config.json").write_text(json.dumps({
        "runs": {a: {"dense": f"{L}_dense", **{lab: f"{L}_{key}{s}" for lab, key in CRIT}} for a, s in ALLOCS.items()},
        "metric": "Lingo-Judge accuracy (LingoQA evaluate.py rule), CIs resample the 100 segments, 200k draws",
        "plan": "iclr27-rebuttal/plans/2026-11-13_a2-lingoqa.md"}, indent=2))
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    text = "\n".join(["Alpamayo 2 Super, LingoQA val (zero-shot, greedy, front camera x 4 frames)", *lines]) + "\n"
    (out / "summary.txt").write_text(text)
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
