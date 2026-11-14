"""Score the LingoQA-train replication runs with Lingo-Judge and test the pre-registered hypotheses.

Plan: iclr27-rebuttal/plans/2026-11-13_a2-lingoqa.md, "Confirmatory replication" (fixed before any train answer was generated).
Scoring is experiments/lingoqa/score_lingo_vqa.py's score_run (LingoJudge imported verbatim, logit > 0) with the references taken
from each run's own rows (train has one human reference per question). Primary family, Holm-corrected, two-sided alpha 0.05:
  H1  MoOP u50 > FM-only u50      H2  MoOP u50 > CE-only u50
paired per question, CIs and p from a bootstrap over the sampled segments (200,000 draws). Secondary rows (dense, u50-q8) are
reported without correction.

Usage: CUDA_VISIBLE_DEVICES=<card> .venv/bin/python iclr27-rebuttal/code/score_a2_lingo_train.py --runs iclr27-rebuttal/a2_lingotrain_<arm> ...
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "lingoqa"))
import score_lingo_vqa as sl  # also puts LingoQA's judge on sys.path

P = "iclr27-rebuttal/a2_lingotrain"
PRIMARY = [("moop_u50", "fm_only_u50"), ("moop_u50", "ce_only_u50")]
SECONDARY = [("moop_u50_q8", "fm_only_u50_q8"), ("moop_u50_q8", "ce_only_u50_q8"), ("moop_u50", "dense"),
             ("fm_only_u50", "dense"), ("ce_only_u50", "dense")]


def seg_boot(d, seg, n=200_000, seed=0, chunk=10_000):
    """mean of d with whole segments resampled -> (lo, hi, two-sided p of mean = 0)"""
    keys, inv = np.unique(seg, return_inverse=True)
    S = np.bincount(inv, weights=d, minlength=len(keys))  # (G,)
    C = np.bincount(inv, minlength=len(keys)).astype(float)
    rng = np.random.default_rng(seed)
    ms = []
    for s in range(0, n, chunk):
        idx = rng.integers(0, len(keys), (min(chunk, n - s), len(keys)))
        ms.append(S[idx].sum(1) / C[idx].sum(1))
    m = np.concatenate(ms)
    p = min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean()))
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float(p)


def holm(ps):
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--exp-id", default="iclr27-rebuttal/a2_lingotrain_analysis")
    ap.add_argument("--prefix", default=P, help="run-dir prefix (a2_lingotrain2 for the q8 confirmatory sample)")
    ap.add_argument("--primary", nargs="+", default=None, help="a:b pairs for the Holm family (default: the u50 pair)")
    args = ap.parse_args()
    primary = [tuple(x.split(":")) for x in args.primary] if args.primary else PRIMARY
    secondary = [x for x in SECONDARY if x not in primary]
    out = REPO / "outputs" / args.exp_id
    out.mkdir(parents=True, exist_ok=True)

    device = sl.ll.pick_gpu(2.0, [0])
    judge = sl.LingoJudge().eval().to(device)
    acc, rows = {}, {}
    for exp in args.runs:
        d = REPO / "outputs" / exp
        gen = json.loads((d / "rows.json").read_text())
        ref = pd.DataFrame([{"question_id": g["question_id"], "segment_id": g["segment_id"], "question": g["question"],
                             "references": g["references"]} for g in gen])
        m = sl.score_run(judge, exp, ref, device)
        arm = exp.split(f"{Path(args.prefix).name}_", 1)[1]
        acc[arm] = m
        rows[arm] = {(r["question_id"], r["segment_id"]): r for r in json.loads((d / "scored.json").read_text())}
        print(f"{arm:<16} n {m['n_matched']}  acc {m['accuracy'] * 100:.1f}%  words {m['answer_words_median']:.0f}", flush=True)

    def pair(a, b):
        if a not in rows or b not in rows:
            return None
        keys = sorted(set(rows[a]) & set(rows[b]))
        d = np.array([float(rows[a][k]["correct"]) - float(rows[b][k]["correct"]) for k in keys])
        seg = np.array([k[1] for k in keys])
        lo, hi, p = seg_boot(d, seg)
        return {"n": len(keys), "n_segments": len(set(seg)), "mean": float(d.mean()), "ci": [lo, hi], "p": p}

    prim = {f"{a} - {b}": pair(a, b) for a, b in primary}
    ok = [k for k, v in prim.items() if v]
    for k, padj in zip(ok, holm([prim[k]["p"] for k in ok]), strict=True):
        prim[k]["p_holm"] = padj
    sec = {f"{a} - {b}": pair(a, b) for a, b in secondary}
    (out / "config.json").write_text(json.dumps({
        "runs": args.runs, "primary": primary, "secondary": secondary, "correction": "Holm over the primary family",
        "bootstrap": "segments resampled, 200,000 draws, seed 0", "plan": "iclr27-rebuttal/plans/2026-11-13_a2-lingoqa.md"},
        indent=2))
    (out / "metrics.json").write_text(json.dumps({"accuracy": acc, "primary": prim, "secondary": sec}, indent=2))
    L = ["LingoQA train sample (confirmatory replication), Lingo-Judge accuracy", ""]
    L += [f"  {a:<16}{v['accuracy'] * 100:6.1f}%  (n {v['n_matched']}, median words {v['answer_words_median']:.0f})"
          for a, v in acc.items()]
    L += ["", "primary (Holm):"]
    for k, v in prim.items():
        if v:
            L.append(f"  {k:<30} {v['mean'] * 100:+.1f} pp [{v['ci'][0] * 100:+.1f}, {v['ci'][1] * 100:+.1f}]  "
                     f"p {v['p']:.2g}  p_holm {v['p_holm']:.2g}  ({v['n']} q, {v['n_segments']} segments)")
    L += ["", "secondary (uncorrected):"]
    for k, v in sec.items():
        if v:
            L.append(f"  {k:<30} {v['mean'] * 100:+.1f} pp [{v['ci'][0] * 100:+.1f}, {v['ci'][1] * 100:+.1f}]  p {v['p']:.2g}")
    (out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
