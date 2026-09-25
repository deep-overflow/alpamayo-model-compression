"""Gates of plans/2026-09-25_depth-aware-dual.md, step 1: the dual criterion's trunk (layers
0-21) with a different late-layer (22-35) selection, read on held-out clips with the dense
model's text and noise (run_token_ablation.py --arm-masks), paired per clip:

  dual                       the shipped union arm (reference)
  dual_trunk__coc_late       candidate A: union below layer 22, I_CE alone from layer 22 on
  dual_trunk__traj_late      mirror control: I_FM alone from layer 22 on
  dual_trunk__rand_late      random late units, same count per layer
  dual_trunk                 union trunk with every late layer intact (upper bound for the late band)

  G1 (action unchanged)   FM loss and minADE of A are not worse than dual's: paired one-sided
                          Wilcoxon (A > dual) p >= 0.05 AND the mean difference's 95% CI lies
                          below +1% of the dense FM loss (+0.02 m for minADE)
  G2 (language better)    CoC NLL of A is below dual's: paired one-sided Wilcoxon p < 0.01,
                          and below the random-late control's
  G3 (mirror)             dual_trunk__traj_late raises the NLL above dual (p < 0.01) -- the
                          late I_FM term is what the union pays for in language

Usage:
  python experiments/head_analysis/analyze_depth_mix.py --shards depthmix_v1_s0 ... --out depthmix_v1
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
READ = (("fm", "FM loss"), ("nll", "CoC NLL"), ("ade", "minADE"))
SHOW = ["dual", "dual_trunk", "dual_trunk__coc_late", "dual_trunk__traj_late", "dual_trunk__rand_late"]
rng = np.random.default_rng(0)


def merge(shards):
    clips, buckets, per = [], [], {}
    for s in shards:
        m = json.loads((REPO / "outputs" / s / "metrics.json").read_text())
        clips += m["clip_ids"]
        buckets += m["buckets"]
        for cfg, v in m["per_clip"].items():
            for k, arr in v.items():
                per.setdefault(cfg, {}).setdefault(k, []).extend(arr)
    per = {("dense" if c == "dense" else c.replace("arm_", "")): {k: np.array(a, float) for k, a in v.items()}
           for c, v in per.items()}
    return clips, np.array(buckets), per


def ci(x, n=5000):
    x = np.asarray(x, float)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def wil(a, b, alt):
    d = np.asarray(a) - np.asarray(b)
    d = d[np.isfinite(d)]
    return float(wilcoxon(d, alternative=alt)[1]) if len(d) >= 5 and (d != 0).any() else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = REPO / "outputs" / args.out
    out.mkdir(parents=True, exist_ok=True)
    clips, buckets, per = merge(args.shards)
    dense = per["dense"]
    base = {k: float(np.nanmean(dense[k])) for k, _ in READ}
    d = {c: {k: per[c][k] - dense[k] for k, _ in READ} for c in SHOW if c in per}
    lines = [f"depth-aware dual, step 1 -- {len(clips)} held-out clips, dense text; shards {', '.join(args.shards)}",
             f"dense: FM {base['fm']:.4f}  NLL {base['nll']:.4f}  minADE {base['ade']:.3f}", "",
             f"  {'mask':24s} {'dFM [95% CI]':>28s} {'rel':>7s} {'dNLL [95% CI]':>28s} {'rel':>7s} {'dminADE [95% CI]':>26s} {'rel':>7s}"]
    res = {}
    for c in SHOW:
        if c not in d:
            continue
        cells, res[c] = [], {}
        for k, _ in READ:
            m = ci(d[c][k])
            res[c][k] = {"mean": m[0], "lo": m[1], "hi": m[2], "rel": m[0] / base[k]}
            cells.append(f"{m[0]:+.4f} [{m[1]:+.4f},{m[2]:+.4f}] {m[0] / base[k]:+6.1%}")
        lines.append(f"  {c:24s} " + " ".join(cells))
    # by manoeuvre, relative FM and NLL
    lines.append("\n  relative dFM / dNLL by manoeuvre (cruise / accel / decel_stop / turn):")
    for c in SHOW:
        if c not in d:
            continue
        cells = []
        for b in ("cruise", "accel", "decel_stop", "turn"):
            m = buckets == b
            cells.append(f"{np.nanmean(d[c]['fm'][m]) / np.nanmean(dense['fm'][m]):+.1%}/{np.nanmean(d[c]['nll'][m]) / np.nanmean(dense['nll'][m]):+.1%}")
        lines.append(f"  {c:24s} " + "  ".join(cells))
    gates = {}
    A, ref, rnd, mir = "dual_trunk__coc_late", "dual", "dual_trunk__rand_late", "dual_trunk__traj_late"
    if A in d and ref in d:
        p_fm = wil(d[A]["fm"], d[ref]["fm"], "greater"); p_ade = wil(d[A]["ade"], d[ref]["ade"], "greater")
        dfm = ci(d[A]["fm"] - d[ref]["fm"]); dade = ci(d[A]["ade"] - d[ref]["ade"])
        g1 = p_fm >= 0.05 and p_ade >= 0.05 and dfm[2] < 0.01 * base["fm"] and dade[2] < 0.02
        p_nll = wil(d[A]["nll"], d[ref]["nll"], "less")
        p_nll_r = wil(d[A]["nll"], d[rnd]["nll"], "less") if rnd in d else 1.0
        dnll = ci(d[A]["nll"] - d[ref]["nll"])
        g2 = p_nll < 0.01 and p_nll_r < 0.01
        p_mir = wil(d[mir]["nll"], d[ref]["nll"], "greater") if mir in d else 1.0
        g3 = p_mir < 0.01
        gates = {"G1": g1, "G1_fm_p": p_fm, "G1_ade_p": p_ade, "A_minus_dual_fm": dfm, "A_minus_dual_ade": dade,
                 "G2": g2, "G2_nll_vs_dual_p": p_nll, "G2_nll_vs_random_p": p_nll_r, "A_minus_dual_nll": dnll,
                 "G3": g3, "G3_mirror_nll_p": p_mir}
        lines += ["",
                  (f"  G1 action unchanged: {'PASS' if g1 else 'FAIL'}  (A - dual: FM {dfm[0]:+.4f} [{dfm[1]:+.4f},{dfm[2]:+.4f}] "
                   f"= {dfm[0] / base['fm']:+.1%} of dense, p(A worse)={p_fm:.2g}; minADE {dade[0]:+.3f} [{dade[1]:+.3f},{dade[2]:+.3f}], p={p_ade:.2g})"),
                  (f"  G2 language better:   {'PASS' if g2 else 'FAIL'}  (A - dual: NLL {dnll[0]:+.4f} [{dnll[1]:+.4f},{dnll[2]:+.4f}] "
                   f"= {dnll[0] / base['nll']:+.1%} of dense, p(A better)={p_nll:.2g}; vs random late p={p_nll_r:.2g})"),
                  f"  G3 mirror:            {'PASS' if g3 else 'FAIL'}  (traj late raises NLL above dual, p={p_mir:.2g})"]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "metrics.json").write_text(json.dumps({"n_clips": len(clips), "dense": base, "delta": res, "gates": gates}, indent=1, default=float))
    (out / "config.json").write_text(json.dumps({"shards": args.shards, "plan": "plans/2026-09-25_depth-aware-dual.md"}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
