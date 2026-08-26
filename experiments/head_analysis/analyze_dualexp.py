"""Gates G1-G4 of plans/2026-08-26_dual-plus-znorm.md: is the expert znorm cut free on top
of dual_u40_v2?

Four runs of the fixed protocol are joined per clip: baseline (unpruned), dual (VLM-only
dual_u40_v2), expert (expert-only znorm r25), combined (dualexp_u40_e25). All Ada, K=8,
clip-derived seeds, so every contrast is paired on the same noise draw.

Pre-registered gates (restated from the plan):
  G1 CoC identity (sanity): combined shares the dual arm's VLM weights and seeds, so
     per-clip gen_coc / nll_self must match dual_u40_v2_ps_indist exactly on Ada; any
     mismatch means protocol drift, investigate before reading trajectories.
     Fallback: combined coc_degenerate rate <= dual + 1pp.
  G2 primary: paired dADE = minADE@6(combined) - minADE@6(dual), 95% bootstrap CI [lo,hi],
     threshold +0.013 = dual's VLM cost 0.0668 x param share 532/2657.
     ADOPT (free) if hi < +0.013; REJECT if lo >= +0.013; else INCONCLUSIVE.
  G3 additivity (secondary): I_c = (combined-dual) - (expert-baseline) per clip, mean CI,
     expected 0; flag super-additive if lo > +0.01. Also reports expert-baseline, the
     first fixed-protocol number for znorm r25 (prior +0.0003 on 200 clips).
  G4 vs baseline (secondary, no gate): combined-baseline, expected ~ +0.067.

Usage:
  python experiments/head_analysis/analyze_dualexp.py \
      --combined dualexp_u40_e25_ps_indist --dual dual_u40_v2_ps_indist \
      --expert expert_znorm_r25_ps_indist --baseline baseline_ada_ps_indist \
      --out dualexp_arms_val
"""

import argparse
import json

from analyze_stepvlm_arms import REPO, boot_ci, load_rows, reduce_at

G2_THRESH = 0.013  # proportional-cost prediction: 0.0668 * 532/2657
G3_FLAG = 0.01

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", default="dualexp_u40_e25_ps_indist")
    ap.add_argument("--dual", default="dual_u40_v2_ps_indist")
    ap.add_argument("--expert", default="expert_znorm_r25_ps_indist")
    ap.add_argument("--baseline", default="baseline_ada_ps_indist")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out", default="dualexp_arms_val")
    args = ap.parse_args()

    runs = {n: load_rows(getattr(args, n)) for n in ("combined", "dual", "expert", "baseline")}
    ids = sorted(set.intersection(*(set(r) for r in runs.values())))
    if not ids:
        raise SystemExit("no shared clips across the four runs")
    mism = [c for c in ids
            if len({runs[n][c]["seed"] for n in runs}) != 1]
    if mism:
        raise SystemExit(f"{len(mism)} clips have different seeds; pairing is invalid")

    ade = {n: reduce_at(runs[n], ids, "ade_rollout_k", args.k) for n in runs}
    fde = {n: reduce_at(runs[n], ids, "fde_rollout_k", args.k) for n in runs}
    res = {"runs": {n: getattr(args, n) for n in runs}, "n_paired": len(ids), "k": args.k,
           "minADE_mean": {n: float(a.mean()) for n, a in ade.items()},
           "minFDE_mean": {n: float(a.mean()) for n, a in fde.items()}}
    lines = [f"dual + znorm gates -- paired on {len(ids)} clips, minADE@{args.k}",
             "  " + "  ".join(f"{n} {a.mean():.4f}" for n, a in ade.items()), ""]

    # G1 -- CoC identity vs the dual arm
    coc_diff = [c for c in ids if runs["combined"][c]["gen_coc"] != runs["dual"][c]["gen_coc"]]
    nll_diff = [c for c in ids
                if runs["combined"][c]["nll_self"] != runs["dual"][c]["nll_self"]]
    degen = {n: float(sum(runs[n][c]["coc_degenerate"] for c in ids)) / len(ids) for n in runs}
    g1 = "PASS" if not coc_diff and not nll_diff else (
        "PASS (fallback)" if degen["combined"] <= degen["dual"] + 0.01 else "FAIL")
    res["G1"] = {"coc_text_mismatch": len(coc_diff), "nll_self_mismatch": len(nll_diff),
                 "degenerate_rate": degen, "verdict": g1}
    lines.append(f"  G1 CoC identity: gen_coc mismatch {len(coc_diff)}/{len(ids)}, "
                 f"nll_self mismatch {len(nll_diff)}/{len(ids)}")
    lines.append("     degen rate  " + "  ".join(f"{n} {v:.4f}" for n, v in degen.items()))
    lines.append(f"     -> {g1}")

    # G2 -- primary: combined - dual
    d = ade["combined"] - ade["dual"]
    m, lo, hi = boot_ci(d)
    g2 = ("ADOPT" if hi < G2_THRESH else
          ("REJECT" if lo >= G2_THRESH else "INCONCLUSIVE"))
    res["G2"] = {"delta_mean": m, "lo": lo, "hi": hi, "threshold": G2_THRESH, "verdict": g2}
    lines.append(f"  G2 combined - dual: {m:+.4f} [{lo:+.4f},{hi:+.4f}]  "
                 f"threshold +{G2_THRESH}  -> {g2}")

    # G3 -- additivity: (combined - dual) - (expert - baseline)
    e = ade["expert"] - ade["baseline"]
    em_, elo, ehi = boot_ci(e)
    im, ilo, ihi = boot_ci(d - e)
    g3 = "SUPER-ADDITIVE" if ilo > G3_FLAG else "OK"
    res["G3"] = {"expert_minus_baseline": {"mean": em_, "lo": elo, "hi": ehi},
                 "interaction": {"mean": im, "lo": ilo, "hi": ihi}, "verdict": g3}
    lines.append(f"  G3 expert - baseline: {em_:+.4f} [{elo:+.4f},{ehi:+.4f}]  "
                 f"(prior +0.0003 on 200 clips)")
    lines.append(f"     interaction I_c:   {im:+.4f} [{ilo:+.4f},{ihi:+.4f}]  -> {g3}")

    # G4 -- descriptive: combined - baseline
    b = ade["combined"] - ade["baseline"]
    bm, blo, bhi = boot_ci(b)
    res["G4"] = {"delta_mean": bm, "lo": blo, "hi": bhi}
    lines.append(f"  G4 combined - baseline: {bm:+.4f} [{blo:+.4f},{bhi:+.4f}]  (no gate)")

    fd = fde["combined"] - fde["dual"]
    fm, flo, fhi = boot_ci(fd)
    res["minFDE_combined_minus_dual"] = {"mean": fm, "lo": flo, "hi": fhi}
    lines.append(f"  side minFDE@{args.k} combined - dual: {fm:+.4f} [{flo:+.4f},{fhi:+.4f}]")

    out_dir = REPO / "outputs" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    print("\n".join(lines))
    print("saved ->", out_dir)


if __name__ == "__main__":
    main()
