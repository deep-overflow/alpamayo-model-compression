"""Gates A2-heldout and A2-bands of plans/2026-09-21_importance-causal-validation.md.

A2 read the three shipped arms on calib_100, where every criterion is in-sample for its own
loss. This reads the same masks on held-out clips (run_token_ablation.py --arm-masks): the
dense model's text, the same noise, so FM loss, CoC NLL and minADE are paired per clip.

  A2-heldout  FM loss highest under `coc`, NLL highest under `traj`, `dual` lowest on both
              (paired one-sided Wilcoxon, p < 0.01 each)
  A2-bands    each arm's mask applied in layers 22-35 only ("late") and in layers 0-21 only
              ("trunk"), dense elsewhere: at which depth does a criterion lose which channel?
              (i)   late-only masks leave the FM loss within 1% of dense for every arm, and
                    late-only `traj` raises the NLL more than late-only `coc`
              (ii)  trunk-only `coc` raises the FM loss more than trunk-only `traj`
              (iii) trunk-only + late-only damage is within 25% of the full mask's, per arm and loss

Usage:
  python experiments/head_analysis/analyze_arm_heldout.py --shards armheld_v1_s0 armheld_v1_s1 \
      armheld_v1_s2 armheld_v1_s3 --out armheld_v1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))

from analyze_gradient_anatomy import C1, C2, C3, MUTED, plt  # noqa: E402
from analyze_token_ablation import merge  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ARMS = ("dual", "traj", "coc")
READ = (("fm", "FM loss"), ("nll", "CoC NLL"), ("ade", "minADE"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    per, ids, buckets = merge(args.shards)
    cfg0 = json.loads((REPO / "outputs" / args.shards[0] / "config.json").read_text())
    rng = np.random.default_rng(0)
    x = {k: {c.removeprefix("arm_"): np.array(r[k], float) for c, r in per.items()} for k, _ in READ}
    has_bands = all(f"{a}_{b}" in x["fm"] for a in ARMS for b in ("late", "trunk"))

    def ci(d):
        b = rng.choice(d, (args.n_boot, len(d))).mean(1)
        return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    def greater(k, hi, lo):
        return (float(wilcoxon(x[k][hi], x[k][lo], alternative="greater").pvalue),
                float(np.median(x[k][hi] - x[k][lo])))

    out = {"n_clips": len(ids), "manifest": cfg0["manifest"], "configs": {}, "tests": {}}
    lines = [f"shipped arms on held-out clips -- {len(ids)} clips of {cfg0['manifest']}, dense text, K={cfg0['k_samples']}",
             f"  {'config':11s} " + " ".join(f"{name:>9s} {'d mean [95% CI]':>27s} {'rel':>7s}" for _, name in READ)]
    for c in x["fm"]:
        row, out["configs"][c] = f"  {c:11s}", {}
        for k, _ in READ:
            d = x[k][c] - x[k]["dense"]
            lo, hi = ci(d) if c != "dense" else (0.0, 0.0)
            out["configs"][c][k] = {"mean": float(x[k][c].mean()), "delta": float(d.mean()), "ci": [lo, hi],
                                    "median_delta": float(np.median(d)), "rel": float(d.mean() / x[k]["dense"].mean())}
            row += f" {x[k][c].mean():9.4f} {d.mean():+9.4f} [{lo:+.4f},{hi:+.4f}] {d.mean() / x[k]['dense'].mean():+7.1%}"
        lines.append(row)
    lines.append("")

    lines.append("A2-heldout (full masks)")
    for label, k, hi, lo, gated in (("FM", "fm", "coc", "dual", True), ("FM", "fm", "coc", "traj", True),
                                    ("NLL", "nll", "traj", "dual", True), ("NLL", "nll", "traj", "coc", True),
                                    ("FM", "fm", "traj", "dual", True), ("NLL", "nll", "coc", "dual", True),
                                    ("minADE", "ade", "coc", "dual", False), ("minADE", "ade", "traj", "dual", False),
                                    ("minADE", "ade", "coc", "traj", False)):
        p, med = greater(k, hi, lo)
        out["tests"][f"{label}:{hi}>{lo}"] = {"p": p, "median_diff": med, "gated": gated, "pass": bool(p < 0.01)}
        lines.append(f"  {label:6s} {hi:4s} > {lo:4s}: median paired diff {med:+.4f}, one-sided Wilcoxon p = {p:.2e}"
                     + (f" -> {'PASS' if p < 0.01 else 'FAIL'}" if gated else "  (descriptive)"))
    out["pass"] = bool(all(t["pass"] for t in out["tests"].values() if t["gated"]))
    a2 = REPO / "outputs" / "gradanat_pruned_v1" / "metrics.json"
    gap_in = None
    if a2.exists():
        m = json.loads(a2.read_text())["A2"]["nll_mean"]
        gap_in = m["traj"] - m["coc"]
    gap_out = float(x["nll"]["traj"].mean() - x["nll"]["coc"].mean())
    out["nll_gap_traj_minus_coc"] = {"held_out": gap_out, "in_sample": gap_in}
    lines += [f"  A2-heldout -> {'PASS' if out['pass'] else 'FAIL'}",
              f"  NLL gap traj - coc: held out {gap_out:+.4f}" + (f", in sample (calib_100) {gap_in:+.4f}" if gap_in is not None else ""), ""]

    if has_bands:
        bands = {"tests": {}}
        lines.append("A2-bands (masks applied in one depth band only)")
        rel_late = {a: out["configs"][f"{a}_late"]["fm"]["rel"] for a in ARMS}
        p_nll, med_nll = greater("nll", "traj_late", "coc_late")
        bands["i"] = {"late_only_fm_rel": rel_late, "fm_within_1pct": bool(all(abs(v) < 0.01 for v in rel_late.values())),
                      "nll_traj_late_gt_coc_late": {"p": p_nll, "median_diff": med_nll}}
        bands["i"]["pass"] = bool(bands["i"]["fm_within_1pct"] and p_nll < 0.01)
        lines.append("  (i)   late-only FM loss change / dense: " + ", ".join(f"{a} {v:+.2%}" for a, v in rel_late.items())
                     + f" (|.| < 1%: {'yes' if bands['i']['fm_within_1pct'] else 'no'}); NLL traj_late > coc_late: median "
                     f"{med_nll:+.4f}, p = {p_nll:.2e} -> {'PASS' if bands['i']['pass'] else 'FAIL'}")
        p_fm, med_fm = greater("fm", "coc_trunk", "traj_trunk")
        bands["ii"] = {"fm_coc_trunk_gt_traj_trunk": {"p": p_fm, "median_diff": med_fm}, "pass": bool(p_fm < 0.01)}
        lines.append(f"  (ii)  FM loss coc_trunk > traj_trunk: median {med_fm:+.4f}, p = {p_fm:.2e} -> "
                     f"{'PASS' if p_fm < 0.01 else 'FAIL'}")
        add = {}
        for a in ARMS:
            for k, _ in READ[:2]:
                full = out["configs"][a][k]["delta"]
                parts = out["configs"][f"{a}_trunk"][k]["delta"] + out["configs"][f"{a}_late"][k]["delta"]
                add[f"{a}|{k}"] = {"trunk": out["configs"][f"{a}_trunk"][k]["delta"], "late": out["configs"][f"{a}_late"][k]["delta"],
                                   "full": full, "parts_over_full": float(parts / full) if full != 0 else None}
        bands["iii"] = {"additivity": add,
                        "pass": bool(all(v["parts_over_full"] is not None and 0.75 <= v["parts_over_full"] <= 1.25
                                         for v in add.values()))}
        lines.append("  (iii) damage by band, mean delta (trunk-only + late-only vs the full mask):")
        for key, v in add.items():
            lines.append(f"          {key:9s} trunk {v['trunk']:+.4f}  late {v['late']:+.4f}  full {v['full']:+.4f}  "
                         f"parts / full {v['parts_over_full']:.2f}")
        lines.append(f"        -> {'PASS' if bands['iii']['pass'] else 'FAIL'} (every ratio within 0.75-1.25)")
        for label, k, hi, lo in (("NLL", "nll", "traj_trunk", "coc_trunk"), ("NLL", "nll", "traj_late", "dual_late"),
                                 ("NLL", "nll", "coc_late", "dual_late"), ("FM", "fm", "coc_trunk", "dual_trunk"),
                                 ("FM", "fm", "traj_trunk", "dual_trunk"), ("minADE", "ade", "coc_trunk", "traj_trunk"),
                                 ("minADE", "ade", "coc_late", "traj_late")):
            p, med = greater(k, hi, lo)
            bands["tests"][f"{label}:{hi}>{lo}"] = {"p": p, "median_diff": med}
            lines.append(f"  descriptive {label:6s} {hi:10s} > {lo:10s}: median {med:+.4f}, p = {p:.2e}")
        out["bands"] = bands

    # descriptive: the full masks by driving situation (small n per bucket; the 2,533-clip stored
    # records of analyze_failure_by_situation.py are the place to read situations from)
    b = np.array(buckets)
    out["by_bucket"] = {}
    lines += ["", "full masks by driving situation (mean delta; descriptive, small n)",
              f"  {'bucket':11s} {'n':>3s} " + " ".join(f"{a + ' dFM':>10s} {a + ' dNLL':>10s}" for a in ARMS)]
    for name in ("cruise", "accel", "decel_stop", "turn"):
        m = b == name
        if m.sum() < 3:
            continue
        out["by_bucket"][name] = {"n": int(m.sum()), **{
            a: {k: float((x[k][a][m] - x[k]["dense"][m]).mean()) for k in ("fm", "nll", "ade")} for a in ARMS}}
        lines.append(f"  {name:11s} {int(m.sum()):3d} " + " ".join(
            f"{out['by_bucket'][name][a]['fm']:+10.4f} {out['by_bucket'][name][a]['nll']:+10.4f}" for a in ARMS))

    out_dir = REPO / "outputs" / args.out
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    if has_bands:
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
        for ax_, (k, name) in zip(axs, READ):
            for i, (a, col) in enumerate(zip(ARMS, (C2, C1, C3))):
                vals = [out["configs"][c][k]["delta"] for c in (f"{a}_trunk", f"{a}_late", a)]
                err = [[v - out["configs"][c][k]["ci"][0] for v, c in zip(vals, (f"{a}_trunk", f"{a}_late", a))],
                       [out["configs"][c][k]["ci"][1] - v for v, c in zip(vals, (f"{a}_trunk", f"{a}_late", a))]]
                ax_.bar(np.arange(3) + 0.27 * (i - 1), vals, width=0.25, color=col, yerr=err, error_kw={"lw": 0.7}, label=a)
            ax_.set_xticks(np.arange(3), ["layers 0-21 only", "layers 22-35 only", "full mask"])
            ax_.axhline(0, color=MUTED, lw=0.6)
            ax_.set_title(f"{name}: mask - dense (held-out clips)")
        fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(out_dir / "plots" / "damage_by_band.png", dpi=150)
        plt.close(fig)

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    (out_dir / "config.json").write_text(json.dumps({
        "plan": "plans/2026-09-21_importance-causal-validation.md (A2-heldout, A2-bands)", "shards": args.shards,
        "clip_ids": ids, "buckets": buckets, "arm_masks": cfg0["arm_masks"], "n_boot": args.n_boot}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
