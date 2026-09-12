"""Does the calibration draw rule move a pruned model? G0-G5 of the action-stratified draw.

plans/2026-09-11_action-stratified-calib.md. Nine `dual_u40_v2` arms that differ only in
which 100 calibration clips the Taylor scores were measured on: three draw rules (rd
random / se stratified to the test500 mix / su stratified uniform) x three seeds. Every
arm is paired with the same unpruned test500 run.

The unit of comparison is the clip. For each rule, a clip's delta is the mean over that
rule's seeds of (arm - baseline) minADE@6, so a rule contrast is a 500-clip paired
difference of 3-seed means -- SE ~ 0.808/sqrt(1500) = 0.021 measured on the nt pairs --
rather than three arm-level numbers. The bootstrap CI is on the mean (the protocol's
headline); the median and Wilcoxon are printed beside it. Arm-level SD per rule (G5) is
reported but is not a gate: with three seeds its interval is 0.5-6x the estimate.

Buckets are the evaluation record's `bucket` (eval_lib.bucket, turn sign merged), the same
rule the calibration strata used, so calibration and evaluation strata share one ruler.

Arms not yet evaluated are skipped and the gates that need them read "pending", so this
runs while the queue is still filling.

Usage:
  python experiments/evaluation/analyze_strat_calib.py [--out outputs/strat_calib]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

import paper_numbers as pn

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

BASELINE = "baseline_ada_ps_test"
RULES = {"rd": "random", "se": "strat-eval", "su": "strat-uniform"}
SEEDS = "abc"
REF = {"nt": "abcde", "tr": "abcde"}          # the earlier n=100 families, for context
CALIB_100 = "dual_u40_v2_ps_test"
BUCKETS = ["cruise", "decel_stop", "accel", "turn"]
BOOT = 10000
EXPECT = 500
MDE = 0.04       # rule-contrast resolution (2 SE), pre-registered
G2_EFFECT = -0.05
G4_TOL = 0.04


def load_arm(dirname):
    try:
        r = pn.load(dirname, False)
    except (SystemExit, ValueError):
        return None
    return r if len(r) >= EXPECT else None


def deltas(base, arm):
    """clip -> arm - baseline minADE@6 over the paired clips."""
    return {c: pn.at6(arm[c], "ade_rollout_k") - pn.at6(base[c], "ade_rollout_k")
            for c in base if c in arm}


def rule_delta(arms):
    """clip -> mean over the rule's available seeds; only clips every seed has."""
    if not arms:
        return {}
    common = set.intersection(*[set(a) for a in arms])
    return {c: float(np.mean([a[c] for a in arms])) for c in common}


def contrast(da, db, clips):
    """Paired (a - b) over clips: mean with bootstrap CI, median, Wilcoxon."""
    ids = sorted(c for c in clips if c in da and c in db)
    if len(ids) < 10:
        return None
    d = np.array([da[c] - db[c] for c in ids])
    rng = np.random.default_rng(0)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = float(wilcoxon(d).pvalue) if np.any(d) else 1.0
    return {"n": len(ids), "mean": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "median": float(np.median(d)), "p": p, "sig": bool(lo > 0 or hi < 0)}


def fmt(c):
    if c is None:
        return "pending"
    return (f"{c['mean']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}]{'*' if c['sig'] else ' '} "
            f"med {c['median']:+.4f} p={c['p']:.2g} n={c['n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/strat_calib")
    args = ap.parse_args()
    out = REPO / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)

    base = load_arm(BASELINE)
    assert base is not None, f"{BASELINE} missing"
    bucket = {c: base[c]["bucket"] for c in base}
    clips_by_bucket = {b: [c for c in base if bucket[c] == b] for b in BUCKETS}

    # ---- arms -----------------------------------------------------------------------
    per_arm = {}      # arm -> clip deltas
    arm_level = {}    # arm -> {minADE, dminADE, coc_deg}
    found = {}
    for rule in RULES:
        for s in SEEDS:
            name = f"{rule}_{s}"
            r = load_arm(f"dual_{name}_test")
            found[name] = r is not None
            if r is None:
                continue
            per_arm[name] = deltas(base, r)
            a, _, deg = pn.stats(r)
            arm_level[name] = {"rule": rule, "minADE": float(a.mean()),
                               "dminADE": float(np.mean(list(per_arm[name].values()))),
                               "coc_degenerate": deg}
    for fam, letters in REF.items():
        for s in letters:
            r = load_arm(f"dual_{fam}_{s}_test")
            if r is not None:
                per_arm[f"{fam}_{s}"] = deltas(base, r)
                a, _, deg = pn.stats(r)
                arm_level[f"{fam}_{s}"] = {"rule": fam, "minADE": float(a.mean()),
                                           "dminADE": float(np.mean(list(per_arm[f"{fam}_{s}"].values()))),
                                           "coc_degenerate": deg}
    r = load_arm(CALIB_100)
    if r is not None:
        per_arm["calib_100"] = deltas(base, r)
        a, _, deg = pn.stats(r)
        arm_level["calib_100"] = {"rule": "calib_100", "minADE": float(a.mean()),
                                  "dminADE": float(np.mean(list(per_arm["calib_100"].values()))),
                                  "coc_degenerate": deg}
    n_found = sum(found.values())
    print(f"arms: {n_found}/9 evaluated  " + " ".join(k for k, v in found.items() if v))

    rule_d = {rule: rule_delta([per_arm[f"{rule}_{s}"] for s in SEEDS if f"{rule}_{s}" in per_arm])
              for rule in RULES}
    rule_d["nt"] = rule_delta([per_arm[f"nt_{s}"] for s in REF["nt"] if f"nt_{s}" in per_arm])
    rule_d["tr"] = rule_delta([per_arm[f"tr_{s}"] for s in REF["tr"] if f"tr_{s}" in per_arm])
    n_seeds = {rule: sum(f"{rule}_{s}" in per_arm for s in SEEDS) for rule in RULES}

    # ---- per-rule loss by bucket (vs unpruned), the plan's appendix table prospectively
    by_bucket = {}
    for rule, d in rule_d.items():
        if not d:
            continue
        by_bucket[rule] = {"all": float(np.mean(list(d.values())))}
        for b in BUCKETS:
            v = [d[c] for c in clips_by_bucket[b] if c in d]
            by_bucket[rule][b] = float(np.mean(v)) if v else None

    # ---- contrasts and gates ------------------------------------------------------
    def full(a, b):
        c = {"all": contrast(rule_d[a], rule_d[b], list(base))}
        for bk in BUCKETS:
            c[bk] = contrast(rule_d[a], rule_d[b], clips_by_bucket[bk])
        return c

    complete = all(n_seeds[r] == len(SEEDS) for r in RULES)
    con = {"rd-nt": full("rd", "nt"), "su-rd": full("su", "rd"), "se-rd": full("se", "rd"),
           "su-se": full("su", "se")}

    gates = {"complete": complete, "n_seeds": n_seeds}
    c = con["rd-nt"]["all"]
    if c is None or n_seeds["rd"] < len(SEEDS) or not rule_d["nt"]:
        gates["G1_matching"] = "pending"
    elif abs(c["mean"]) < MDE or not c["sig"]:
        gates["G1_matching"] = "meaningless"
    else:
        gates["G1_matching"] = "helps" if c["mean"] > 0 else "harms"
    g2 = [con["su-rd"][b] for b in ("decel_stop", "accel", "turn")]
    if any(x is None for x in g2) or not complete:
        gates["G2_diversity_noncruise"] = "pending"
    else:
        hits = sum(x["mean"] <= G2_EFFECT and x["sig"] for x in g2)
        gates["G2_diversity_noncruise"] = {"buckets_passing": hits, "pass": hits >= 2}
    c = con["su-rd"]["cruise"]
    gates["G3_cruise_price"] = "pending" if c is None or not complete else \
        {"mean": c["mean"], "sig": c["sig"], "price_ge_0.05": c["mean"] >= 0.05}
    g4 = [con["se-rd"][k] for k in ["all"] + BUCKETS]
    if any(x is None for x in g4) or not complete:
        gates["G4_evalmatch_null"] = "pending"
    else:
        gates["G4_evalmatch_null"] = {"pass": all(abs(x["mean"]) < G4_TOL and not x["sig"]
                                                 for x in g4)}
    g5 = {}
    for rule in list(RULES) + list(REF):
        v = [arm_level[k]["minADE"] for k in arm_level if arm_level[k]["rule"] == rule]
        if len(v) >= 2:
            g5[rule] = {"n": len(v), "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                        "range": float(np.max(v) - np.min(v))}
    gates["G5_arm_sd"] = g5

    # ---- set-level facts: composition, 6-attr L1, raw action, G0 -----------------------
    facts = {}
    q = REPO / "outputs" / "eval_sets" / "quality_calib_strat700.csv"
    if q.exists():
        facts["quality"] = pd.read_csv(q).to_dict(orient="records")
    for f in ("config_labels.json", "g0_raw.json"):
        if (out / f).exists():
            facts[f.split(".")[0]] = json.loads((out / f).read_text())

    # ---- kept-set overlap: descriptive only. Overlap never predicted cost in this repo
    # (dual_st2000 92.1% -> +0.1105, maxstep11 92.5% -> n.s.), so it is reported to be
    # read against the contrasts, not as a gate.
    imp_dirs = {"calib_100": "importance_v2",
                **{f"nt_{s}": f"importance_nt500_{s}" for s in REF["nt"]},
                **{f"{r}_{s}": f"importance_{r}100_{s}" for r in RULES for s in SEEDS}}

    def complete(d):
        # run_importance checkpoints importance.npz every 10 clips, so a fetched directory
        # can hold a 10-clip partial whose selection overlaps ~80% instead of ~90%: only a
        # run whose metrics.json says n_clips == 100 (or the shipped v2 file) counts
        m = REPO / "outputs" / d / "metrics.json"
        if not (REPO / "outputs" / d / "importance.npz").exists():
            return False
        if d == "importance_v2" or not m.exists():
            return True
        n = json.loads(m.read_text()).get("n_clips")   # absent in derived nt block files
        return n is None or n == 100

    imp_dirs = {k: d for k, d in imp_dirs.items() if complete(d)}
    try:
        from analyze_calib_variance import kept_sets, overlap
        ks = kept_sets(imp_dirs)
    except Exception as e:  # noqa: BLE001  the contrasts do not depend on this block
        print(f"kept-set overlap skipped: {e}")
        ks = {}
    ov = {}
    if "calib_100" in ks:
        for k, (q, m) in ks.items():
            if k != "calib_100":
                ov[f"{k}-calib_100"] = {"q": overlap(q, ks["calib_100"][0]),
                                        "mlp": overlap(m, ks["calib_100"][1])}
    for grp in list(RULES) + ["nt"]:
        keys = [k for k in ks if k.startswith(grp + "_")]
        pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]
        if pairs:
            ov[f"within_{grp}"] = {
                "n_pairs": len(pairs),
                "q": float(np.mean([overlap(ks[a][0], ks[b][0]) for a, b in pairs])),
                "mlp": float(np.mean([overlap(ks[a][1], ks[b][1]) for a, b in pairs]))}
    # between-rule pairs: if these equal the within-rule means, the rule moved the
    # selection no more than a re-draw of natural clips does
    groups = list(RULES) + ["nt"]
    for i, ga in enumerate(groups):
        for gb in groups[i + 1:]:
            ka = [k for k in ks if k.startswith(ga + "_")]
            kb = [k for k in ks if k.startswith(gb + "_")]
            pairs = [(a, b) for a in ka for b in kb]
            if pairs:
                ov[f"between_{ga}_{gb}"] = {
                    "n_pairs": len(pairs),
                    "q": float(np.mean([overlap(ks[a][0], ks[b][0]) for a, b in pairs])),
                    "mlp": float(np.mean([overlap(ks[a][1], ks[b][1]) for a, b in pairs]))}
    facts["kept_overlap"] = ov

    metrics = {"arms": arm_level, "rule_loss_by_bucket": by_bucket, "contrasts": con,
               "gates": gates, "facts": facts}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "config.json").write_text(json.dumps({
        "baseline": BASELINE, "rules": RULES, "seeds": list(SEEDS), "reference": REF,
        "calib_100": CALIB_100, "boot": BOOT, "mde": MDE, "g2_effect": G2_EFFECT,
        "g4_tol": G4_TOL, "arms_found": found}, indent=2))

    # ---- summary ---------------------------------------------------------------------
    L = [f"== action-stratified calibration draw: {n_found}/9 arms evaluated ==", ""]
    L.append("arm-level test500 minADE@6 (mean), delta vs unpruned, CoC degeneracy:")
    for k, v in arm_level.items():
        L.append(f"  {k:10s} {v['rule']:14s} {v['minADE']:.4f}  {v['dminADE']:+.4f}  "
                 f"coc {v['coc_degenerate'] * 100:.1f}%")
    L.append("")
    L.append("rule loss vs unpruned by bucket (clip-level mean of seed means):")
    L.append(f"  {'rule':6s} " + " ".join(f"{b:>11s}" for b in ["all"] + BUCKETS))
    for rule, v in by_bucket.items():
        L.append(f"  {rule:6s} " + " ".join(
            f"{v[b]:+11.4f}" if v.get(b) is not None else f"{'-':>11s}" for b in ["all"] + BUCKETS))
    L.append("")
    for name, c in con.items():
        L.append(f"contrast {name}:")
        for k in ["all"] + BUCKETS:
            L.append(f"  {k:10s} {fmt(c[k])}")
    L.append("")
    if ov:
        L.append("kept-set overlap (dual_u40_v2 selection; Q heads / MLP channels):")
        for k, v in ov.items():
            extra = f" (n_pairs={v['n_pairs']})" if "n_pairs" in v else ""
            L.append(f"  {k:18s} Q {v['q'] * 100:5.1f}%  MLP {v['mlp'] * 100:5.1f}%{extra}")
        L.append("")
    L.append("gates: " + json.dumps(gates, ensure_ascii=False))
    text = "\n".join(L) + "\n"
    (out / "summary.txt").write_text(text)
    print(text)

    # ---- plots -----------------------------------------------------------------------
    if arm_level:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        groups = [("calib_100", C4), ("nt", MUTED), ("tr", MUTED), ("rd", C1), ("se", C2), ("su", C3)]
        for i, (g, col) in enumerate(groups):
            v = [arm_level[k]["minADE"] for k in arm_level if arm_level[k]["rule"] == g]
            if v:
                ax.scatter([i] * len(v), v, color=col, s=28, zorder=3)
                ax.hlines(np.mean(v), i - 0.25, i + 0.25, color=col, lw=2)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g for g, _ in groups])
        ax.set_ylabel("test500 minADE@6")
        ax.set_title("arm-level minADE by draw rule (bar = rule mean)")
        fig.tight_layout()
        fig.savefig(out / "plots" / "arm_minade_by_rule.png", dpi=150)
        plt.close(fig)
    if any(v is not None for c in con.values() for v in c.values()):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
        for ax, name in zip(axes, ["rd-nt", "su-rd", "se-rd"]):
            keys = ["all"] + BUCKETS
            for j, k in enumerate(keys):
                c = con[name][k]
                if c is None:
                    continue
                ax.errorbar(j, c["mean"], yerr=[[c["mean"] - c["lo"]], [c["hi"] - c["mean"]]],
                            fmt="o", color=C1 if c["sig"] else MUTED, capsize=3)
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels(keys, rotation=30)
            ax.set_title(name)
        axes[0].set_ylabel("paired dminADE@6 (mean, 95% CI)")
        fig.tight_layout()
        fig.savefig(out / "plots" / "rule_contrasts.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
