"""Step 2 of the deeper layer 1-4 MLP cut: open loop, paired per clip with B0 (l0keep) and dense.

Plan: cvpr27/plans/2026-10-29_early-mlp-cut.md (section 4). B0's rows are swap_ship_L0fullM_*, the
same selection as slim_dual_u40_l0keep_v2 (check_early_builds.py G0a); dense is baseline_ada_ps_*.
All rows come from run_baseline.py on Ada with k=8, seed 42, max_gen 256.

  minADE@6 / minFDE@6  min over the first six of the eight stored samples (the paper's protocol)
  verdict   mean delta(arm - B0) of minADE@6 against (B0 - dense) * (extra removed / B0 removed):
            FREE if CI hi < threshold, REJECT if CI lo >= threshold, else INCONCLUSIVE
  to step 3 no set REJECTed and CoC degeneracy <= B0 + 1 pp on every set; the deepest such arm goes

Bootstrap: paired over clips, 200,000 draws, 95% CI of the mean (and of the median for minADE@6).

Usage:
  .venv/bin/python cvpr27/code/analyze_early_mlp_openloop.py [--exp-id cvpr27/early_mlp]
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, INK, MUTED, GRID = "#FAF9F5", "#29261B", "#6B6555", "#E8E6DC"
C_PT = "#2a78d6"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.size": 9, "axes.grid": True, "axes.axisbelow": True,
})
REPO = Path(__file__).resolve().parents[2]
# set -> (B0 rows, dense rows)
SETS = {"test500": ("swap_ship_L0fullM_500", "baseline_ada_ps_test"),
        "val500": ("swap_ship_L0fullM_val500", "baseline_ada_ps_indist"),
        "oodval": ("swap_ship_L0fullM_oodval", "baseline_ada_ps_oodval")}
ARMS = {"a10": "e10", "a05": "e05"}  # arm -> its mask config in masks_config.json
DEPTH_ORDER = ("a05", "a10")         # deepest first
DRAWS = 200_000


def rows(d):
    out = {}
    for f in glob.glob(str(REPO / "outputs" / d / "*.json")):
        if f.endswith(("config.json", "metrics.json")):
            continue
        data = json.loads(Path(f).read_text())
        if isinstance(data, list):
            out.update({r["clip_id"]: r for r in data if "ade_rollout_k" in r})
    return out


def at6(rr, ids, key):
    return np.array([min(rr[c][key][:6]) for c in ids])  # (n,)


def boot(d, rng, stat=np.mean, chunk=20_000):
    """d: (n,) paired differences -> (lo, hi) 95% CI of `stat`."""
    vals = []
    for s in range(0, DRAWS, chunk):
        idx = rng.integers(0, len(d), size=(min(chunk, DRAWS - s), len(d)))  # (chunk, n)
        vals.append(stat(d[idx], axis=1))
    lo, hi = np.percentile(np.concatenate(vals), [2.5, 97.5])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="cvpr27/early_mlp")
    args = ap.parse_args()
    root = REPO / "outputs" / args.exp_id
    mcfg = json.loads((root / "masks_config.json").read_text())["configs"]
    rng = np.random.default_rng(0)

    res = {}
    for s, (b0_dir, dense_dir) in SETS.items():
        b0, dense = rows(b0_dir), rows(dense_dir)
        for arm, cfg in ARMS.items():
            a = rows(f"cvpr27/early_{arm}_{s}")
            if not a:
                continue
            ids = sorted(set(a) & set(b0) & set(dense))
            ratio = mcfg[cfg]["extra_over_b0_removed"]
            r = {"n": len(ids), "arm_rows": len(a), "ratio": ratio}
            for key, lab in (("ade_rollout_k", "minADE@6"), ("fde_rollout_k", "minFDE@6")):
                x, y, z = at6(a, ids, key), at6(b0, ids, key), at6(dense, ids, key)
                d = x - y
                r[lab] = {"arm": {"mean": float(x.mean()), "median": float(np.median(x))},
                          "b0": {"mean": float(y.mean()), "median": float(np.median(y))},
                          "dense": {"mean": float(z.mean()), "median": float(np.median(z))},
                          "delta_b0": {"mean": float(d.mean()), "ci": boot(d, rng),
                                       "median": float(np.median(d))},
                          "delta_dense_mean": float((x - z).mean())}
                if lab == "minADE@6":
                    r[lab]["delta_b0"]["median_ci"] = boot(d, rng, stat=np.median)
                    cost = y.mean() - z.mean()
                    thr = float(cost * ratio)
                    lo, hi = r[lab]["delta_b0"]["ci"]
                    r["threshold"] = thr
                    r["b0_minus_dense"] = float(cost)
                    r["verdict"] = ("UNDEFINED" if thr <= 0 else "FREE" if hi < thr
                                    else "REJECT" if lo >= thr else "INCONCLUSIVE")
                    buckets = {}
                    for bk in sorted({b0[c]["bucket"] for c in ids}):
                        sel = np.array([b0[c]["bucket"] == bk for c in ids])
                        buckets[bk] = {"n": int(sel.sum()), "b0_mean": float(y[sel].mean()),
                                       "delta_mean": float(d[sel].mean())}
                    r["buckets"] = buckets
            deg_a = float(np.mean([bool(a[c].get("coc_degenerate")) for c in ids]) * 100)
            deg_b = float(np.mean([bool(b0[c].get("coc_degenerate")) for c in ids]) * 100)
            r["coc_degen_pct"] = {"arm": deg_a, "b0": deg_b, "ok": deg_a <= deg_b + 1.0}
            res.setdefault(arm, {})[s] = r

    step3 = {}
    for arm in DEPTH_ORDER:
        sets = res.get(arm, {})
        complete = len(sets) == len(SETS)
        ok = complete and all(v["verdict"] != "REJECT" and v["coc_degen_pct"]["ok"] for v in sets.values())
        step3[arm] = {"complete": complete, "qualifies": ok}
    chosen = next((a for a in DEPTH_ORDER if step3[a]["qualifies"]), None)
    all_done = all(step3[a]["complete"] for a in DEPTH_ORDER)

    out = root / "openloop_analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "plan": "cvpr27/plans/2026-10-29_early-mlp-cut.md (section 4)", "sets": SETS,
        "arms": {a: f"cvpr27/early_{a}_<set>" for a in ARMS}, "bootstrap_draws": DRAWS,
        "bootstrap_seed": 0, "metric": "min over the first 6 of 8 stored samples"}, indent=2))
    (out / "metrics.json").write_text(json.dumps(
        {"results": res, "step3": step3, "chosen_for_step3": chosen if all_done else None}, indent=2))

    lines = ["Step 2 open loop: arm vs B0 (l0keep, swap_ship_L0fullM_*) and dense (baseline_ada_ps_*)", ""]
    for arm in ARMS:
        for s, r in res.get(arm, {}).items():
            m, f = r["minADE@6"], r["minFDE@6"]
            lines += [
                f"{arm} {s} (n={r['n']}, arm rows {r['arm_rows']})",
                (f"  minADE@6 mean / median: arm {m['arm']['mean']:.4f} / {m['arm']['median']:.4f}   "
                f"B0 {m['b0']['mean']:.4f} / {m['b0']['median']:.4f}   dense {m['dense']['mean']:.4f} / {m['dense']['median']:.4f}"),
                (f"  delta vs B0: mean {m['delta_b0']['mean']:+.4f} [{m['delta_b0']['ci'][0]:+.4f}, {m['delta_b0']['ci'][1]:+.4f}]  "
                f"median {m['delta_b0']['median']:+.4f} [{m['delta_b0']['median_ci'][0]:+.4f}, {m['delta_b0']['median_ci'][1]:+.4f}]  "
                f"vs dense {m['delta_dense_mean']:+.4f}"),
                f"  threshold {r['threshold']:+.4f} (B0-dense {r['b0_minus_dense']:+.4f} x {r['ratio']}) -> {r['verdict']}",
                (f"  minFDE@6: arm {f['arm']['mean']:.4f}  B0 {f['b0']['mean']:.4f}  delta {f['delta_b0']['mean']:+.4f} "
                f"[{f['delta_b0']['ci'][0]:+.4f}, {f['delta_b0']['ci'][1]:+.4f}]"),
                (f"  CoC degeneracy: arm {r['coc_degen_pct']['arm']:.1f}%  B0 {r['coc_degen_pct']['b0']:.1f}%  "
                f"{'ok' if r['coc_degen_pct']['ok'] else 'OVER B0 + 1 pp'}"),
                "  buckets (n, B0 mean, delta): " + "  ".join(
                    f"{bk} {v['n']} {v['b0_mean']:.3f} {v['delta_mean']:+.3f}" for bk, v in r["buckets"].items()),
                ""]
    lines.append("Step 3: " + ", ".join(
        f"{a} {'qualifies' if v['qualifies'] else 'does not qualify' if v['complete'] else 'incomplete'}"
        for a, v in step3.items()) + (f" -> {chosen}" if all_done else " (waiting for runs)"))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # dot-and-whisker of delta minADE@6 vs B0, one row per (set, arm); ticks mark the thresholds
    items = [(s, a) for s in SETS for a in ARMS if s in res.get(a, {})]
    if items:
        fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(items) + 1.2))
        ys = np.arange(len(items))[::-1]
        for y, (s, a) in zip(ys, items):
            r = res[a][s]
            dm = r["minADE@6"]["delta_b0"]
            ax.errorbar(dm["mean"], y, xerr=[[dm["mean"] - dm["ci"][0]], [dm["ci"][1] - dm["mean"]]],
                        fmt="o", color=C_PT, ms=5, lw=2, capsize=0)
            ax.plot([r["threshold"]] * 2, [y - 0.35, y + 0.35], color=INK, lw=1.5)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(ys, [f"{a}  {s}" for s, a in items])
        ax.set_xlabel("minADE@6: arm - B0 (m, mean, 95% CI)")
        ax.set_title("Open loop vs l0keep   (| = proportional threshold)", fontsize=9, loc="left")
        fig.tight_layout()
        fig.savefig(out / "openloop.png", dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
