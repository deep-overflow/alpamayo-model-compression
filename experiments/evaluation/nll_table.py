"""CoC NLL table across arms: reference NLL on OOD-val, dense-reference NLL on test500.

Two metrics, one table format, picked with `--metric`:

  nll_gtcoc   mean per-token NLL (nats) of the curated `gt_coc` teacher-forced as context,
              written by `run_baseline.py --set ood` unless `--no-tf`. OOD-val only, since
              test500 carries no reference text. Correctness-anchored.
  nll_dense   the same span rule against the unpruned model's own rollout on that clip,
              written by `run_nll_dense.py --set test`. In-distribution, but a fidelity to
              the dense model rather than a correctness score, and the quantity `I_CoC`
              was fitted on -- so `coc` and `dual` are structurally favoured. Read it
              beside `nll_gtcoc`, never in its place.

Neither uses `nll_self`: that scores the arm's own sample, and on a collapsed arm it
measures the soup (wanda: 11.9 overall, 0.97 on its 62 non-degenerate clips).

Rows are paired clip-for-clip against baseline over the intersection of clips; Δ is the
per-clip difference, with a bootstrap CI of the mean and a Wilcoxon signed-rank test,
following `analyze_arms.py`. Arms whose rows dir is missing or lacks the metric are listed
as pending rather than failing the table, so it can be rebuilt as runs land. For
`nll_dense` the baseline row doubles as the round-trip gate: its value must reproduce the
stored `nll_self` (same span), and the summary prints that gap.

Usage:
  python experiments/evaluation/nll_table.py --metric nll_gtcoc
  python experiments/evaluation/nll_table.py --metric nll_dense --exp-id nll_dense_table
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]

# metric -> arm -> (rows dir under outputs/, restrict to split == "val")
# nll_gtcoc: the *_ood dirs are the full 1,533-clip OOD runs and carry `split`; the
# *_oodval / *_pred_oodval dirs already are the 262-clip ood_val manifest. Mixing the two
# is safe: dual from both sources is bit-identical on every clip (checked 2026-09-23).
ARMS = {
    "nll_gtcoc": {
        "baseline": ("baseline_pred_oodval", False),
        "wanda": ("wanda_u40_v2_tf_oodval", False),
        "llm-pruner": ("lp_r50_oodval", False),
        "tyr-the-pruner": ("tyr_u40_r_pred_oodval", False),
        "traj": ("traj_u40_v2_ood", True),
        "coc": ("coc_u40_v2_ood", True),
        "dual": ("dual_u40_v2_pred_oodval", False),
    },
    "nll_dense": {
        "baseline": ("nll_dense_baseline_test", False),
        "wanda": ("nll_dense_slim_wanda_u40_v2_test", False),
        "llm-pruner": ("nll_dense_lp_r50_test", False),
        "tyr-the-pruner": ("nll_dense_slim_tyr_u40_r_test", False),
        "traj": ("nll_dense_slim_traj_u40_v2_test", False),
        "coc": ("nll_dense_slim_coc_u40_v2_test", False),
        "dual": ("nll_dense_slim_dual_u40_v2_test", False),
    },
}
SET_OF = {"nll_gtcoc": "ood_val", "nll_dense": "test_500"}
EXPECT_N = {"nll_gtcoc": 262, "nll_dense": 500}
BUCKETS = ("cruise", "decel_stop", "accel", "turn")

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1 = "#2a78d6"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})


def load_rows(d, val_only):
    """clip_id -> row for one run dir, merging shards. None if the dir has no rows."""
    if not d.is_dir():
        return None
    rows = []
    for p in sorted(d.glob("*.json")):
        if p.name == "config.json":
            continue
        rows += json.loads(p.read_text())
    if val_only:
        rows = [r for r in rows if r.get("split") == "val"]
    return {r["clip_id"]: r for r in rows} or None


def gpu_of(d):
    cfg = d / "config.json"
    return json.loads(cfg.read_text()).get("gpu") if cfg.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=list(ARMS), default="nll_gtcoc")
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--exp-id", default=None, help="default <metric>_table")
    ap.add_argument("--out-root", default=str(REPO / "outputs"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    metric = args.metric
    arms = args.arms or list(ARMS[metric])
    out_root = Path(args.out_root)
    out_dir = out_root / (args.exp_id or f"{metric}_table")
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    rows, pending, gpus = {}, {}, {}
    for a in arms:
        d, val_only = ARMS[metric][a]
        R = load_rows(out_root / d, val_only)
        if R is None:
            pending[a] = f"{d}: no rows"
            continue
        R = {k: r for k, r in R.items() if metric in r}
        if not R:
            pending[a] = f"{d}: rows carry no {metric}"
            continue
        if len(R) < EXPECT_N[metric]:
            pending[a] = f"{d}: partial, {len(R)}/{EXPECT_N[metric]} clips so far"
        rows[a] = R
        gpus[a] = gpu_of(out_root / d)
    if "baseline" not in rows:
        raise SystemExit("baseline rows are required for the paired deltas")

    common = sorted(set.intersection(*(set(R) for R in rows.values())))
    base = np.array([rows["baseline"][k][metric] for k in common])
    bucket = {k: rows["baseline"][k]["bucket"] for k in common}
    rng = np.random.default_rng(args.seed)

    table = {}
    for a, R in rows.items():
        v = np.array([R[k][metric] for k in common])
        ent = {"n": len(v), "mean": float(v.mean()), "median": float(np.median(v)),
               "sd": float(v.std()),
               "by_bucket": {b: float(np.mean([R[k][metric] for k in common
                                                if bucket[k] == b])) for b in BUCKETS}}
        if a != "baseline":
            d = v - base
            boot = np.array([d[rng.integers(0, len(d), len(d))].mean()
                             for _ in range(args.n_boot)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            ent["delta"] = {"mean": float(d.mean()), "ci95": [float(lo), float(hi)],
                            "median": float(np.median(d)),
                            "wilcoxon_p": float(wilcoxon(d).pvalue),
                            "worse_frac": float((d > 0).mean())}
        table[a] = ent

    # nll_dense only: the baseline scored on its own rollout must reproduce nll_self
    gate = None
    if metric == "nll_dense" and "ref_nll_self" in next(iter(rows["baseline"].values())):
        g = np.array([rows["baseline"][k][metric] - rows["baseline"][k]["ref_nll_self"]
                      for k in common])
        gate = {"mean_gap": float(g.mean()), "max_abs_gap": float(np.abs(g).max()),
                "n": len(g)}

    # ---- plot: paired delta with CI ----
    arms_d = [a for a in table if a != "baseline"]
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * max(1, len(arms_d)) + 1.6))
    y = np.arange(len(arms_d))
    dm = [table[a]["delta"]["mean"] for a in arms_d]
    err = np.array([[dm[i] - table[a]["delta"]["ci95"][0],
                     table[a]["delta"]["ci95"][1] - dm[i]]
                    for i, a in enumerate(arms_d)]).T if arms_d else None
    ax.barh(y, dm, xerr=err, color=C1, ecolor=INK, capsize=3, height=0.6)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(arms_d)
    ax.invert_yaxis()
    ax.set_xlabel(f"Δ {metric} vs baseline (nats/token, mean, bootstrap 95% CI)")
    ax.set_title(f"{metric} on {SET_OF[metric]}, n={len(common)}")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / f"delta_{metric}.png", dpi=150)
    plt.close(fig)

    # ---- records ----
    (out_dir / "config.json").write_text(json.dumps({
        "metric": metric, "set": SET_OF[metric], "arms": {a: ARMS[metric][a][0] for a in arms},
        "n_common_clips": len(common), "n_boot": args.n_boot, "seed": args.seed,
        "gpu": gpus, "pending": pending, "source": "experiments/evaluation/nll_table.py",
    }, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(
        {"table": table, "gate": gate, "pending": pending, "n_common_clips": len(common)},
        indent=2))

    lines = [(f"{metric} (nats/token) on {SET_OF[metric]}, n={len(common)} common clips, "
              f"paired vs baseline"),
             (f"{'arm':16}{'mean':>8}{'median':>8}{'sd':>7} | {'dmean':>8}{'95% CI':>20}"
              f"{'dmedian':>9}{'p(Wilcoxon)':>13}{'worse%':>8}")]
    for a, e in table.items():
        s = f"{a:16}{e['mean']:>8.3f}{e['median']:>8.3f}{e['sd']:>7.3f} |"
        if "delta" in e:
            dd = e["delta"]
            s += (f" {dd['mean']:>+8.3f}  [{dd['ci95'][0]:+.3f}, {dd['ci95'][1]:+.3f}]"
                  f"{dd['median']:>+9.3f}{dd['wilcoxon_p']:>13.2e}{100 * dd['worse_frac']:>7.1f}%")
        lines.append(s)
    lines += ["", "per-bucket mean  " + "".join(f"{b:>13}" for b in BUCKETS)]
    for a, e in table.items():
        lines.append(f"{a:16}" + "".join(f"{e['by_bucket'][b]:>13.3f}" for b in BUCKETS))
    lines.append(f"{'n':16}" + "".join(
        f"{sum(1 for k in common if bucket[k] == b):>13}" for b in BUCKETS))
    if gate:
        lines += ["", (f"round-trip gate (baseline nll_dense - stored nll_self): mean "
                       f"{gate['mean_gap']:+.4f}, max|.| {gate['max_abs_gap']:.4f}, n={gate['n']}")]
    if len(set(gpus.values())) > 1:
        lines += ["", "WARNING: mixed GPU architectures: " + json.dumps(gpus)]
    if pending:
        lines += ["", "pending:"] + [f"  {a}: {why}" for a, why in pending.items()]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"\n-> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
