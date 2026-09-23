"""Reference-CoC NLL table: how well each arm predicts the curated OOD-val reasoning.

`nll_gtcoc` is the mean per-token NLL (nats) of the curated `gt_coc` teacher-forced as
context, written by `run_baseline.py --set ood` unless `--no-tf`. It is one forward pass
with no sampling, so it is seed-independent and defined regardless of whether the arm's
own generation has collapsed. That is why it is the language column here and `nll_self`
is not: `nll_self` scores the model's own sample, and on a collapsed arm it measures the
soup (wanda: 11.9 overall, 0.97 on its 62 non-degenerate clips).

OOD-val only. test500 carries no reference text, so no arm can have this number there.

Rows are paired clip-for-clip against baseline over the intersection of clips; Δ is the
per-clip difference, with a bootstrap CI of the mean and a Wilcoxon signed-rank test,
following `analyze_arms.py`. Arms whose rows dir is missing or has no `nll_gtcoc` are
listed as pending rather than failing the table, so it can be rebuilt as runs land.

Usage:
  python experiments/evaluation/nll_gtcoc_table.py
  python experiments/evaluation/nll_gtcoc_table.py --arms baseline dual tyr_r --exp-id nll_gtcoc_3
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

# arm -> (rows dir under outputs/, restrict to split == "val")
# The *_ood dirs are the full 1,533-clip OOD runs and carry `split`; the *_oodval and
# *_pred_oodval dirs already are the 262-clip ood_val manifest. Mixing the two is safe:
# dual from both sources is bit-identical on every clip (checked 2026-09-23).
ARMS = {
    "baseline": ("baseline_pred_oodval", False),
    "wanda": ("wanda_u40_v2_tf_oodval", False),
    "llm-pruner": ("lp_r50_oodval", False),
    "tyr-the-pruner": ("tyr_u40_r_pred_oodval", False),
    "traj": ("traj_u40_v2_ood", True),
    "coc": ("coc_u40_v2_ood", True),
    "dual": ("dual_u40_v2_pred_oodval", False),
}
BUCKETS = ("cruise", "decel_stop", "accel", "turn")

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2 = "#2a78d6", "#e87ba4"
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
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--exp-id", default="nll_gtcoc_table")
    ap.add_argument("--out-root", default=str(REPO / "outputs"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expect-n", type=int, default=262,
                    help="an arm with fewer rows than this is reported as partial")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_dir = out_root / args.exp_id
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    rows, pending, gpus = {}, {}, {}
    for a in args.arms:
        d, val_only = ARMS[a]
        R = load_rows(out_root / d, val_only)
        if R is None:
            pending[a] = f"{d}: no rows"
            continue
        if "nll_gtcoc" not in next(iter(R.values())):
            pending[a] = f"{d}: rows carry no nll_gtcoc (run with teacher forcing)"
            continue
        R = {k: r for k, r in R.items() if "nll_gtcoc" in r}
        if len(R) < args.expect_n:
            pending[a] = f"{d}: partial, {len(R)}/{args.expect_n} clips so far"
        rows[a] = R
        gpus[a] = gpu_of(out_root / d)
    if "baseline" not in rows:
        raise SystemExit("baseline rows are required for the paired deltas")

    common = sorted(set.intersection(*(set(R) for R in rows.values())))
    base = np.array([rows["baseline"][k]["nll_gtcoc"] for k in common])
    bucket = {k: rows["baseline"][k]["bucket"] for k in common}
    rng = np.random.default_rng(args.seed)

    table = {}
    for a, R in rows.items():
        v = np.array([R[k]["nll_gtcoc"] for k in common])
        ent = {"n": len(v), "mean": float(v.mean()), "median": float(np.median(v)),
               "sd": float(v.std()),
               "by_bucket": {b: float(np.mean([R[k]["nll_gtcoc"] for k in common
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

    # ---- plot: paired delta with CI ----
    arms_d = [a for a in table if a != "baseline"]
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(arms_d) + 1.6))
    y = np.arange(len(arms_d))
    dm = [table[a]["delta"]["mean"] for a in arms_d]
    err = np.array([[dm[i] - table[a]["delta"]["ci95"][0],
                     table[a]["delta"]["ci95"][1] - dm[i]] for i, a in enumerate(arms_d)]).T
    ax.barh(y, dm, xerr=err, color=C1, ecolor=INK, capsize=3, height=0.6)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(arms_d)
    ax.invert_yaxis()
    ax.set_xlabel("Δ nll_gtcoc vs baseline (nats/token, mean, bootstrap 95% CI)")
    ax.set_title(f"reference-CoC NLL, OOD-val n={len(common)}")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "delta_nll_gtcoc.png", dpi=150)
    plt.close(fig)

    # ---- records ----
    (out_dir / "config.json").write_text(json.dumps({
        "metric": "nll_gtcoc: mean per-token NLL (nats) of curated gt_coc, teacher-forced",
        "set": "ood_val", "arms": {a: ARMS[a][0] for a in args.arms},
        "n_common_clips": len(common), "n_boot": args.n_boot, "seed": args.seed,
        "gpu": gpus, "pending": pending,
        "source": "experiments/evaluation/nll_gtcoc_table.py",
    }, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(
        {"table": table, "pending": pending, "n_common_clips": len(common)}, indent=2))

    lines = [(f"reference-CoC NLL (nll_gtcoc, nats/token) on OOD-val, n={len(common)} "
              f"common clips, paired vs baseline"),
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
    if len(set(gpus.values())) > 1:
        lines += ["", "WARNING: mixed GPU architectures: " + json.dumps(gpus)]
    if pending:
        lines += ["", "pending:"] + [f"  {a}: {why}" for a, why in pending.items()]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"\n-> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
