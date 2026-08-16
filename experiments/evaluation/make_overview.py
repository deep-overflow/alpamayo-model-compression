"""One axis for every compression arm measured in this project.

Three tracks grew separately -- pruning criteria, quantization budgets, and their
composition -- and each has its own report. This puts all of them side by side on the two
measurements that matter (open-loop minADE on official test 500, closed-loop scene score
on alpasim public_2601) against the storage each one buys.

Sizes: a pruned arm's checkpoint is real (parameters are gone). A quantized arm's is
*projected* from the bit allocation -- fake quant keeps bf16, so the file on disk is full
size and nothing about runtime changes. The two columns are not the same kind of number
and the report says so.

Writes outputs/overview_<set>/{metrics.json,summary.txt,plots/*.png}.

Usage:
  python experiments/evaluation/make_overview.py --set test
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(Path(__file__).parent))

import eval_lib as el
from analyze_baseline import load_rows

FULL_PARAMS = 11_078_526_194
FULL_GB = FULL_PARAMS * 2 / 1e9
OUT = REPO / "outputs"

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

# label, open-loop exp dir, rows tag, closed-loop config, slim dir, quant spec, colour
ARMS = [
    ("baseline", "baseline_ada_{s}", "baseline", "baseline", None, None, MUTED),
    ("w8_all", "w8_all_{s}", "uniform_w8_all", None, None, ("uniform_w8_all", None), C1),
    ("w4_all", "w4_all_{s}", "uniform_w4_all", None, None, ("uniform_w4_all", None), C1),
    ("traj", "traj_u40_v2_{s}", "slim_traj_u40_v2", "slim_traj_u40_v2",
     "slim_traj_u40_v2", None, C4),
    ("coc", "coc_u40_v2_{s}", "slim_coc_u40_v2", "slim_coc_u40_v2",
     "slim_coc_u40_v2", None, C4),
    ("j", "j_u40_v2_{s}", "slim_j_u40_v2", "slim_j_u40_v2", "slim_j_u40_v2", None, C4),
    ("dual", "dual_u40_v2_{s}", "slim_dual_u40_v2", "slim_dual_u40_v2",
     "slim_dual_u40_v2", None, C4),
    ("jtraj", "jtraj_u40_v2_{s}", "slim_jtraj_u40_v2", "slim_jtraj_u40_v2",
     "slim_jtraj_u40_v2", None, C4),
    ("dual+w8", "dual_u40_w8_{s}", "slim_dual_u40_v2_w8_all_dual_u40", "slim_dual_u40_w8",
     "slim_dual_u40_v2", ("w8_all_dual_u40", "slim_dual_u40_v2"), C2),
    ("dual+w4", "dual_u40_w4_{s}", "slim_dual_u40_v2_w4_all_dual_u40", "slim_dual_u40_w4",
     "slim_dual_u40_v2", ("w4_all_dual_u40", "slim_dual_u40_v2"), C2),
]
CL_RUNS = ("alpasim_singles_2601", "alpasim_compose_2601")


def storage_gb(slim_dir, quant):
    params = (json.loads((OUT / slim_dir / "config.json").read_text())["params"]["slim"]
              if slim_dir else FULL_PARAMS)
    if not quant:
        return params * 2 / 1e9, bool(slim_dir)
    spec, _ = quant
    m = json.loads((OUT / "quant_specs" / f"{spec}.json").read_text())
    return m["projected_pool_gb"] + (params - m["pool_params"]) * 2 / 1e9, True


def closed_loop():
    cfgs, paired = {}, {}
    for r in CL_RUNS:
        p = OUT / r / "metrics.json"
        if p.exists():
            m = json.loads(p.read_text())
            cfgs.update(m["configs"])
            paired.update(m["paired_vs_baseline"])
    return cfgs, paired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="which", default="test")
    args = ap.parse_args()
    w = args.which

    base = {r["clip_id"]: r["minADE_rollout"]
            for r in load_rows(OUT / f"baseline_ada_{w}", "baseline")}
    cl_cfgs, cl_paired = closed_loop()
    rows = []
    for label, exp, tag, cl, slim, quant, colour in ARMS:
        gb, projected = storage_gb(slim, quant)
        rec = {"arm": label, "gb": gb, "ratio": FULL_GB / gb,
               "projected": projected and bool(quant), "colour": colour}
        r = load_rows(OUT / exp.format(s=w), tag)
        if r:
            cur = {x["clip_id"]: x["minADE_rollout"] for x in r}
            ids = sorted(set(base) & set(cur))
            rec["ol_mean"] = float(np.mean([cur[i] for i in ids]))
            rec["ol_n"] = len(ids)
            if label != "baseline":
                d = np.array([cur[i] - base[i] for i in ids])
                m, lo, hi = el.paired_bootstrap_ci(d)
                rec["ol_delta"] = {"mean": m, "ci": [lo, hi], "sig": not (lo <= 0 <= hi)}
        if cl and cl in cl_cfgs:
            rec["cl_score"] = cl_cfgs[cl]["score"]
            rec["cl_pass"] = cl_cfgs[cl]["passed"]
            rec["cl_collision"] = cl_cfgs[cl]["collision_at_fault"]
            p = cl_paired.get(cl)
            if p:
                rec["cl_delta"] = {
                    "mean": p["d_score_mean"], "ci": [p["d_score_ci_lo"], p["d_score_ci_hi"]],
                    "sig": not (p["d_score_ci_lo"] <= 0 <= p["d_score_ci_hi"]),
                    "wlt": [p["wins"], p["losses"], p["ties"]]}
        rows.append(rec)

    d = OUT / f"overview_{w}"
    (d / "plots").mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps({"set": w, "arms": rows}, indent=2))

    head = (f"{'arm':10s} {'GB':>6} {'ratio':>6} {'OL minADE':>10} {'OL delta':>10} "
            f"{'CL score':>9} {'CL delta':>10}")
    title = f"compression overview, open-loop set={w}, closed-loop public_2601 150 scenes"
    lines = [title, head]
    for r in rows:
        ol = f"{r.get('ol_mean', float('nan')):10.4f}"
        od = (f"{r['ol_delta']['mean']:+10.4f}" if "ol_delta" in r else " " * 10)
        cs = (f"{r['cl_score']:9.3f}" if "cl_score" in r else f"{'-':>9}")
        cd = (f"{r['cl_delta']['mean']:+10.4f}" if "cl_delta" in r else " " * 10)
        lines.append(f"{r['arm']:10s} {r['gb']:6.2f} {r['ratio']:5.2f}x {ol} {od} {cs} {cd}")
    (d / "summary.txt").write_text("\n".join(lines) + "\n")
    frontier_plot(rows, w, d / "plots" / "frontier.png")
    print("\n".join(lines))
    print("\nsaved ->", d)


def frontier_plot(rows, which, path):
    """Storage against each of the two measurements, with the CIs that decide them."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    ax = axes[0]
    for r in rows:
        if "ol_delta" not in r and r["arm"] != "baseline":
            continue
        m = r.get("ol_delta", {}).get("mean", 0.0)
        ci = r.get("ol_delta", {}).get("ci", [0.0, 0.0])
        ax.errorbar([r["gb"]], [m], yerr=[[m - ci[0]], [ci[1] - m]], fmt="o",
                    color=r["colour"], ecolor=MUTED, capsize=3, ms=7)
        ax.annotate(f"  {r['arm']}", (r["gb"], m), fontsize=8.5, color=INK, va="center")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.invert_xaxis()
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xlabel("checkpoint size (GB, projected for quantized arms)")
    ax.set_ylabel("open-loop paired dminADE@8 (m, symlog)")
    ax.set_title(f"open loop ({which} 500)")

    ax = axes[1]
    for r in rows:
        if "cl_delta" not in r and r["arm"] != "baseline":
            continue
        m = r.get("cl_delta", {}).get("mean", 0.0)
        ci = r.get("cl_delta", {}).get("ci", [0.0, 0.0])
        ax.errorbar([r["gb"]], [m], yerr=[[m - ci[0]], [ci[1] - m]], fmt="o",
                    color=r["colour"], ecolor=MUTED, capsize=3, ms=7)
        ax.annotate(f"  {r['arm']}", (r["gb"], m), fontsize=8.5, color=INK, va="center")
    ax.axhline(0, color=MUTED, lw=0.8)
    # delta_min(80% power, alpha=.05) = 2.80*sigma/sqrt(N) = 0.080 at sigma~0.32, N=150.
    # This is the effect the design is *powered* for, not a detection floor: an arm inside
    # the band can still have a CI that excludes zero (dual does), it just means the design
    # was not guaranteed to find it. Differences *between* arms inside the band are the
    # ones that must not be ranked.
    ax.axhspan(-0.080, 0.080, color=MUTED, alpha=0.10)
    ax.invert_xaxis()
    ax.set_xlabel("checkpoint size (GB, projected for quantized arms)")
    ax.set_ylabel("closed-loop paired dscene score")
    ax.set_title("closed loop (public_2601, 150 scenes; shaded = 80% power band)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
