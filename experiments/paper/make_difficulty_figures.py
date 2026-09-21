"""Reproduce static-difficulty analyses from stored closed-loop results, without inference.

Run from the repository root:
    MPLCONFIGDIR=/tmp/alpamayo-mpl .venv/bin/python experiments/paper/make_difficulty_figures.py

Reads shared artifacts; writes PNG/PDF/SVG, scene-level CSVs and provenance to ppt-fig.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, rankdata, spearmanr

REPO = Path(__file__).resolve().parents[2]
CL = Path("/home/cvlab21/project/chan/alpasim-runs")
FULL_RUN = Path("/mnt/nvme1n1/ad_vla/results/sangoh/alpasim_runs/eval2601_a1_5")
LP_RUN = Path("/mnt/nvme1n1/ad_vla/outputs/soowon/alpasim-runs/cl150_merged_lp_r50")
TEAL, BLUE, ORANGE = "#006F73", "#0072B2", "#D55E00"
INK, MUTED, GRID = "#142D3E", "#526575", "#DEE5EA"
SEED, N_BOOT = 0, 10000
PERCENTILES = np.array([0, 20, 40, 60, 80, 90, 100])
BAND_LABELS = ["0–20", "20–40", "40–60", "60–80", "80–90", "90–100"]
METRICS = ["score", "progress_clipped_rel", "offroad", "collision_at_fault"]
SOURCES = {}


def source_bytes(path):
    data = path.read_bytes()
    SOURCES[str(path)] = hashlib.sha256(data).hexdigest()
    return data


def read(path):
    return json.loads(source_bytes(path))


def load_run(directory, n_scenes, n_rollouts):
    data = read(directory / "aggregate/results-summary.json")
    groups, seen = {}, set()
    for row in data["rollouts"]:
        key = (row["clipgt_id"], row["rollout_id"])
        assert key not in seen, (directory, "duplicate rollout", key)
        seen.add(key)
        values = [row["score"], *[row["metrics"].get(k) for k in METRICS[1:]]]
        assert all(v is not None and np.isfinite(v) for v in values), (directory, key)
        assert 0 <= values[0] <= 1 and 0 <= values[1] <= 1
        assert all(v in (0, 1) for v in values[2:])
        groups.setdefault(key[0], []).append(values)
    assert len(groups) == n_scenes and all(len(v) == n_rollouts for v in groups.values())
    means = {sid: dict(zip(METRICS, np.mean(v, axis=0))) for sid, v in groups.items()}
    return means, data["score_criteria"]


def mean_ci(values):
    """Percentile bootstrap of independent scene units (paired deltas when supplied)."""
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1 and len(values) >= 2 and np.isfinite(values).all()
    rng = np.random.default_rng(SEED)
    draws = np.concatenate([
        values[rng.integers(0, len(values), (500, len(values)))].mean(axis=1)
        for _ in range(N_BOOT // 500)
    ])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"n": len(values), "mean": float(values.mean()), "lo": float(lo), "hi": float(hi)}


def corr_ci(x, y):
    """Resample paired scenes, recomputing average ranks including ties in every draw."""
    x, y = np.asarray(x), np.asarray(y)
    rng, draws = np.random.default_rng(SEED), []
    for _ in range(N_BOOT // 500):
        idx = rng.integers(0, len(x), (500, len(x)))
        a, b = rankdata(x[idx], axis=1), rankdata(y[idx], axis=1)
        a -= a.mean(axis=1, keepdims=True)
        b -= b.mean(axis=1, keepdims=True)
        denom = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
        assert (denom > 0).all(), "Constant bootstrap sample"
        draws.append((a * b).sum(axis=1) / denom)
    lo, hi = np.quantile(np.concatenate(draws), [0.025, 0.975])
    result = spearmanr(x, y)
    return {"n": len(x), "rho": float(result.statistic), "p": float(result.pvalue),
            "lo": float(lo), "hi": float(hi)}


def collect():
    features = read(REPO / "outputs/scene_difficulty/scene_feats_public2601.json")
    feature_by_id = {r["scene_id"]: r for r in features}
    assert len(features) == len(feature_by_id) == 913
    # Match the historical score exactly: population SD over all 913 scenes, epsilon 1e-9.
    normalization = {}
    hard_score = np.zeros(len(features))
    for key in ("v_mean", "yaw_total_deg"):
        values = np.array([r[key] for r in features])
        assert np.isfinite(values).all()
        normalization[key] = {"mean": float(values.mean()), "std_ddof0": float(values.std())}
        hard_score += (values - values.mean()) / (values.std() + 1e-9)
    hardness = dict(zip(feature_by_id, hard_score))
    percentile = dict(zip(feature_by_id, 100 * (rankdata(hard_score) - 1) / 912))
    cuts = np.percentile(hard_score, PERCENTILES)
    band = {sid: int(np.searchsorted(cuts[1:-1], h, side="right"))
            for sid, h in hardness.items()}
    main, criteria = load_run(CL / "m2601_merged_baseline", 150, 2)
    full, full_criteria = load_run(FULL_RUN, 913, 1)
    assert criteria == full_criteria
    assert set(full) == set(feature_by_id)
    assert set(main) == set(sorted(full)[:150])
    unseen = sorted(set(full) - set(main))
    assert len(unseen) == 763
    populations = {"development150": sorted(main), "heldout763": unseen}
    stats, scene_rows = {}, []
    for name, ids in populations.items():
        run = main if name == "development150" else full
        x = np.array([hardness[sid] for sid in ids])
        score = np.array([run[sid]["score"] for sid in ids])
        corr = corr_ci(x, 1 - score)
        bands = []
        for i, label in enumerate(BAND_LABELS):
            selected = [sid for sid in ids if band[sid] == i]
            bands.append({"band": label, "percentile_mid": float(PERCENTILES[i:i + 2].mean()),
                          **mean_ci([run[sid]["score"] for sid in selected])})
        stats[name] = {"correlation": corr, "bands": bands}
        for sid in ids:
            f = feature_by_id[sid]
            scene_rows.append({"scene_id": sid, "population": name,
                               "n_rollouts": 2 if name == "development150" else 1,
                               "static_difficulty": float(hardness[sid]),
                               "static_percentile": float(percentile[sid]),
                               "static_band": BAND_LABELS[band[sid]],
                               "v_mean_m_s": f["v_mean"],
                               "absolute_net_heading_change_deg": f["yaw_total_deg"],
                               "gt_path_m": f["gt_path_m"],
                               "baseline_difficulty": float(1 - run[sid]["score"]),
                               **run[sid]})

    # Easy/hard subsets use static scores alone; both exclude short GT trajectories.
    pool = [sid for sid in unseen if feature_by_id[sid]["gt_path_m"] >= 5]
    assert len(pool) == 755
    ordered = sorted(pool, key=lambda sid: (hardness[sid], sid))
    easy, hard = ordered[:100], ordered[-100:]
    saved_suite = list(csv.DictReader(io.StringIO(source_bytes(
        CL / "hard100/hard100_suite.csv").decode())))
    assert len(saved_suite) == 100 and {r["scene_id"] for r in saved_suite} == set(hard)
    assert {r["test_suite_id"] for r in saved_suite} == {"public_2601_hard100"}
    a, b = [full[s]["score"] for s in easy], [full[s]["score"] for s in hard]
    stats["selection"] = {"pool_n": len(pool), "easy100": mean_ci(a), "hard100": mean_ci(b),
                          "hard_minus_easy": float(np.mean(b) - np.mean(a)),
                          "mannwhitney_p": float(mannwhitneyu(a, b).pvalue),
                          "easy_scene_ids": sorted(easy), "hard_scene_ids": sorted(hard)}
    for row in scene_rows:
        row["selection"] = ("easy100" if row["scene_id"] in easy else
                            "hard100" if row["scene_id"] in hard else "")

    stats["metric_correlations"] = []
    x = [hardness[sid] for sid in unseen]
    for key, label, invert in [
        ("score", "1 − scene score", True),
        ("progress_clipped_rel", "1 − progress", True),
        ("offroad", "Offroad", False),
        ("collision_at_fault", "Collision@fault", False),
    ]:
        y = np.array([full[sid][key] for sid in unseen])
        stats["metric_correlations"].append({"metric": key, "label": label,
                                              **corr_ci(x, 1 - y if invert else y)})

    # Reproduce the prior analysis before creating new figures; do not silently mix result sets.
    old = read(REPO / "outputs/difficulty_strat/metrics.json")
    np.testing.assert_allclose(stats["heldout763"]["correlation"]["rho"],
                               -old["full_suite"]["spearman_unseen"], atol=1e-12)
    for name, key in [("development150", "baseline_score"), ("heldout763", "unseen_score")]:
        np.testing.assert_allclose([r["mean"] for r in stats[name]["bands"]],
                                   [r[key] for r in old["static_bands"]], atol=1e-12)
    for key in ("easy100", "hard100"):
        np.testing.assert_allclose(stats["selection"][key]["mean"],
                                   old["full_suite"][key + "_score"], atol=1e-12)

    # Explore methods on common static bins, avoiding bins defined by baseline outcome noise.
    method_runs = [
        ("Wanda", CL / "m2601_merged_slim_wanda_u40_v2"),
        ("LLM-Pruner", LP_RUN),
        ("Týr-the-Pruner", CL / "m2601_merged_slim_tyr_u40_r"),
        ("Traj", CL / "m2601_merged_slim_traj_u40_v2"),
        ("CoC", CL / "m2601_merged_slim_coc_u40_v2"),
        ("Dual", CL / "m2601_merged_slim_dual_u40_v2"),
        ("Dual + MLP93.75", CL / "m2601_merged_slim_dualexp_u40_em93p75"),
    ]
    method_rows, method_stats = [], []
    for label, directory in method_runs:
        run, arm_criteria = load_run(directory, 150, 2)
        assert set(run) == set(main) and arm_criteria == criteria
        for sid in sorted(main):
            group = min(band[sid] // 2, 2)
            delta = run[sid]["score"] - main[sid]["score"]
            method_rows.append({"scene_id": sid, "method": label, "static_group": group,
                                "baseline_score": main[sid]["score"],
                                "method_score": run[sid]["score"], "delta": delta})
        for group, label_group in enumerate(["0–40%", "40–80%", "80–100%"]):
            rows = [r for r in method_rows if r["method"] == label and r["static_group"] == group]
            method_stats.append({"method": label, "group": group, "band": label_group,
                                 "baseline_mean": float(np.mean(
                                     [r["baseline_score"] for r in rows])),
                                 "method_mean": float(np.mean([r["method_score"] for r in rows])),
                                 **mean_ci([r["delta"] for r in rows])})
    stats["method_by_static_difficulty"] = method_stats
    stats["definitions"] = {
        "suite": "public_2601", "static": "z(v_mean) + z(yaw_total_deg)",
        "heading": "yaw_total_deg = degrees(abs(unwrapped_heading[-1] - unwrapped_heading[0]))",
        "heading_note": "Absolute net heading change; cumulative absolute turning is yaw_abs_deg",
        "normalization": normalization, "z_epsilon": 1e-9, "normalization_n": 913,
        "percentile_plot": "100 * (average_rank(hard_score) - 1) / (913 - 1)",
        "bands": "np.percentile over all 913 scores; left-inclusive, last band right-inclusive",
        "band_cutpoints": cuts.tolist(), "measured_difficulty": "1 - baseline scene score",
        "bootstrap": {"unit": "scene", "draws": N_BOOT, "seed": SEED,
                      "interval": "95% percentile; paired resampling for correlations and deltas"},
        "score_criteria": criteria,
        "caveats": ["150 scenes informed feature choice; 763 scenes did not",
                    "Separate baseline runs: 2 rollouts vs 1; identical stored score criteria",
                    "Correlations retain GT paths <5m; easy/hard selection excludes them",
                    "Static score uses full offline GT trajectory, not an online difficulty signal",
                    "Per-method static strata are exploratory on the development150 scenes",
                    "CIs are pointwise, not simultaneous; fixed scenes and existing model recipes",
                    "Easy100/hard100 values here are from the 913-scene run, not hard100 reruns"],
    }
    for path in [Path(__file__), REPO / "experiments/head_analysis/extract_scene_feats.py",
                 REPO / "outputs/scene_difficulty/extract_scene_feats.py",
                 REPO / "experiments/head_analysis/analyze_difficulty_strat.py",
                 REPO / "experiments/head_analysis/make_hard_suite.py"]:
        source_bytes(path)
    stats["sources_sha256"] = SOURCES
    return stats, scene_rows, method_rows


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 12,
        "axes.titlesize": 12, "axes.titleweight": "bold", "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": MUTED, "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "axes.grid": False, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "mathtext.fontset": "dejavusans", "mathtext.default": "regular",
    })


def save(fig, out, name):
    # Matplotlib may instantiate tick labels outside the view limits without drawing them.
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            axis.set_ticks([v for v in axis.get_ticklocs() if lo <= v <= hi])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Check text placement before export; margins belong to the figure, not clipped plot text.
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        assert box.x0 >= -1 and box.x1 <= fig.bbox.width + 1, (name, text.get_text(), box)
        assert box.y0 >= -1 and box.y1 <= fig.bbox.height + 1, (name, text.get_text(), box)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{extension}", dpi=400)
    plt.close(fig)


def errorbar(ax, x, row, color, horizontal=False, marker="o", **kwargs):
    mean, lo, hi = row.get("mean", row.get("rho")), row["lo"], row["hi"]
    err = [[max(0, mean - lo)], [max(0, hi - mean)]]
    if horizontal:
        ax.errorbar(mean, x, xerr=err, fmt=marker, color=color, capsize=3, **kwargs)
    else:
        ax.errorbar(x, mean, yerr=err, fmt=marker, color=color, capsize=3, **kwargs)


def render(stats, scenes, out):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharex=True, sharey=True)
    for ax, name, label, color in zip(
        axes, ["development150", "heldout763"],
        ["Rule-development scenes (n = 150)", "Held-out scenes (n = 763)"], [BLUE, TEAL],
    ):
        rows = [r for r in scenes if r["population"] == name]
        ax.scatter([r["static_percentile"] for r in rows],
                   [r["baseline_difficulty"] for r in rows], color=color,
                   s=13, alpha=0.28, edgecolors="none", rasterized=True, clip_on=False)
        means = stats[name]["bands"]
        ax.plot([r["percentile_mid"] for r in means], [1 - r["mean"] for r in means],
                color=color, lw=2)
        for row in means:
            difficulty = {"mean": 1 - row["mean"], "lo": 1 - row["hi"], "hi": 1 - row["lo"]}
            errorbar(ax, row["percentile_mid"], difficulty, color, markersize=5)
        corr = stats[name]["correlation"]
        ax.text(0.04, 0.78, f"Spearman ρ = {corr['rho']:+.3f}\n"
                f"95% CI [{corr['lo']:+.3f}, {corr['hi']:+.3f}]", transform=ax.transAxes,
                fontsize=10, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88})
        ax.set(title=label, xlabel="Static difficulty percentile (%)",
               xlim=(-2, 102), ylim=(-0.03, 1.03))
        ax.set_xticks(np.arange(0, 101, 20))
        ax.grid(axis="y", color=GRID, lw=0.7)
    axes[0].set_ylabel("Baseline difficulty (1 − scene score)")
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.22, top=0.88, wspace=0.12)
    fig.text(0.08, 0.04, "Points: scenes. Lines: band means with 95% CI. "
             "Development: 2 rollouts/scene; held-out: 1.", color=MUTED, fontsize=9)
    save(fig, out, "fig_07_static_difficulty_correlation")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    for name, label, color, offset, marker in [
        ("development150", "Development (150 scenes)", BLUE, -0.06, "s"),
        ("heldout763", "Held-out (763 scenes)", TEAL, 0.06, "o"),
    ]:
        rows = stats[name]["bands"]
        x = np.arange(6) + offset
        ax.plot(x, [r["mean"] for r in rows], color=color, lw=1.8, label=label)
        for i, row in enumerate(rows):
            errorbar(ax, x[i], row, color, marker=marker, markersize=5)
    ax.set_xticks(np.arange(6), BAND_LABELS, fontsize=10)
    ax.set(xlabel="Static difficulty band (%)", ylabel="Baseline scene score ↑",
           ylim=(0, 1.06), title="Performance across static difficulty bands")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax = axes[1]
    for i, (key, color) in enumerate([("easy100", TEAL), ("hard100", ORANGE)]):
        row = stats["selection"][key]
        ax.bar(i, row["mean"], color=color, alpha=0.17, width=0.55)
        errorbar(ax, i, row, color, markersize=6, elinewidth=1.8)
        ax.text(i, row["hi"] + 0.025, f"{row['mean']:.3f}", ha="center", color=color,
                fontsize=13, fontweight="bold")
    ax.set_xticks([0, 1], ["Easy100", "Hard100"])
    ax.set(xlim=(-0.6, 1.6), ylim=(0, 1.06), title="Selection from the held-out pool")
    ax.set_xlabel("100 lowest / highest static scores")
    ax.grid(axis="y", color=GRID, lw=0.7)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.25, top=0.87, wspace=0.27)
    fig.text(0.065, 0.08, "95% scene-bootstrap CI. Easy/hard selection: "
             "755 eligible held-out scenes (GT path ≥ 5 m).", color=MUTED, fontsize=9)
    fig.text(0.065, 0.035, "Easy100 / Hard100 scores use the existing full-suite baseline run "
             "(1 rollout/scene).", color=MUTED, fontsize=9)
    save(fig, out, "fig_08_static_difficulty_validation")

    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    rows = stats["metric_correlations"]
    for i, row in enumerate(rows):
        color = TEAL if i < 3 else MUTED
        errorbar(ax, i, row, color, horizontal=True, markersize=7, elinewidth=2)
        ax.text(0.44, i, f"{row['rho']:+.3f}", va="center", ha="right", fontsize=11, color=color)
    ax.axvline(0, color=MUTED, lw=1, ls="--")
    ax.set_yticks(range(4), [r["label"] for r in rows])
    ax.set(xlim=(-0.1, 0.46), ylim=(3.6, -0.6),
           xlabel="Spearman ρ with static difficulty",
           title="Which baseline outcomes track static difficulty?")
    ax.grid(axis="x", color=GRID, lw=0.7)
    fig.subplots_adjust(left=0.25, right=0.97, bottom=0.29, top=0.85)
    fig.text(0.25, 0.085, "763 held-out scenes; 95% scene-bootstrap CI.", fontsize=9, color=MUTED)
    fig.text(0.25, 0.035, "Positive ρ: higher static difficulty, worse outcome.",
             fontsize=9, color=MUTED)
    save(fig, out, "fig_09_static_difficulty_metrics")

    rows = stats["method_by_static_difficulty"]
    methods = list(dict.fromkeys(r["method"] for r in rows))
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.8), sharex=True, sharey=True)
    low = min(r["lo"] for r in rows) - 0.035
    high = max(r["hi"] for r in rows) + 0.035
    for group, ax in enumerate(axes):
        selected = [r for r in rows if r["group"] == group]
        for i, row in enumerate(selected):
            color = TEAL if row["method"] == "Dual" else (
                ORANGE if row["method"] == "Dual + MLP93.75" else MUTED)
            errorbar(ax, i, row, color, horizontal=True, markersize=6, elinewidth=1.7)
        row = selected[0]
        ax.set_title(f"{row['band']} ({row['n']} scenes)")
        ax.axvline(0, color=MUTED, lw=1, ls="--")
        ax.set(xlim=(low, high), ylim=(len(methods) - 0.4, -0.6))
        ax.grid(axis="x", color=GRID, lw=0.7)
    axes[0].set_yticks(range(len(methods)), methods)
    axes[1].set_xlabel("Scene score difference vs Dense ↑")
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.27, top=0.86, wspace=0.12)
    fig.text(0.18, 0.10, "Main 150 scenes, 2 rollouts/scene; fixed static percentile bins; "
             "paired 95% scene-bootstrap CI.", color=MUTED, fontsize=9)
    fig.text(0.18, 0.05, "Exploratory on rule-development scenes. Dual + MLP93.75 "
             "removes 39.4% of the full model.", color=MUTED, fontsize=9)
    save(fig, out, "fig_10_pruning_by_static_difficulty")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "ppt-fig")
    args = parser.parse_args()
    stats, scene_rows, method_rows = collect()
    args.out.mkdir(parents=True, exist_ok=True)
    render(stats, scene_rows, args.out)
    write_csv(args.out / "difficulty_scene_data.csv", scene_rows)
    write_csv(args.out / "difficulty_method_scene_data.csv", method_rows)
    write_csv(args.out / "difficulty_method_summary.csv", stats["method_by_static_difficulty"])
    (args.out / "difficulty_analysis_data.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"heldout_correlation": stats["heldout763"]["correlation"],
                      "easy100": stats["selection"]["easy100"],
                      "hard100": stats["selection"]["hard100"],
                      "metric_correlations": stats["metric_correlations"]}, indent=2))
    print(f"Wrote four figures (PNG/PDF/SVG), three CSVs and provenance to {args.out}")


if __name__ == "__main__":
    main()
