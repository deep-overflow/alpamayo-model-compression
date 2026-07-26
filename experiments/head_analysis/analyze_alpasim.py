"""Aggregate alpasim closed-loop matrix runs into outputs/alpasim_eval.

Reads each config's <runs-root>/<prefix><config>/aggregate/results-summary.json plus the
per-rollout ASL logs (CoC text via pickled driver debug info), computes per-config
aggregates, per-scene paired deltas vs baseline (bootstrap CI + Wilcoxon signed-rank),
and CoC-degeneracy statistics, then writes config.json / metrics.json / summary.txt /
plots/*.png per the repo output convention.

Runs with the alpasim repo's venv (needs alpasim_utils, alpasim_grpc, numpy, matplotlib):

    cd /home/cvlab21/project/chan/alpasim && uv run python \
        /home/cvlab21/project/chan/alpamayo-model-compression/experiments/head_analysis/analyze_alpasim.py \
        --runs-root /home/cvlab21/project/chan/alpasim-runs \
        --out /home/cvlab21/project/chan/alpamayo-model-compression/outputs/alpasim_eval
"""

import argparse
import asyncio
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG = "#FAF9F5"
INK = "#29261B"
MUTED = "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": INK,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

CONFIGS = ["baseline", "slim_cocsafe_r20", "slim_cocsafe_r30", "slim_integrated_mag"]
COLORS = {c: col for c, col in zip(CONFIGS, [C1, C2, C3, C4])}
HEADLINE = ["score", "passed", "collision_at_fault", "collision_any", "offroad",
            "wrong_lane", "progress_clipped_rel", "progress_rel", "dist_to_gt_trajectory"]


def load_rollouts(run_dir):
    """results-summary.json -> list of per-rollout dicts (scene, score, metrics)."""
    d = json.loads((run_dir / "aggregate" / "results-summary.json").read_text())
    rows = []
    for r in d["rollouts"]:
        rows.append({
            "scene": r["clipgt_id"],
            "rollout_id": r["rollout_id"],
            "status": r["status"],
            "passed": float(bool(r["passed"])),
            "score": float(r["score"]) if r["score"] is not None else np.nan,
            **{k: r["metrics"].get(k) for k in HEADLINE if k in r["metrics"]},
        })
    return rows


async def _read_coc(asl_path):
    from alpasim_utils.logs import async_read_pb_log

    texts = []
    async for entry in async_read_pb_log(str(asl_path)):
        if entry.HasField("driver_return"):
            blob = entry.driver_return.debug_info.unstructured_debug_info
            if blob:
                try:
                    dbg = pickle.loads(blob)
                    t = dbg.get("reasoning_text")
                    if t:
                        texts.append(str(t))
                except Exception:
                    pass
    return texts


def coc_stats(texts):
    """Degeneracy heuristics over one rollout's per-step CoC strings."""
    if not texts:
        return {"n_steps": 0, "degenerate_frac": np.nan, "empty_frac": np.nan,
                "soup_frac": np.nan, "mean_len": np.nan,
                "mean_unique_ratio": np.nan, "mean_nonascii": np.nan}
    empty, soup, lens, uniq, nonascii = [], [], [], [], []
    for t in texts:
        # driver logs the CoC list repr, e.g. "['Keep lane ...']"; strip it
        s = t.strip()
        if s.startswith("['") and s.endswith("']"):
            s = s[2:-2]
        s = s.strip()
        is_empty = len(s) == 0
        words = s.split()
        ur = len(set(words)) / max(len(words), 1)
        na = sum(ord(ch) > 127 for ch in s) / max(len(s), 1)
        # soup: garbled generation (heavy non-ascii, repetitive, or runaway length)
        is_soup = (not is_empty) and (na > 0.05 or ur < 0.5 or len(s) > 300)
        empty.append(is_empty)
        soup.append(is_soup)
        if not is_empty:
            lens.append(len(s))
            uniq.append(ur)
            nonascii.append(na)
    return {"n_steps": len(texts),
            "empty_frac": float(np.mean(empty)),
            "soup_frac": float(np.mean(soup)),
            "degenerate_frac": float(np.mean(np.asarray(empty) | np.asarray(soup))),
            "mean_len": float(np.mean(lens)) if lens else np.nan,
            "mean_unique_ratio": float(np.mean(uniq)) if uniq else np.nan,
            "mean_nonascii": float(np.mean(nonascii)) if nonascii else np.nan}


def per_scene(rows, key):
    """scene -> mean of key over that scene's rollouts."""
    out = {}
    for r in rows:
        out.setdefault(r["scene"], []).append(r[key])
    return {s: float(np.nanmean(v)) for s, v in out.items()}


def boot_ci(x, n_boot=10000, seed=0, alpha=0.05):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def wilcoxon_p(deltas):
    try:
        from scipy.stats import wilcoxon

        d = np.asarray(deltas, dtype=float)
        d = d[~np.isnan(d)]
        d_nz = d[d != 0]
        if len(d_nz) < 5:
            return None
        return float(wilcoxon(d_nz).pvalue)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--prefix", default="matrix_")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-coc", action="store_true")
    args = ap.parse_args()

    out = args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)

    data, coc = {}, {}
    for cfg in CONFIGS:
        run_dir = args.runs_root / f"{args.prefix}{cfg}"
        data[cfg] = load_rollouts(run_dir)
        if not args.skip_coc:
            per_rollout = []
            for r in data[cfg]:
                asl = run_dir / "rollouts" / r["scene"] / r["rollout_id"] / "rollout.asl"
                texts = asyncio.run(_read_coc(asl)) if asl.exists() else []
                st = coc_stats(texts)
                st.update(scene=r["scene"], rollout_id=r["rollout_id"], score=r["score"])
                per_rollout.append(st)
            coc[cfg] = per_rollout

    # scene sets must match across configs
    scene_sets = {cfg: set(per_scene(rows, "score")) for cfg, rows in data.items()}
    common = set.intersection(*scene_sets.values())
    for cfg, s in scene_sets.items():
        if s != common:
            print(f"WARNING: {cfg} scene set differs (n={len(s)} vs common {len(common)})")
    scenes = sorted(common)

    metrics = {"n_scenes": len(scenes), "configs": {}, "paired_vs_baseline": {}, "coc": {}}
    base_scene = {k: per_scene(data["baseline"], k) for k in ["score", "passed"]}

    for cfg, rows in data.items():
        agg = {}
        for k in HEADLINE:
            vals = [r[k] for r in rows if r.get(k) is not None]
            agg[k] = float(np.nanmean(vals)) if vals else None
        sc = per_scene(rows, "score")
        mean, lo, hi = boot_ci([sc[s] for s in scenes])
        agg["score_scene_mean"], agg["score_ci_lo"], agg["score_ci_hi"] = mean, lo, hi
        # repeat noise: |difference| between the two rollouts of the same scene.
        # No seed control exists in alpasim and closed-loop diverges after the first
        # differing action, so this is the noise floor any per-scene delta sits on.
        by_scene = {}
        for r in rows:
            by_scene.setdefault(r["scene"], []).append(r["score"])
        rep = [abs(v[0] - v[1]) for v in by_scene.values() if len(v) >= 2]
        agg["repeat_abs_diff_mean"] = float(np.nanmean(rep)) if rep else None
        agg["repeat_abs_diff_max"] = float(np.nanmax(rep)) if rep else None
        agg["n_rollouts"] = len(rows)
        agg["n_failed_rollouts"] = sum(1 for r in rows if r["status"] != "pass" and not r["passed"])
        metrics["configs"][cfg] = agg

        if cfg != "baseline":
            sc_cfg = per_scene(rows, "score")
            deltas = [sc_cfg[s] - base_scene["score"][s] for s in scenes]
            dmean, dlo, dhi = boot_ci(deltas)
            metrics["paired_vs_baseline"][cfg] = {
                "d_score_mean": dmean, "d_score_ci_lo": dlo, "d_score_ci_hi": dhi,
                "per_scene_delta": {s: sc_cfg[s] - base_scene["score"][s] for s in scenes},
                "wilcoxon_p": wilcoxon_p(deltas),
                "wins": int(np.sum(np.asarray(deltas) > 0)),
                "losses": int(np.sum(np.asarray(deltas) < 0)),
                "ties": int(np.sum(np.asarray(deltas) == 0)),
            }

        if cfg in coc:
            rs = coc[cfg]
            metrics["coc"][cfg] = {
                "mean_degenerate_frac": float(np.nanmean([r["degenerate_frac"] for r in rs])),
                "mean_empty_frac": float(np.nanmean([r["empty_frac"] for r in rs])),
                "mean_soup_frac": float(np.nanmean([r["soup_frac"] for r in rs])),
                "mean_len": float(np.nanmean([r["mean_len"] for r in rs])),
                "mean_unique_ratio": float(np.nanmean([r["mean_unique_ratio"] for r in rs])),
                "mean_nonascii": float(np.nanmean([r["mean_nonascii"] for r in rs])),
            }

    # ---- plots
    # 1. scene score by config
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, cfg in enumerate(CONFIGS):
        a = metrics["configs"][cfg]
        ax.bar(i, a["score_scene_mean"], color=COLORS[cfg], width=0.62)
        ax.errorbar(i, a["score_scene_mean"],
                    yerr=[[a["score_scene_mean"] - a["score_ci_lo"]],
                          [a["score_ci_hi"] - a["score_scene_mean"]]],
                    color=INK, capsize=4, lw=1.2)
    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels([c.replace("slim_", "") for c in CONFIGS], rotation=12)
    ax.set_ylabel("mean scene score (95% CI over scenes)")
    ax.set_title(f"Closed-loop scene score, {len(scenes)} scenes x 2 rollouts")
    fig.tight_layout()
    fig.savefig(out / "plots" / "scene_score.png", dpi=150)

    # 2. paired per-scene deltas
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, cfg in enumerate([c for c in CONFIGS if c != "baseline"]):
        sc_cfg = per_scene(data[cfg], "score")
        d = [sc_cfg[s] - base_scene["score"][s] for s in scenes]
        x = np.full(len(d), i) + np.random.default_rng(1).uniform(-0.12, 0.12, len(d))
        ax.scatter(x, d, s=18, color=COLORS[cfg], alpha=0.75)
        ax.hlines(np.mean(d), i - 0.25, i + 0.25, color=INK, lw=2)
    ax.axhline(0, color=MUTED, lw=1, ls="--")
    ax.set_xticks(range(3))
    ax.set_xticklabels([c.replace("slim_", "") for c in CONFIGS if c != "baseline"])
    ax.set_ylabel("Δ scene score vs baseline (per scene)")
    ax.set_title("Paired per-scene deltas")
    fig.tight_layout()
    fig.savefig(out / "plots" / "paired_deltas.png", dpi=150)

    # 3. incident rates
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    keys = ["collision_at_fault", "collision_any", "offroad", "wrong_lane"]
    w = 0.2
    for i, cfg in enumerate(CONFIGS):
        vals = [metrics["configs"][cfg][k] for k in keys]
        ax.bar(np.arange(len(keys)) + (i - 1.5) * w, vals, width=w,
               color=COLORS[cfg], label=cfg.replace("slim_", ""))
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=10)
    ax.set_ylabel("rate over rollouts")
    ax.set_title("Incident rates")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "plots" / "incident_rates.png", dpi=150)

    # 4. CoC degeneracy vs score
    if coc:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for cfg in CONFIGS:
            rs = coc.get(cfg, [])
            xs = [r["degenerate_frac"] for r in rs]
            ys = [r["score"] for r in rs]
            ax.scatter(xs, ys, s=16, alpha=0.7, color=COLORS[cfg],
                       label=cfg.replace("slim_", ""))
        ax.set_xlabel("degenerate-CoC step fraction (per rollout)")
        ax.set_ylabel("scene score")
        ax.set_title("CoC health vs closed-loop outcome")
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "plots" / "coc_vs_score.png", dpi=150)

    # ---- write outputs
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "config.json").write_text(json.dumps({
        "runs_root": str(args.runs_root), "prefix": args.prefix,
        "configs": CONFIGS, "n_scenes": len(scenes), "scenes": scenes,
    }, indent=2))

    lines = [f"alpasim closed-loop eval — {len(scenes)} scenes x 2 rollouts, 4 configs", ""]
    hdr = f"{'config':22s} {'score':>7s} {'pass%':>6s} {'col@f':>6s} {'offrd':>6s} {'prog':>6s} {'d2gt':>6s}"
    lines += [hdr, "-" * len(hdr)]
    for cfg in CONFIGS:
        a = metrics["configs"][cfg]
        lines.append(f"{cfg:22s} {a['score_scene_mean']:7.3f} {a['passed']*100:6.1f} "
                     f"{a['collision_at_fault']:6.3f} {a['offroad']:6.3f} "
                     f"{a['progress_clipped_rel']:6.3f} {a['dist_to_gt_trajectory']:6.2f}")
    lines.append("")
    lines.append("repeat noise (|score diff| between the 2 rollouts of a scene):")
    for cfg in CONFIGS:
        a = metrics["configs"][cfg]
        if a.get("repeat_abs_diff_mean") is not None:
            lines.append(f"  {cfg:22s} mean {a['repeat_abs_diff_mean']:.3f} "
                         f"max {a['repeat_abs_diff_max']:.3f}")
    lines.append("")
    for cfg, p in metrics["paired_vs_baseline"].items():
        wp = f"{p['wilcoxon_p']:.4f}" if p["wilcoxon_p"] is not None else "n/a"
        lines.append(f"{cfg}: d_score {p['d_score_mean']:+.4f} "
                     f"[{p['d_score_ci_lo']:+.4f}, {p['d_score_ci_hi']:+.4f}] "
                     f"wilcoxon_p {wp}  W/L/T {p['wins']}/{p['losses']}/{p['ties']}")
    if metrics["coc"]:
        lines.append("")
        for cfg in CONFIGS:
            c = metrics["coc"].get(cfg)
            if c:
                lines.append(f"{cfg}: CoC degen {c['mean_degenerate_frac']:.3f} "
                             f"(empty {c['mean_empty_frac']:.3f} soup {c['mean_soup_frac']:.3f}) "
                             f"len {c['mean_len']:.0f} uniq {c['mean_unique_ratio']:.2f} "
                             f"nonascii {c['mean_nonascii']:.3f}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
