"""t0 camera blackout vs the unmodified baseline, open loop (test500 + OOD-val262).

Plan and gates: plans/2026-10-18_camera-blackout-openloop.md.

Each clip has one camera's t0 frame replaced with a black image (`run_baseline.py
--blackout`). Sampling seeds are unchanged, so every metric is a paired per-clip delta
against the stored Ada baseline (`baseline_pred_*`):

  G0  the unmodified path, re-run now on the first 10 test500 clips (`blackout_g0`), is
      bitwise identical to `baseline_pred_test` (ade_rollout_k and gen_coc). If it is not,
      the stored baseline cannot serve as the reference.
  G1  the mean delta of minADE@6 (6.4 s) has a bootstrap CI excluding 0, per set.

The per-camera breakdown is exploratory (~60-130 clips per camera).

Usage:
  python experiments/evaluation/analyze_blackout.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import trim_mean, wilcoxon

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs"

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

SETS = {"test500": ("baseline_pred_test", "blackout_test"),
        "oodval262": ("baseline_pred_oodval", "blackout_oodval")}
# scale reference: the shipped 24% pruned arm, same Ada path, same clips and seeds
REF = {"test500": "dual_u40_v2_pred_test", "oodval262": "dual_u40_v2_pred_oodval"}
REF_TAG = "slim_dual_u40_v2"
CAMS = {0: "cross_left_120", 1: "front_wide_120", 2: "cross_right_120", 6: "front_tele_30"}
BUCKETS = ["decel_stop", "turn", "accel", "cruise"]
HORIZONS = {"1.6s": "_h16", "3.2s": "_h32", "6.4s": ""}
K = 6            # minADE@6: the first 6 of the 8 stored samples (seeds base+k)
N_BOOT = 200_000


def load_rows(exp_dir, tag):
    """Merge every shard file for one model; last write per clip wins."""
    rows = {}
    for p in sorted(exp_dir.glob(f"{tag}_s*of*.json")):
        for r in json.loads(p.read_text()):
            rows[r["clip_id"]] = r
    return rows


def metric(r, kind, h):
    return float(min(r[f"{kind}_rollout_k{HORIZONS[h]}"][:K]))


def boot_ci(d, n_boot=N_BOOT, seed=0, chunk=10_000):
    """Bootstrap CI of the mean, chunked so 200k resamples fit in memory."""
    d = np.asarray(d, dtype=float)
    rng = np.random.RandomState(seed)
    means = []
    for s in range(0, n_boot, chunk):
        idx = rng.randint(0, len(d), size=(min(chunk, n_boot - s), len(d)))
        means.append(d[idx].mean(1))
    lo, hi = np.percentile(np.concatenate(means), [2.5, 97.5])
    return float(lo), float(hi)


def paired(base, bk):
    """Paired delta (blackout - baseline) summary."""
    b, x = np.asarray(base, float), np.asarray(bk, float)
    d = x - b
    lo, hi = boot_ci(d)
    return {"n": len(d), "base_mean": float(b.mean()), "blackout_mean": float(x.mean()),
            "delta_mean": float(d.mean()), "ci": [lo, hi],
            "delta_median": float(np.median(d)),
            "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
            "frac_worse": float(np.mean(d > 0)), "frac_better": float(np.mean(d < 0)),
            "frac_abs_gt_0.05": float(np.mean(np.abs(d) > 0.05))}


def tail(d, n_boot=20_000, seed=0):
    """How much of the mean comes from a few clips: top-5 share and a 10% trimmed mean."""
    d = np.asarray(d, float)
    top = np.argsort(-np.abs(d))[:5]
    rng = np.random.RandomState(seed)
    tm = [trim_mean(d[rng.randint(0, len(d), len(d))], 0.1) for _ in range(n_boot)]
    lo, hi = np.percentile(tm, [2.5, 97.5])
    return {"top5_share_of_sum": float(d[top].sum() / d.sum()) if d.sum() != 0 else None,
            "top5_deltas": [float(x) for x in d[top]],
            "trim10_mean": float(trim_mean(d, 0.1)), "trim10_ci": [float(lo), float(hi)]}


def check_g0():
    base = load_rows(OUT / "baseline_pred_test", "baseline")
    g0 = load_rows(OUT / "blackout_g0", "baseline")
    if not g0:
        return {"status": "MISSING"}
    same_ade = [g0[c]["ade_rollout_k"] == base[c]["ade_rollout_k"] for c in g0]
    same_coc = [g0[c]["gen_coc"] == base[c]["gen_coc"] for c in g0]
    ok = all(same_ade) and all(same_coc)
    return {"status": "PASS" if ok else "FAIL", "n": len(g0),
            "ade_identical": int(sum(same_ade)), "coc_identical": int(sum(same_coc))}


def analyse_set(base, bk, ref):
    common = [c for c in bk if c in base]
    B = [base[c] for c in common]
    X = [bk[c] for c in common]
    res = {"n_blackout_rows": len(bk), "n_paired": len(common), "horizons": {}}
    for h in HORIZONS:
        res["horizons"][h] = {
            kind: paired([metric(r, kind, h) for r in B], [metric(r, kind, h) for r in X])
            for kind in ("ade", "fde")}
    res["tail"] = {h: tail([metric(x, "ade", h) - metric(b, "ade", h) for b, x in zip(B, X)])
                   for h in ("1.6s", "6.4s")}
    # did the trajectory move because the reasoning text changed, or with the text intact?
    same = [b["gen_coc"] == x["gen_coc"] for b, x in zip(B, X)]
    res["by_coc_change"] = {}
    for label, flag in (("coc_identical", True), ("coc_changed", False)):
        idx = [i for i, s in enumerate(same) if s == flag]
        if len(idx) >= 2:
            res["by_coc_change"][label] = {
                h: paired([metric(B[i], "ade", h) for i in idx], [metric(X[i], "ade", h) for i in idx])
                for h in ("1.6s", "6.4s")}
    rc = [c for c in common if c in ref]
    if rc:
        res["reference_dual"] = {
            "n": len(rc),
            **{h: paired([metric(base[c], "ade", h) for c in rc],
                         [metric(ref[c], "ade", h) for c in rc]) for h in HORIZONS}}
    # which camera was blacked out
    res["by_camera"] = {}
    for cam, name in CAMS.items():
        idx = [i for i, r in enumerate(X) if r["blackout_cam"] == cam]
        if len(idx) < 2:
            continue
        res["by_camera"][name] = {
            h: paired([metric(B[i], "ade", h) for i in idx], [metric(X[i], "ade", h) for i in idx])
            for h in ("1.6s", "6.4s")}
    res["by_bucket"] = {}
    for bu in BUCKETS:
        idx = [i for i, r in enumerate(B) if r["bucket"] == bu]
        if len(idx) >= 2:
            res["by_bucket"][bu] = paired([metric(B[i], "ade", "6.4s") for i in idx],
                                          [metric(X[i], "ade", "6.4s") for i in idx])
    # reasoning: degeneracy and whether the text changed at all
    db = np.array([r["coc_degenerate"] for r in B], float)
    dx = np.array([r["coc_degenerate"] for r in X], float)
    lo, hi = boot_ci(dx - db)
    res["coc"] = {"degenerate_base": float(db.mean()), "degenerate_blackout": float(dx.mean()),
                  "degenerate_delta": float((dx - db).mean()), "degenerate_delta_ci": [lo, hi],
                  "identical_text": float(np.mean([b["gen_coc"] == x["gen_coc"]
                                                   for b, x in zip(B, X)]))}
    by_cam_coc = {}
    for cam, name in CAMS.items():
        idx = [i for i, r in enumerate(X) if r["blackout_cam"] == cam]
        if idx:
            by_cam_coc[name] = float(np.mean([B[i]["gen_coc"] == X[i]["gen_coc"] for i in idx]))
    res["coc"]["identical_text_by_camera"] = by_cam_coc
    d = res["horizons"]["6.4s"]["ade"]
    res["G1"] = "PASS" if d["ci"][0] > 0 else ("FAIL" if d["ci"][0] <= 0 <= d["ci"][1] else
                                                 "NEGATIVE")
    return res, B, X


def plot_horizons(res, out):
    fig, axes = plt.subplots(1, len(res), figsize=(6 * len(res), 4.2), squeeze=False)
    for ax, (s, r) in zip(axes[0], res.items()):
        x = np.arange(len(HORIZONS))
        series = [(-0.15, r["horizons"], "ade", C1, "blackout minADE@6"),
                  (0.0, r["horizons"], "fde", C4, "blackout minFDE@6")]
        if "reference_dual" in r:
            series.append((0.15, {h: {"ade": r["reference_dual"][h]} for h in HORIZONS}, "ade",
                           MUTED, "ref: 24% pruning (dual) minADE@6"))
        for off, src, kind, col, lab in series:
            m = [src[h][kind]["delta_mean"] for h in HORIZONS]
            lo = [m[i] - src[h][kind]["ci"][0] for i, h in enumerate(HORIZONS)]
            hi = [src[h][kind]["ci"][1] - m[i] for i, h in enumerate(HORIZONS)]
            ax.errorbar(x + off, m, yerr=[lo, hi], fmt="o", color=col, capsize=4, label=lab)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x, list(HORIZONS))
        ax.set_xlabel("horizon")
        ax.set_ylabel("delta vs baseline (m), mean, 95% CI")
        ax.set_title(f"{s} (n={r['n_paired']})")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "delta_by_horizon.png", dpi=150)
    plt.close(fig)


def plot_relative(res, out):
    """Same deltas as a percentage of the baseline mean, so horizons share one scale."""
    fig, axes = plt.subplots(1, len(res), figsize=(6 * len(res), 4.2), squeeze=False)
    for ax, (s, r) in zip(axes[0], res.items()):
        x = np.arange(len(HORIZONS))
        series = [(-0.1, {h: r["horizons"][h]["ade"] for h in HORIZONS}, C1, "blackout (1 of 16 frames)")]
        if "reference_dual" in r:
            series.append((0.1, {h: r["reference_dual"][h] for h in HORIZONS}, MUTED,
                           "ref: 24% pruning (dual)"))
        for off, src, col, lab in series:
            base = [src[h]["base_mean"] for h in HORIZONS]
            m = [100 * src[h]["delta_mean"] / b for h, b in zip(HORIZONS, base)]
            lo = [m[i] - 100 * src[h]["ci"][0] / base[i] for i, h in enumerate(HORIZONS)]
            hi = [100 * src[h]["ci"][1] / base[i] - m[i] for i, h in enumerate(HORIZONS)]
            ax.errorbar(x + off, m, yerr=[lo, hi], fmt="o", color=col, capsize=4, label=lab)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x, list(HORIZONS))
        ax.set_xlabel("horizon")
        ax.set_ylabel("minADE@6 change vs baseline (%), 95% CI")
        ax.set_title(f"{s} (n={r['n_paired']})")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "relative_by_horizon.png", dpi=150)
    plt.close(fig)


def plot_cameras(res, out):
    fig, axes = plt.subplots(1, len(res), figsize=(6 * len(res), 4.2), squeeze=False)
    for ax, (s, r) in zip(axes[0], res.items()):
        names = list(r["by_camera"])
        x = np.arange(len(names))
        for off, h, col in ((-0.12, "1.6s", C2), (0.12, "6.4s", C1)):
            m = [r["by_camera"][n][h]["delta_mean"] for n in names]
            lo = [m[i] - r["by_camera"][n][h]["ci"][0] for i, n in enumerate(names)]
            hi = [r["by_camera"][n][h]["ci"][1] - m[i] for i, n in enumerate(names)]
            ax.errorbar(x + off, m, yerr=[lo, hi], fmt="o", color=col, capsize=4,
                        label=f"minADE@6 {h}")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x, [f"{n}\nn={r['by_camera'][n]['6.4s']['n']}" for n in names])
        ax.set_ylabel("blackout - baseline (m), mean, 95% CI")
        ax.set_title(f"{s}: by blacked-out camera (exploratory)")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "delta_by_camera.png", dpi=150)
    plt.close(fig)


def plot_scatter(pairs, out):
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.6), squeeze=False)
    for ax, (s, (B, X)) in zip(axes[0], pairs.items()):
        b = np.array([metric(r, "ade", "6.4s") for r in B])
        x = np.array([metric(r, "ade", "6.4s") for r in X])
        lim = [max(min(b.min(), x.min()), 1e-2), max(b.max(), x.max()) * 1.2]
        ax.loglog(b, x, "o", ms=3, alpha=0.5, color=C1)
        ax.plot(lim, lim, color=MUTED, lw=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("baseline minADE@6 (m)")
        ax.set_ylabel("blackout minADE@6 (m)")
        ax.set_title(s)
    fig.tight_layout()
    fig.savefig(out / "scatter_minade.png", dpi=150)
    plt.close(fig)


def fmt_p(d):
    return (f"{d['delta_mean']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}] "
            f"(base {d['base_mean']:.4f} -> {d['blackout_mean']:.4f}; median {d['delta_median']:+.4f}, "
            f"Wilcoxon p={d['wilcoxon_p']:.3g}, worse/better {d['frac_worse']:.2f}/{d['frac_better']:.2f})")


def main():
    out = OUT / "blackout_analysis"
    (out / "plots").mkdir(parents=True, exist_ok=True)
    g0 = check_g0()
    res, pairs = {}, {}
    for s, (base_id, bk_id) in SETS.items():
        base = load_rows(OUT / base_id, "baseline")
        bk = load_rows(OUT / bk_id, "baseline_bk1")
        if not bk:
            print(f"{s}: no blackout rows yet")
            continue
        res[s], B, X = analyse_set(base, bk, load_rows(OUT / REF[s], REF_TAG))
        pairs[s] = (B, X)
    if not res:
        sys.exit("nothing to analyse")

    plot_horizons(res, out / "plots")
    plot_relative(res, out / "plots")
    plot_cameras(res, out / "plots")
    plot_scatter(pairs, out / "plots")

    (out / "config.json").write_text(json.dumps({
        "sets": {s: {"baseline": SETS[s][0], "blackout": SETS[s][1]} for s in SETS},
        "blackout_tag": "baseline_bk1", "reference": {s: [REF[s], REF_TAG] for s in REF},
        "k": K, "n_boot": N_BOOT, "boot_seed": 0, "trim_n_boot": 20_000,
        "camera_names": CAMS, "plan": "plans/2026-10-18_camera-blackout-openloop.md",
    }, indent=2))
    (out / "metrics.json").write_text(json.dumps({"G0": g0, "sets": res}, indent=2))

    lines = ["t0 camera blackout (1 random camera per clip) vs baseline, open loop",
             f"G0 (stored baseline reproduces): {g0}", ""]
    for s, r in res.items():
        lines.append(f"== {s}: {r['n_paired']} paired clips "
                     f"({r['n_blackout_rows']} blackout rows) | G1: {r['G1']}")
        for h in HORIZONS:
            for kind in ("ade", "fde"):
                lines.append(f"  min{kind.upper()}@6 {h}: {fmt_p(r['horizons'][h][kind])}")
        for h, t in r["tail"].items():
            lines.append(f"  tail {h}: top-5 clips carry {t['top5_share_of_sum']:.2f} of the sum "
                         f"({', '.join(f'{x:+.3f}' for x in t['top5_deltas'])}); 10% trimmed mean "
                         f"{t['trim10_mean']:+.4f} [{t['trim10_ci'][0]:+.4f}, {t['trim10_ci'][1]:+.4f}]")
        for label, v in r["by_coc_change"].items():
            for h in ("1.6s", "6.4s"):
                lines.append(f"  {label:13s} n={v[h]['n']:3d} minADE@6 {h}: {fmt_p(v[h])}")
        if "reference_dual" in r:
            for h in HORIZONS:
                lines.append(f"  ref dual_u40_v2 minADE@6 {h}: {fmt_p(r['reference_dual'][h])}")
        lines.append("  by blacked-out camera, minADE@6 6.4s (exploratory):")
        for n, v in r["by_camera"].items():
            lines.append(f"    {n:16s} n={v['6.4s']['n']:3d}  {fmt_p(v['6.4s'])}")
        lines.append("  by bucket, minADE@6 6.4s:")
        for bu, v in r["by_bucket"].items():
            lines.append(f"    {bu:10s} n={v['n']:3d}  {fmt_p(v)}")
        c = r["coc"]
        lines.append(f"  CoC degenerate {c['degenerate_base']:.3f} -> {c['degenerate_blackout']:.3f} "
                     f"(delta {c['degenerate_delta']:+.3f} [{c['degenerate_delta_ci'][0]:+.3f}, "
                     f"{c['degenerate_delta_ci'][1]:+.3f}]); identical CoC text "
                     f"{c['identical_text']:.3f} | by camera "
                     + ", ".join(f"{k} {v:.2f}" for k, v in c["identical_text_by_camera"].items()))
        lines.append("")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
