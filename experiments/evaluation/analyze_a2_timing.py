"""Alpamayo 2 Super under per-frame capture-time jitter vs its synchronised input, test500.

Plan: plans/2026-10-26_camera-timing-openloop.md ("Alpamayo 2"). Arms from run_a2_timing.py:
sync, jit50_black (future t0 -> black), jit50_drop (future t0 left out of the prompt), all on
the same clips and sampling seeds, so every metric is a paired per-clip delta against A2 sync.
Alpamayo 1.5's arms (test2000 runs) are restricted to the same 500 clips for the cross-model
view; the models differ, so that comparison is in relative % and in how per-clip deltas line up.

Row-level G0: the shared cameras (0, 1, 2, 6) carry Alpamayo 1.5's dt and frames exactly, and
every drop row holds 24 - (#future t0) images.

Usage:
  python experiments/evaluation/analyze_a2_timing.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, trim_mean

sys.path.insert(0, str(Path(__file__).parent))
import analyze_blackout as ab  # plot style, loaders, metric
import analyze_camera_timing as act  # paired, vec

OUT = ab.OUT / "camtiming_analysis" / "a2"
N = 500
CAMS = [0, 1, 2, 3, 5, 6]
CAM_NAMES = {0: "cross_left", 1: "front_wide", 2: "cross_right", 3: "rear_left",
             5: "rear_right", 6: "front_tele"}
A2 = {"sync": ("a2timing_sync_test500", "alpamayo2_sync"),
      "jit50_black": ("a2timing_jit50_black_test500", "alpamayo2_jit50_black"),
      "jit50_drop": ("a2timing_jit50_drop_test500", "alpamayo2_jit50_drop")}
A15 = {"sync": ("camtiming_sync_test2000", "baseline"),
       "jit50_black": ("camtiming_jit50_black_test2000", "baseline_jit50_black"),
       "jit50_drop": ("camtiming_jit50_drop_test2000", "baseline_jit50_drop")}
HS = ("1.6s", "3.2s", "6.4s")
NB2 = act.N_BOOT_SECONDARY


def load(spec, clips=None):
    rows = {n: ab.load_rows(ab.OUT / d, t) for n, (d, t) in spec.items()}
    if clips is not None:
        rows = {n: {c: r[c] for c in clips if c in r} for n, r in rows.items()}
    return rows


def absolute(rows, clips):
    return {n: {f"{k}_{h}": {"mean": float(act.vec(r, clips, k, h).mean()),
                             "median": float(np.median(act.vec(r, clips, k, h)))}
                for k in ("ade", "fde") for h in HS} for n, r in rows.items()}


def vs_sync(rows, clips):
    out = {}
    for n in ("jit50_black", "jit50_drop"):
        out[n] = {f"{k}_{h}": act.paired(act.vec(rows["sync"], clips, k, h),
                                         act.vec(rows[n], clips, k, h),
                                         ab.N_BOOT if (k, h) == ("ade", "6.4s") else NB2)
                  for k in ("ade", "fde") for h in HS}
        d = act.vec(rows[n], clips) - act.vec(rows["sync"], clips)
        lo, hi = np.percentile([trim_mean(d[np.random.RandomState(s).randint(0, len(d), len(d))], 0.1)
                                for s in range(5000)], [2.5, 97.5])
        out[n]["trim10_6.4s"] = {"mean": float(trim_mean(d, 0.1)), "ci": [float(lo), float(hi)]}
        out[n]["coc"] = {
            "identical_text": float(np.mean([rows[n][c]["gen_coc"] == rows["sync"][c]["gen_coc"]
                                             for c in clips])),
            "degenerate_sync": float(np.mean([rows["sync"][c]["coc_degenerate"] for c in clips])),
            "degenerate_arm": float(np.mean([rows[n][c]["coc_degenerate"] for c in clips]))}
    return out


def g0_rows(a2, a15):
    """Shared cameras carry Alpamayo 1.5's draws and frames; drop rows hold the right count."""
    shared = [CAMS.index(ci) for ci in (0, 1, 2, 6)]
    bad_dt = bad_ts = bad_n = 0
    for n in ("jit50_black", "jit50_drop"):
        for c, r in a2[n].items():
            o = a15[n][c]
            bad_dt += [r["jitter_dt_ms"][i] for i in shared] != o["jitter_dt_ms"]
            bad_ts += [r["frame_ts_ms"][i] for i in shared] != o["frame_ts_ms"]
            if n == "jit50_drop":
                bad_n += r["n_images"] != 24 - len(r["future_t0_cams"])
    zero = [c for c, r in a2["jit50_drop"].items() if not r["future_t0_cams"]]
    same = sum(a2["jit50_drop"][c]["ade_rollout_k"] == a2["jit50_black"][c]["ade_rollout_k"]
               and a2["jit50_drop"][c]["gen_coc"] == a2["jit50_black"][c]["gen_coc"] for c in zero)
    return {"shared_cam_dt_mismatch": int(bad_dt), "shared_cam_frame_mismatch": int(bad_ts),
            "drop_image_count_mismatch": int(bad_n),
            "no_future_clips": len(zero), "no_future_clips_black_eq_drop": int(same)}


def breakdowns(rows, clips):
    ref = rows["jit50_black"]
    n_fut = np.array([len(ref[c]["future_t0_cams"]) for c in clips])          # (N,)
    out = {"n_future_hist": {int(k): int((n_fut == k).sum()) for k in range(7)},
           "by_n_future": {}, "by_bucket": {}, "drop_vs_black": {}, "ols_6.4s": {}}
    for n in ("jit50_black", "jit50_drop"):
        out["by_n_future"][n] = {}
        for k in range(7):
            idx = [c for c, b in zip(clips, n_fut) if b == k]
            if len(idx) >= 10:
                out["by_n_future"][n][k] = act.paired(act.vec(rows["sync"], idx),
                                                      act.vec(rows[n], idx), NB2)
        out["by_bucket"][n] = {}
        for b in ab.BUCKETS:
            idx = [c for c in clips if rows["sync"][c]["bucket"] == b]
            out["by_bucket"][n][b] = act.paired(act.vec(rows["sync"], idx), act.vec(rows[n], idx), NB2)
    B, D = rows["jit50_black"], rows["jit50_drop"]
    out["drop_vs_black"]["all"] = {h: act.paired(act.vec(B, clips, "ade", h),
                                                 act.vec(D, clips, "ade", h), NB2) for h in HS}
    d = act.vec(D, clips) - act.vec(B, clips)
    top2 = set(np.argsort(-np.abs(d))[:2])
    rest = [c for j, c in enumerate(clips) if j not in top2]
    out["drop_vs_black"]["without_top2_6.4s"] = act.paired(act.vec(B, rest), act.vec(D, rest), NB2)
    for label, test in (("tele_t0_handled", lambda r: 6 in r["future_t0_cams"]),
                        ("rear_t0_handled", lambda r: bool({3, 5} & set(r["future_t0_cams"])))):
        out["drop_vs_black"][label] = {}
        for flag in (True, False):
            idx = [c for c in clips if test(ref[c]) == flag]
            out["drop_vs_black"][label][str(flag)] = {
                "black_vs_sync": act.paired(act.vec(rows["sync"], idx), act.vec(B, idx), NB2),
                "drop_vs_sync": act.paired(act.vec(rows["sync"], idx), act.vec(D, idx), NB2),
                "drop_vs_black": act.paired(act.vec(B, idx), act.vec(D, idx), NB2)}
    X = np.array([[1.0] + [float(ci in ref[c]["future_t0_cams"]) for ci in CAMS] for c in clips])
    rng = np.random.RandomState(0)
    for n in ("jit50_black", "jit50_drop"):
        y = act.vec(rows[n], clips) - act.vec(rows["sync"], clips)
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        bs = []
        for _ in range(5000):
            i = rng.randint(0, len(y), len(y))
            bs.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0])
        lo, hi = np.percentile(np.array(bs), [2.5, 97.5], axis=0)
        out["ols_6.4s"][n] = {name: {"coef": float(coef[j]), "ci": [float(lo[j]), float(hi[j])]}
                              for j, name in enumerate(["intercept"] + [CAM_NAMES[ci] for ci in CAMS])}
    return out


def cross_model(a2, a15, clips):
    out = {}
    for n in ("jit50_black", "jit50_drop"):
        out[n] = {}
        for h in ("1.6s", "6.4s"):
            d2 = act.vec(a2[n], clips, "ade", h) - act.vec(a2["sync"], clips, "ade", h)
            d1 = act.vec(a15[n], clips, "ade", h) - act.vec(a15["sync"], clips, "ade", h)
            rho, p = spearmanr(d1, d2)
            out[n][h] = {"a15_rel_pct": float(100 * d1.mean() / act.vec(a15["sync"], clips, "ade", h).mean()),
                         "a2_rel_pct": float(100 * d2.mean() / act.vec(a2["sync"], clips, "ade", h).mean()),
                         "a15_delta": float(d1.mean()), "a2_delta": float(d2.mean()),
                         "per_clip_delta_spearman": float(rho), "p": float(p)}
    s1, s2 = act.vec(a15["sync"], clips), act.vec(a2["sync"], clips)
    out["sync_a2_minus_a15_6.4s"] = act.paired(s1, s2, NB2)
    return out


def plot(res):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, h in zip(axes, ("1.6s", "6.4s")):
        names = ["jit50_black", "jit50_drop"]
        x = np.arange(len(names))
        for off, model, col in ((-0.15, "a15", ab.C1), (0.15, "a2", ab.C3)):
            if model == "a2":
                m = [res["vs_sync"][n][f"ade_{h}"]["rel_pct"] for n in names]
                lo = [m[i] - res["vs_sync"][n][f"ade_{h}"]["rel_ci_pct"][0] for i, n in enumerate(names)]
                hi = [res["vs_sync"][n][f"ade_{h}"]["rel_ci_pct"][1] - m[i] for i, n in enumerate(names)]
                ax.errorbar(x + off, m, yerr=[lo, hi], fmt="o", color=col, capsize=4,
                            label="Alpamayo 2 Super (95% CI)")
            else:
                m = [res["cross_model"][n][h]["a15_rel_pct"] for n in names]
                ax.plot(x + off, m, "s", color=col, label="Alpamayo 1.5, same 500 clips")
        ax.axhline(0, color=ab.MUTED, lw=0.8)
        ax.set_xticks(x, names)
        ax.set_ylabel("minADE@6 change vs own sync (%)")
        ax.set_title(f"test500, {h}")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "a2_vs_a15_relative.png", dpi=150)
    plt.close(fig)


def main():
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    a2 = load(A2)
    counts = {n: len(r) for n, r in a2.items()}
    print("A2 rows:", counts)
    if any(v < N for v in counts.values()):
        sys.exit(f"incomplete: {counts}")
    clips = sorted(a2["sync"])
    a15 = load(A15, clips)
    assert all(len(r) == N for r in a15.values()), {n: len(r) for n, r in a15.items()}

    res = {"n_clips": N, "G0_rows": g0_rows(a2, a15),
           "absolute": {"a2": absolute(a2, clips), "a15": absolute(a15, clips)},
           "vs_sync": vs_sync(a2, clips), "breakdowns": breakdowns(a2, clips),
           "cross_model": cross_model(a2, a15, clips)}
    plot(res)
    (OUT / "config.json").write_text(json.dumps({
        "plan": "plans/2026-10-26_camera-timing-openloop.md (Alpamayo 2)",
        "a2_arms": A2, "a15_arms_same_clips": A15, "manifest": "test_500",
        "metric": "minADE@6 / minFDE@6 (first 6 of 8 samples), horizons 1.6/3.2/6.4 s",
        "bootstrap": {"primary_ade_6.4s": ab.N_BOOT, "secondary": NB2}}, indent=2))
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2))

    ab_ = res["absolute"]
    lines = [f"Alpamayo 2 Super camera timing, test500 (n={N})", f"G0 rows: {res['G0_rows']}", "",
             f"{'metric':10s} " + " ".join(f"{m}:{n:>12s}" for m in ("a2", "a15") for n in A2)]
    for k in ("ade", "fde"):
        for h in HS:
            lines.append(f"{k}_{h:6s} " + " ".join(f"{ab_[m][n][f'{k}_{h}']['mean']:>15.4f}"
                                                  for m in ("a2", "a15") for n in A2))
    lines.append("")
    for n, r in res["vs_sync"].items():
        for h in HS:
            v = r[f"ade_{h}"]
            lines.append(f"A2 {n:12s} ade {h}: d {v['delta_mean']:+.4f} [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] "
                         f"{v['rel_pct']:+.1f}% med {v['delta_median']:+.4f} p {v['wilcoxon_p']:.1e}")
        lines.append(f"   trim10 6.4s {r['trim10_6.4s']}, coc {r['coc']}")
    for n, r in res["cross_model"].items():
        lines.append(f"cross-model {n}: {r}")
    dvb = res["breakdowns"]["drop_vs_black"]
    lines.append(f"drop-black all 6.4s: {dvb['all']['6.4s']['delta_mean']:+.4f} {dvb['all']['6.4s']['ci']}")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
