"""Does a clip's realised capture-time perturbation track how much its prediction changes?

Plan: plans/2026-10-26_camera-timing-openloop.md (per-frame jitter). For every test2000 clip of
the jitter arms (jit50_black, jit50_drop; same dt draws, so the same frame times):

  features   how far the frames actually moved in capture time vs the sync frames, over the
             frames still in the input (an excluded future t0 frame is not "moved")
  outcomes   dADE    minADE@6 arm - sync (signed: harm)
             dtraj   mean displacement between the arm's and sync's k-th sampled path, averaged
                     over k = 8 (unsigned: sensitivity; same noise seeds, so paired per sample)

Spearman rho per feature x outcome, raw and partial. Partial residualises both ranks on which
cameras' t0 was excluded, because excluded slots are never shifted and exclusion costs on its
own -- without it a timing feature would partly stand in for "fewer frames were excluded".
Bootstrap CIs over clips, Holm over the features of one arm x outcome.

Usage:
  python experiments/evaluation/analyze_jitter_corr.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from scipy.stats import t as t_dist

sys.path.insert(0, str(Path(__file__).parent))
import analyze_blackout as ab  # plot style, loaders
import sample_cache as sc

OUT = ab.OUT / "camtiming_analysis" / "jitter_corr"
CAMS = [0, 1, 2, 6]
CAM_NAMES = {0: "cross_left", 1: "front_wide", 2: "cross_right", 6: "front_tele"}
ARMS = {"jit50_black": "baseline_jit50_black", "jit50_drop": "baseline_jit50_drop"}
N_BOOT = 2000


def sync_frame_ts(clip_id, t0_us):
    """Capture times (ms rel. t0) of the frames the unjittered loader picks: (4 cam, 4 slot)."""
    z = np.load(sc.path_for("test_timing", clip_id, t0_us))
    req = t0_us - (3 - np.arange(4)) * 100_000                      # (4,) us
    return np.stack([z[f"ts_c{ci}"][np.searchsorted(z[f"ts_c{ci}"], req, side="right") - 1]
                     for ci in CAMS]).astype(float) / 1e3 - t0_us / 1e3


def features(row, sync_ts):
    ts = np.asarray(row["frame_ts_ms"], float)                      # (4 cam, 4 slot) ms
    shift = ts - sync_ts                                            # (4, 4) ms
    excl = np.zeros((4, 4), bool)
    for c, ci in enumerate(CAMS):
        excl[c, 3] = ci in row["future_t0_cams"]
    valid = ~excl
    a = np.abs(shift)
    dt = np.abs(np.asarray(row["jitter_dt_ms"], float))             # (4, 4) drawn |dt|
    f = {"mean_abs_shift": a[valid].mean(), "max_abs_shift": a[valid].max(),
         "mean_signed_shift": shift[valid].mean(), "n_frames_moved": float((a[valid] > 1).sum()),
         "mean_abs_dt_drawn": dt[valid].mean(),
         "t0_mean_abs_shift": a[:, 3][valid[:, 3]].mean() if valid[:, 3].any() else np.nan}
    for i in range(3):
        f[f"slot_t0-{0.1 * (3 - i):.1f}_mean_abs_shift"] = a[:, i].mean()
    for c, ci in enumerate(CAMS):
        f[f"{CAM_NAMES[ci]}_mean_abs_shift"] = a[c][valid[c]].mean()
    gaps = [np.abs(np.diff(ts[c][valid[c]]) - 100) for c in range(4)]
    f["interval_irregularity"] = float(np.concatenate(gaps).mean())  # |slot gap - 100 ms|
    t0v = ts[:, 3][valid[:, 3]]
    f["t0_cross_cam_spread"] = float(t0v.max() - t0v.min()) if len(t0v) >= 2 else np.nan
    f["n_dup_frames"] = float(row["dup_frames"])
    return f


def traj_change(row, srow, h):
    pa = np.asarray(row["pred_xy_k"], float)[:, :h]                 # (8, h, 2)
    ps = np.asarray(srow["pred_xy_k"], float)[:, :h]                # (8, h, 2)
    return float(np.linalg.norm(pa - ps, axis=-1).mean())


def resid(r, Z):
    return r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0]


def rho(x, y, Z=None):
    rx, ry = rankdata(x), rankdata(y)
    if Z is not None:
        rx, ry = resid(rx, Z), resid(ry, Z)
    return float(np.corrcoef(rx, ry)[0, 1])


def corr_with_ci(x, y, Z, rng_idx):
    ok = ~np.isnan(x)
    x, y, Z = x[ok], y[ok], Z[ok]
    n, k = len(x), Z.shape[1]
    out = {"n": int(n)}
    for name, zz, df in (("raw", None, n - 2), ("partial", Z, n - 2 - (k - 1))):
        r = rho(x, y, zz)
        tt = r * np.sqrt(df / max(1e-12, 1 - r ** 2))
        bs = [rho(x[i], y[i], None if zz is None else zz[i]) for i in rng_idx if len(i) == n]
        if not bs:  # NaN rows removed: redraw at this n
            rs = np.random.RandomState(1)
            bs = [rho(x[i], y[i], None if zz is None else zz[i])
                  for i in rs.randint(0, n, size=(N_BOOT, n))]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[name] = {"rho": r, "ci": [float(lo), float(hi)],
                     "p": float(2 * t_dist.sf(abs(tt), df))}
    return out


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(run, 1.0)
    return adj.tolist()


def binned(x, y, q=5):
    edges = np.percentile(x, np.linspace(0, 100, q + 1))
    res = []
    for j in range(q):
        m = (x >= edges[j]) & ((x < edges[j + 1]) if j < q - 1 else (x <= edges[j + 1]))
        lo, hi = ab.boot_ci(y[m], n_boot=5000)
        res.append({"x_range": [float(edges[j]), float(edges[j + 1])], "x_mean": float(x[m].mean()),
                    "n": int(m.sum()), "y_mean": float(y[m].mean()), "ci": [lo, hi]})
    return res


def main():
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    sync = ab.load_rows(ab.OUT / "camtiming_sync_test2000", "baseline")
    clips = sorted(sync)
    arms = {n: ab.load_rows(ab.OUT / f"camtiming_{n}_test2000", t) for n, t in ARMS.items()}
    for n, r in arms.items():
        assert len(r) == len(clips), f"{n}: {len(r)} rows"
    s_ts = {c: sync_frame_ts(c, sc.CALIB_T0) for c in clips}        # (4, 4) ms each
    ref = arms["jit50_black"]
    F = [features(ref[c], s_ts[c]) for c in clips]
    fnames = list(F[0])
    X = {f: np.array([d[f] for d in F], float) for f in fnames}     # (N,) each
    excl = np.array([[ci in ref[c]["future_t0_cams"] for ci in CAMS] for c in clips], float)
    Z = np.column_stack([np.ones(len(clips)), excl])                # (N, 5)
    rs = np.random.RandomState(0)
    boot_idx = list(rs.randint(0, len(clips), size=(N_BOOT, len(clips))))

    res = {"n_clips": len(clips), "feature_summary": {
        f: {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v)),
            "p5_p95": np.nanpercentile(v, [5, 95]).round(2).tolist()} for f, v in X.items()},
        "arms": {}}
    for n, rows in arms.items():
        Y = {"dADE_6.4s": np.array([ab.metric(rows[c], "ade", "6.4s")
                                    - ab.metric(sync[c], "ade", "6.4s") for c in clips]),
             "dADE_1.6s": np.array([ab.metric(rows[c], "ade", "1.6s")
                                    - ab.metric(sync[c], "ade", "1.6s") for c in clips]),
             "dtraj_6.4s": np.array([traj_change(rows[c], sync[c], 64) for c in clips]),
             "dtraj_1.6s": np.array([traj_change(rows[c], sync[c], 16) for c in clips])}
        coc_same = np.array([rows[c]["gen_coc"] == sync[c]["gen_coc"] for c in clips])
        res["arms"][n] = {"outcome_summary": {k: {"mean": float(v.mean()), "median": float(np.median(v))}
                                              for k, v in Y.items()}, "corr": {}, "binned": {}}
        for yk, y in Y.items():
            print(f"{n} {yk}", flush=True)
            cc = {f: corr_with_ci(X[f], y, Z, boot_idx) for f in fnames}
            for kind in ("raw", "partial"):
                for f, p in zip(fnames, holm([cc[f][kind]["p"] for f in fnames])):
                    cc[f][kind]["holm_p"] = p
            res["arms"][n]["corr"][yk] = cc
            res["arms"][n]["binned"][yk] = binned(X["mean_abs_shift"], y)
        # same, on clips whose reasoning text did not change: timing acting on the trajectory alone
        idx = np.flatnonzero(coc_same)
        res["arms"][n]["coc_unchanged"] = {"n": len(idx), "corr": {
            yk: {f: corr_with_ci(X[f][idx], Y[yk][idx], Z[idx], [])["partial"]
                 for f in ("mean_abs_shift", "t0_mean_abs_shift", "front_tele_mean_abs_shift")}
            for yk in ("dADE_6.4s", "dtraj_1.6s", "dtraj_6.4s")}}
        res["arms"][n]["coc_change_rate_by_shift_quintile"] = [
            {"x_mean": b["x_mean"], "coc_changed": float(1 - coc_same[
                (X["mean_abs_shift"] >= b["x_range"][0]) & (X["mean_abs_shift"] <= b["x_range"][1])].mean())}
            for b in binned(X["mean_abs_shift"], coc_same.astype(float))]
    plot_heatmap(res, fnames)
    plot_binned(res)

    (OUT / "config.json").write_text(json.dumps({
        "arms": ARMS, "baseline": "camtiming_sync_test2000",
        "features": "realised capture-time shift vs sync frames, excluded future t0 frames left out",
        "outcomes": {"dADE": "minADE@6 arm - sync", "dtraj": "mean |path_k(arm) - path_k(sync)| over "
                                                            "k=8 and steps (first 16 / all 64)"},
        "partial_on": "intercept + 4 indicators: camera c's t0 excluded",
        "bootstrap": N_BOOT, "multiple_testing": "Holm over features per arm x outcome x raw/partial",
    }, indent=2))
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2))
    write_summary(res, fnames)


def plot_heatmap(res, fnames):
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.42 * len(fnames) + 2.2))
    for ax, (n, r) in zip(axes, res["arms"].items()):
        ys = list(r["corr"])
        M = np.array([[r["corr"][y][f]["partial"]["rho"] for y in ys] for f in fnames])
        S = np.array([[r["corr"][y][f]["partial"]["holm_p"] < 0.05 for y in ys] for f in fnames])
        im = ax.imshow(M, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
        for i in range(len(fnames)):
            for j in range(len(ys)):
                ax.text(j, i, f"{M[i, j]:+.2f}{'*' if S[i, j] else ''}", ha="center", va="center",
                        fontsize=8, color=ab.INK)
        ax.set_xticks(range(len(ys)), ys, rotation=20)
        ax.set_yticks(range(len(fnames)), fnames if n == "jit50_black" else [""] * len(fnames))
        ax.set_title(f"{n}: partial Spearman rho (* Holm p<0.05)")
    fig.colorbar(im, ax=axes, shrink=0.6)
    fig.savefig(OUT / "plots" / "partial_rho_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_binned(res):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, yk in zip(axes, ("dADE_1.6s", "dADE_6.4s", "dtraj_1.6s", "dtraj_6.4s")):
        for n, col in zip(res["arms"], (ab.C3, ab.C2)):
            b = res["arms"][n]["binned"][yk]
            x = [v["x_mean"] for v in b]
            m = [v["y_mean"] for v in b]
            ax.errorbar(x, m, yerr=[[v["y_mean"] - v["ci"][0] for v in b],
                                    [v["ci"][1] - v["y_mean"] for v in b]],
                        fmt="o-", color=col, capsize=3, label=n)
        ax.axhline(0, color=ab.MUTED, lw=0.8)
        ax.set_xlabel("mean |capture-time shift| of the frames kept (ms), quintiles")
        ax.set_ylabel(f"{yk} (m), mean, 95% CI")
        ax.set_title(yk)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "binned_by_mean_shift.png", dpi=150)
    plt.close(fig)


def write_summary(res, fnames):
    lines = [f"jitter timing vs prediction change, test2000 (n={res['n_clips']})", "",
             "feature spread (mean, sd, p5-p95):"]
    for f, v in res["feature_summary"].items():
        lines.append(f"  {f:34s} {v['mean']:7.2f} {v['sd']:6.2f}  {v['p5_p95']}")
    for n, r in res["arms"].items():
        lines += ["", f"== {n}  partial Spearman rho [95% CI] (Holm p)"]
        ys = list(r["corr"])
        lines.append(f"  {'feature':34s} " + " ".join(f"{y:>26s}" for y in ys))
        for f in fnames:
            cells = []
            for y in ys:
                v = r["corr"][y][f]["partial"]
                cells.append(f"{v['rho']:+.3f}[{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}]({v['holm_p']:.0e})")
            lines.append(f"  {f:34s} " + " ".join(f"{c:>26s}" for c in cells))
        lines.append(f"  CoC unchanged subset n={r['coc_unchanged']['n']}: " + "; ".join(
            f"{y}/{f}: {v['rho']:+.3f}[{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}]"
            for y, d in r["coc_unchanged"]["corr"].items() for f, v in d.items()))
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
