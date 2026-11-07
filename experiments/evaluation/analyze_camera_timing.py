"""Camera timing mismatch vs the synchronised input, open loop on test2000.

Plan and gates: plans/2026-10-26_camera-timing-openloop.md.

Every arm feeds some cameras' frames from shifted capture times (`run_baseline.py
--lag-ms/--stride-ms/--timing-cams`) with the sampling seeds unchanged, so each metric is
a paired per-clip delta against the sync arm run on the same Blackwell cards:

  G0a  the frame-window cache reproduces the 4-frame cache byte for byte (builder summary).
  G0b  the sync arm is bitwise identical to the earlier Blackwell baseline
       (`baseline_bw_ps_test`) on the clips they share.
  G1   per arm, the bootstrap CI of the mean delta of minADE@6 (6.4 s) excludes 0.
  H1   single-camera 100 ms lag, camera vs camera (Holm-corrected Wilcoxon).
  H2   front-tele lag 67 < 100 < 200 < 300 ms: monotone means, adjacent-pair CIs.
  H3   all-camera lag vs single-camera lag at 100 ms, and by bucket / speed.

Arms still running are reported with their row count and left out of the statistics.

Usage:
  python experiments/evaluation/analyze_camera_timing.py
"""

import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
import analyze_blackout as ab  # plot style, loaders, paired statistics
import sample_cache as sc

OUT = ab.OUT
N_CLIPS = 2000
ARMS = {  # name -> (run_baseline tag, cameras, lag ms, stride ms)
    "sync": ("baseline", [], 0, 100),
    "tm6_lag100": ("baseline_tm6_lag100_st100", [6], 100, 100),
    "tm0_lag100": ("baseline_tm0_lag100_st100", [0], 100, 100),
    "tm1_lag100": ("baseline_tm1_lag100_st100", [1], 100, 100),
    "tm2_lag100": ("baseline_tm2_lag100_st100", [2], 100, 100),
    "tmall_lag100": ("baseline_tmall_lag100_st100", [0, 1, 2, 6], 100, 100),
    "tmall_lag300": ("baseline_tmall_lag300_st100", [0, 1, 2, 6], 300, 100),
    "tmall_st200": ("baseline_tmall_lag0_st200", [0, 1, 2, 6], 0, 200),
    "tm6_lag300": ("baseline_tm6_lag300_st100", [6], 300, 100),
    "tm6_lag67": ("baseline_tm6_lag66.667_st100", [6], 66.667, 100),
    "tm6_lag200": ("baseline_tm6_lag200_st100", [6], 200, 100),
    "tmall_st33": ("baseline_tmall_lag0_st33.333", [0, 1, 2, 6], 0, 33.333),
}
# per-frame jitter dt ~ U(-50, 50) ms on all 16 slots, same draws in all three arms
JITTER = {"jit50_black": "baseline_jit50_black",                   # design: future t0 -> black
          "jit50_drop": "baseline_jit50_drop",                     # future t0 left out of prompt
          "jit50_keep": "baseline_jit50_keep",                     # jitter part alone
          "jit50_black_noshift": "baseline_jit50_black_noshift"}   # blackout part alone
# every camera's t0 frame excluded, no jitter (shards 3, Blackwell 1-3)
T0X = {"t0x_black": "baseline_t0x_black", "t0x_drop": "baseline_t0x_drop"}
SINGLE = {6: "tm6_lag100", 0: "tm0_lag100", 1: "tm1_lag100", 2: "tm2_lag100"}
TELE_DOSE = [("sync", 0), ("tm6_lag67", 66.667), ("tm6_lag100", 100), ("tm6_lag200", 200),
             ("tm6_lag300", 300)]
N_BOOT_SECONDARY = 20_000   # primary (minADE@6, 6.4 s) keeps ab.N_BOOT = 200k


def paired(b, x, n_boot):
    b, x = np.asarray(b, float), np.asarray(x, float)
    d = x - b
    lo, hi = ab.boot_ci(d, n_boot=n_boot)
    return {"n": len(d), "base_mean": float(b.mean()), "arm_mean": float(x.mean()),
            "delta_mean": float(d.mean()), "ci": [lo, hi],
            "rel_pct": float(100 * d.mean() / b.mean()),
            "rel_ci_pct": [float(100 * lo / b.mean()), float(100 * hi / b.mean())],
            "delta_median": float(np.median(d)),
            "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
            "frac_worse": float(np.mean(d > 0)), "frac_better": float(np.mean(d < 0))}


def vec(rows, clips, kind="ade", h="6.4s"):
    return np.array([ab.metric(rows[c], kind, h) for c in clips])


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(run, 1.0)
    return adj.tolist()


def t0_speed(clips):
    """Ego speed at t0 (m/s) from the last history step, read off the 4-frame cache."""
    v = {}
    for c in clips:
        z = np.load(sc.path_for("test", c, sc.CALIB_T0))
        xyz = z["ego_history_xyz"][0, 0]                          # (16, 3)
        v[c] = float(np.linalg.norm(xyz[-1, :2] - xyz[-2, :2]) / 0.1)
    return v


def check_g0b(sync):
    ref = ab.load_rows(OUT / "baseline_bw_ps_test", "baseline")
    ov = [c for c in sync if c in ref]
    if not ov:
        return {"status": "MISSING"}
    same = [sync[c]["ade_rollout_k"] == ref[c]["ade_rollout_k"]
            and sync[c]["gen_coc"] == ref[c]["gen_coc"] for c in ov]
    return {"status": "PASS" if all(same) else "FAIL", "n_overlap": len(ov),
            "identical": int(sum(same))}


def realised_offsets(rows, cams, lag, stride):
    """Mean picked t0-slot time and slot spacing of the shifted cameras (ms), as a check."""
    order = [0, 1, 2, 6]
    t0s, gaps = [], []
    for r in rows.values():
        ts = np.asarray(r["frame_ts_ms"])                          # (4 cam, 4 slot) ms
        for ci in cams:
            t0s.append(ts[order.index(ci), -1])
            gaps.extend(np.diff(ts[order.index(ci)]))
    return {"t0_slot_ms_mean": float(np.mean(t0s)), "t0_slot_ms_range": [float(np.min(t0s)),
            float(np.max(t0s))], "slot_gap_ms_range": [float(np.min(gaps)), float(np.max(gaps))],
            "requested_lag_ms": lag, "requested_stride_ms": stride}


def analyse_arm(sync, arm, clips, speed):
    S, X = sync, arm
    r = {"horizons": {}}
    for h in ab.HORIZONS:
        r["horizons"][h] = {kind: paired(vec(S, clips, kind, h), vec(X, clips, kind, h),
                                         ab.N_BOOT if (kind, h) == ("ade", "6.4s")
                                         else N_BOOT_SECONDARY)
                            for kind in ("ade", "fde")}
    d = vec(X, clips) - vec(S, clips)
    r["tail_6.4s"] = ab.tail(d)
    r["coc"] = {"degenerate_sync": float(np.mean([S[c]["coc_degenerate"] for c in clips])),
                "degenerate_arm": float(np.mean([X[c]["coc_degenerate"] for c in clips])),
                "identical_text": float(np.mean([S[c]["gen_coc"] == X[c]["gen_coc"]
                                                 for c in clips]))}
    r["by_bucket"] = {}
    for b in ab.BUCKETS:
        idx = [c for c in clips if S[c]["bucket"] == b]
        if len(idx) >= 10:
            r["by_bucket"][b] = {h: paired(vec(S, idx, "ade", h), vec(X, idx, "ade", h),
                                           N_BOOT_SECONDARY) for h in ("1.6s", "6.4s")}
    cuts = np.percentile(list(speed.values()), [100 / 3, 200 / 3])
    r["speed_tertile_cuts_mps"] = [float(x) for x in cuts]
    r["by_speed"] = {}
    for k, (lo, hi) in enumerate(((-np.inf, cuts[0]), (cuts[0], cuts[1]), (cuts[1], np.inf))):
        idx = [c for c in clips if lo <= speed[c] < hi]
        r["by_speed"][f"T{k + 1}"] = {h: paired(vec(S, idx, "ade", h), vec(X, idx, "ade", h),
                                                N_BOOT_SECONDARY) for h in ("1.6s", "6.4s")}
    return r


def pairwise(rows_by_name, names, clips, label=lambda n: n):
    """delta = a - b on the same clips, minADE@6 at 1.6 s and 6.4 s, Holm over the family."""
    res = {}
    for h in ("1.6s", "6.4s"):
        fam = {}
        for a, b in itertools.combinations(names, 2):
            d = vec(rows_by_name[a], clips, "ade", h) - vec(rows_by_name[b], clips, "ade", h)
            lo, hi = ab.boot_ci(d, n_boot=N_BOOT_SECONDARY)
            fam[f"{label(a)}-{label(b)}"] = {
                "delta_mean": float(d.mean()), "ci": [lo, hi], "delta_median": float(np.median(d)),
                "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}
        for k, p in zip(fam, holm([v["wilcoxon_p"] for v in fam.values()])):
            fam[k]["holm_p"] = p
        res[h] = fam
    return res


def jitter_section(sync, rows, clips):
    """Jitter arms: cost by number of blacked-out t0 frames, additivity, per-camera blackout."""
    have = [n for n in JITTER if len(rows[n]) == N_CLIPS]
    if not have:
        return None
    ref = rows[have[0]]
    n_black = np.array([len(ref[c]["future_t0_cams"]) for c in clips])          # (N,)
    out = {"n_future_t0_hist": {int(k): int((n_black == k).sum()) for k in range(5)},
           "by_n_future_t0": {}}
    if "jit50_black" in have:
        dups = np.array([rows["jit50_black"][c]["dup_frames"] for c in clips])  # (N,)
        out["clips_with_dup_frames"] = int((dups > 0).sum())
    for n in have:
        out["by_n_future_t0"][n] = {}
        for k in range(5):
            idx = [c for c, b in zip(clips, n_black) if b == k]
            out["by_n_future_t0"][n][k] = {
                h: paired(vec(sync, idx, "ade", h), vec(rows[n], idx, "ade", h), N_BOOT_SECONDARY)
                for h in ("1.6s", "6.4s")}
    if {"jit50_black", "jit50_drop"} <= set(have):
        # same draws, same frames; only how a future t0 frame is handled differs: drop - black
        B, D = rows["jit50_black"], rows["jit50_drop"]
        out["drop_vs_black"] = {
            "all": {h: paired(vec(B, clips, "ade", h), vec(D, clips, "ade", h), N_BOOT_SECONDARY)
                    for h in ("1.6s", "6.4s")},
            "by_n_future_t0": {}}
        for k in range(1, 5):
            idx = [c for c, b in zip(clips, n_black) if b == k]
            out["drop_vs_black"]["by_n_future_t0"][k] = {
                h: paired(vec(B, idx, "ade", h), vec(D, idx, "ade", h), N_BOOT_SECONDARY)
                for h in ("1.6s", "6.4s")}
        # the difference is heavy-tailed: report it with and without the two largest clips,
        # and split by whether the front-tele t0 frame was the one handled
        d = vec(D, clips) - vec(B, clips)                                        # (N,) 6.4 s
        top2 = set(np.argsort(-np.abs(d))[:2])
        rest = [c for j, c in enumerate(clips) if j not in top2]
        out["drop_vs_black"]["without_top2_6.4s"] = paired(vec(B, rest), vec(D, rest),
                                                           N_BOOT_SECONDARY)
        out["drop_vs_black"]["top2_clips"] = [clips[j] for j in top2]
        out["drop_vs_black"]["by_tele_t0_handled"] = {}
        for flag in (True, False):
            idx = [c for c in clips if (6 in ref[c]["future_t0_cams"]) == flag]
            dt = vec(D, idx) - vec(B, idx)
            keep = [c for j, c in enumerate(idx) if j not in set(np.argsort(-np.abs(dt))[:2])]
            out["drop_vs_black"]["by_tele_t0_handled"][str(flag)] = {
                "black_vs_sync": paired(vec(sync, idx), vec(B, idx), N_BOOT_SECONDARY),
                "drop_vs_sync": paired(vec(sync, idx), vec(D, idx), N_BOOT_SECONDARY),
                "drop_vs_black": paired(vec(B, idx), vec(D, idx), N_BOOT_SECONDARY),
                "drop_vs_black_without_top2": paired(vec(B, keep), vec(D, keep), N_BOOT_SECONDARY)}
    if {"jit50_black", "jit50_keep", "jit50_black_noshift"} <= set(have):
        # interaction: per-clip delta_design - (delta_jitter + delta_blackout)
        out["additivity"] = {}
        for h in ("1.6s", "6.4s"):
            s = vec(sync, clips, "ade", h)
            d = {n: vec(rows[n], clips, "ade", h) - s
                 for n in ("jit50_black", "jit50_keep", "jit50_black_noshift")}
            inter = d["jit50_black"] - d["jit50_keep"] - d["jit50_black_noshift"]  # (N,)
            lo, hi = ab.boot_ci(inter, n_boot=N_BOOT_SECONDARY)
            out["additivity"][h] = {
                "design": float(d["jit50_black"].mean()),
                "jitter_alone": float(d["jit50_keep"].mean()),
                "blackout_alone": float(d["jit50_black_noshift"].mean()),
                "interaction": float(inter.mean()), "interaction_ci": [lo, hi]}
    # which camera's missing t0 frame costs what: OLS of the delta on 0/1 blackout indicators
    cams = [0, 1, 2, 6]
    Xd = np.array([[1.0] + [float(ci in ref[c]["future_t0_cams"]) for ci in cams]
                   for c in clips])                                              # (N, 5)
    rng = np.random.RandomState(0)
    out["per_camera_blackout_ols"] = {}
    for n in have:
        y = vec(rows[n], clips) - vec(sync, clips)                               # (N,) 6.4 s
        coef = np.linalg.lstsq(Xd, y, rcond=None)[0]
        bs = []
        for _ in range(5000):
            i = rng.randint(0, len(y), len(y))
            bs.append(np.linalg.lstsq(Xd[i], y[i], rcond=None)[0])
        lo, hi = np.percentile(np.array(bs), [2.5, 97.5], axis=0)
        out["per_camera_blackout_ols"][n] = {
            name: {"coef": float(coef[j]), "ci": [float(lo[j]), float(hi[j])]}
            for j, name in enumerate(["intercept"] + [ab.CAMS[ci] for ci in cams])}
    return out


def plot_jitter(jit, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, h in zip(axes, ("1.6s", "6.4s")):
        for k, (n, col) in enumerate(zip(jit["by_n_future_t0"], (ab.C3, ab.C2, ab.C1, ab.C4))):
            r = jit["by_n_future_t0"][n]
            xs = np.arange(5) + (k - 1) * 0.08
            m = [r[q][h]["delta_mean"] for q in range(5)]
            lo = [m[q] - r[q][h]["ci"][0] for q in range(5)]
            hi = [r[q][h]["ci"][1] - m[q] for q in range(5)]
            ax.errorbar(xs, m, yerr=[lo, hi], fmt="o-", color=col, capsize=3, label=n)
        ax.axhline(0, color=ab.MUTED, lw=0.8)
        ax.set_xticks(range(5), [f"{q}\n(n={jit['n_future_t0_hist'][q]})" for q in range(5)])
        ax.set_xlabel("t0 frames whose dt > 0 (blacked out in *_black, removed in *_drop)")
        ax.set_ylabel("minADE@6 delta vs sync (m), 95% CI")
        ax.set_title(f"per-frame jitter +/-50 ms: {h}")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "jitter_by_n_black.png", dpi=150)
    plt.close(fig)


def blackout_context():
    """Per-camera t0 blackout on test500 (Ada) as relative %, for scale only."""
    base = ab.load_rows(OUT / "baseline_pred_test", "baseline")
    out = {}
    for cam in SINGLE:
        bk = ab.load_rows(OUT / f"blackout_cam{cam}_test", f"baseline_bkcam{cam}")
        clips = [c for c in bk if c in base]
        if clips:
            b, x = vec(base, clips), vec(bk, clips)
            out[ab.CAMS[cam]] = {"n": len(clips), "rel_pct": float(100 * (x - b).mean() / b.mean())}
    dual = ab.load_rows(OUT / ab.REF["test500"], ab.REF_TAG)
    clips = [c for c in dual if c in base]
    if clips:
        b, x = vec(base, clips), vec(dual, clips)
        out["pruning_dual_u40_v2"] = {"n": len(clips),
                                      "rel_pct": float(100 * (x - b).mean() / b.mean())}
    return out


def plot_arms(res, out):
    """Relative change of minADE@6 vs sync per arm, three horizons."""
    names = list(res["arms"])
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(names) + 2), 4.6))
    x = np.arange(len(names))
    for off, h, col in ((-0.22, "1.6s", ab.C1), (0.0, "3.2s", ab.C2), (0.22, "6.4s", ab.C3)):
        m = [res["arms"][n]["horizons"][h]["ade"]["rel_pct"] for n in names]
        lo = [m[i] - res["arms"][n]["horizons"][h]["ade"]["rel_ci_pct"][0] for i, n in enumerate(names)]
        hi = [res["arms"][n]["horizons"][h]["ade"]["rel_ci_pct"][1] - m[i] for i, n in enumerate(names)]
        ax.errorbar(x + off, m, yerr=[lo, hi], fmt="o", color=col, capsize=3, label=h)
    ax.axhline(0, color=ab.MUTED, lw=0.8)
    ax.set_xticks(x, names, rotation=30, ha="right")
    ax.set_ylabel("minADE@6 change vs sync (%), 95% CI")
    ax.set_title("test2000: camera timing arms vs synchronised input")
    ax.legend(frameon=False, title="horizon")
    fig.tight_layout()
    fig.savefig(out / "relative_by_arm.png", dpi=150)
    plt.close(fig)


def plot_tele_dose(res, out):
    pts = [(ms, res["arms"][n]) if n != "sync" else (0, None) for n, ms in TELE_DOSE
           if n == "sync" or n in res["arms"]]
    if len(pts) < 3:
        return
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for h, col in (("1.6s", ab.C1), ("6.4s", ab.C3)):
        xs, m, lo, hi = [], [], [], []
        for ms, r in pts:
            xs.append(ms)
            if r is None:
                m.append(0.0), lo.append(0.0), hi.append(0.0)
                continue
            v = r["horizons"][h]["ade"]
            m.append(v["delta_mean"]), lo.append(v["delta_mean"] - v["ci"][0])
            hi.append(v["ci"][1] - v["delta_mean"])
        ax.errorbar(xs, m, yerr=[lo, hi], fmt="o-", color=col, capsize=3, label=h)
    ax.axvspan(0, 33.3, color=ab.MUTED, alpha=0.12, label="natural skew (<= 1 frame)")
    ax.axhline(0, color=ab.MUTED, lw=0.8)
    ax.set_xlabel("front-tele lag (ms)")
    ax.set_ylabel("minADE@6 delta vs sync (m), 95% CI")
    ax.set_title("front telephoto camera lag: dose-response")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "tele_dose_response.png", dpi=150)
    plt.close(fig)


def plot_breakdown(res, out, key, groups, fname, title):
    names = list(res["arms"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, h in zip(axes, ("1.6s", "6.4s")):
        x = np.arange(len(names))
        offs = np.linspace(-0.3, 0.3, len(groups))
        for k, (g, col) in enumerate(zip(groups, (ab.C1, ab.C2, ab.C3, ab.C4))):
            m, lo, hi = [], [], []
            for n in names:
                v = res["arms"][n][key].get(g, {}).get(h)
                if v is None:
                    m.append(np.nan), lo.append(0), hi.append(0)
                    continue
                m.append(v["delta_mean"]), lo.append(v["delta_mean"] - v["ci"][0])
                hi.append(v["ci"][1] - v["delta_mean"])
            ax.errorbar(x + offs[k], m, yerr=[lo, hi], fmt="o", color=col, capsize=2, label=g)
        ax.axhline(0, color=ab.MUTED, lw=0.8)
        ax.set_xticks(x, names, rotation=30, ha="right")
        ax.set_ylabel("minADE@6 delta vs sync (m), 95% CI")
        ax.set_title(f"{title}: {h}")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / fname, dpi=150)
    plt.close(fig)


def main():
    out = OUT / "camtiming_analysis"
    (out / "plots").mkdir(parents=True, exist_ok=True)
    tags = {**{n: v[0] for n, v in ARMS.items()}, **JITTER, **T0X}
    rows = {n: ab.load_rows(OUT / f"camtiming_{n}_test2000", tag) for n, tag in tags.items()}
    counts = {n: len(r) for n, r in rows.items()}
    print("rows:", counts)
    sync = rows["sync"]
    if len(sync) < N_CLIPS:
        sys.exit(f"sync arm incomplete ({len(sync)}/{N_CLIPS})")
    done = [n for n in tags if n != "sync" and counts[n] == N_CLIPS]
    clips = sorted(sync)
    speed = t0_speed(clips)

    cache_summary = sc.PRE / "test_timing" / "summary.txt"
    res = {"row_counts": counts, "complete_arms": done,
           "G0a_cache": {"ready": (sc.PRE / "test_timing" / "READY").exists(),
                         "summary": cache_summary.read_text() if cache_summary.exists() else None},
           "G0b_sync_vs_bw_baseline": check_g0b(sync),
           "sync": {"minADE6_6.4s_mean": float(vec(sync, clips).mean()),
                    "minADE6_1.6s_mean": float(vec(sync, clips, "ade", "1.6s").mean()),
                    "coc_degenerate": float(np.mean([sync[c]["coc_degenerate"] for c in clips]))},
           "arms": {}, "realised": {}}
    for n in done:
        print(f"analysing {n}", flush=True)
        res["arms"][n] = analyse_arm(sync, rows[n], clips, speed)
        if n in ARMS:
            _, cams, lag, stride = ARMS[n]
            res["realised"][n] = realised_offsets(rows[n], cams, lag, stride)
        res["arms"][n]["G1_ci_excludes_0"] = bool(
            res["arms"][n]["horizons"]["6.4s"]["ade"]["ci"][0] > 0
            or res["arms"][n]["horizons"]["6.4s"]["ade"]["ci"][1] < 0)

    if all(SINGLE[c] in done for c in SINGLE):
        res["H1_single_camera_pairwise"] = pairwise(
            rows, [SINGLE[c] for c in SINGLE], clips,
            label=lambda n: ab.CAMS[next(c for c, v in SINGLE.items() if v == n)])
    dose = [(n, ms) for n, ms in TELE_DOSE if n == "sync" or n in done]
    if len(dose) == len(TELE_DOSE):
        means = [0.0] + [res["arms"][n]["horizons"]["6.4s"]["ade"]["delta_mean"]
                         for n, _ in dose[1:]]
        adj = {}
        for (a, _), (b, _) in itertools.pairwise(dose):
            d = vec(rows[b], clips) - vec(rows[a], clips)
            lo, hi = ab.boot_ci(d, n_boot=N_BOOT_SECONDARY)
            adj[f"{b}-{a}"] = {"delta_mean": float(d.mean()), "ci": [lo, hi]}
        res["H2_tele_dose"] = {"lags_ms": [ms for _, ms in dose], "delta_means_6.4s": means,
                               "monotone": bool(np.all(np.diff(means) > 0)),
                               "adjacent_pairs": adj}
    if "tmall_lag100" in done and all(SINGLE[c] in done for c in SINGLE):
        res["H3_all_vs_single_100ms"] = {
            ab.CAMS[c]: {h: paired(vec(rows[SINGLE[c]], clips, "ade", h),
                                   vec(rows["tmall_lag100"], clips, "ade", h), N_BOOT_SECONDARY)
                         for h in ("1.6s", "6.4s")} for c in SINGLE}
    res["jitter"] = jitter_section(sync, rows, clips)
    if all(n in done for n in T0X):
        B, D = rows["t0x_black"], rows["t0x_drop"]
        res["t0x_drop_vs_black"] = {h: paired(vec(B, clips, "ade", h), vec(D, clips, "ade", h),
                                              N_BOOT_SECONDARY) for h in ab.HORIZONS}
        for j in ("jit50_black", "jit50_drop"):
            if j in done:  # all four t0 frames gone vs the jitter arm's ~2 of 4
                m = j.split("_")[1]
                res[f"t0x_{m}_vs_{j}"] = {h: paired(vec(rows[j], clips, "ade", h),
                                                    vec(rows[f"t0x_{m}"], clips, "ade", h),
                                                    N_BOOT_SECONDARY) for h in ab.HORIZONS}
    res["blackout_context_test500"] = blackout_context()
    in500 = ab.load_rows(OUT / "baseline_pred_test", "baseline")
    test500 = [c for c in clips if c in in500]
    res["test500_subset_rel_pct_6.4s"] = {
        n: float(100 * (vec(rows[n], test500) - vec(sync, test500)).mean()
                 / vec(sync, test500).mean()) for n in done}

    if done:
        plot_arms(res, out / "plots")
        plot_tele_dose(res, out / "plots")
        plot_breakdown(res, out / "plots", "by_bucket", ab.BUCKETS, "delta_by_bucket.png",
                       "by GT-path bucket")
        plot_breakdown(res, out / "plots", "by_speed", ["T1", "T2", "T3"], "delta_by_speed.png",
                       "by t0 speed tertile")
    if res["jitter"]:
        plot_jitter(res["jitter"], out / "plots")

    (out / "config.json").write_text(json.dumps({
        "plan": "plans/2026-10-26_camera-timing-openloop.md",
        "arms": {n: {"exp_id": f"camtiming_{n}_test2000", "tag": v[0], "cams": v[1],
                     "lag_ms": v[2], "stride_ms": v[3]} for n, v in ARMS.items()},
        "jitter_arms": {n: {"exp_id": f"camtiming_{n}_test2000", "tag": t, "jitter_ms": 50}
                        for n, t in JITTER.items()},
        "t0x_arms": {n: {"exp_id": f"camtiming_{n}_test2000", "tag": t, "shards": 3,
                         "cards": "Blackwell 1-3"} for n, t in T0X.items()},
        "metric": "minADE@6 / minFDE@6 (first 6 of 8 samples), horizons 1.6/3.2/6.4 s",
        "bootstrap": {"primary_ade_6.4s": ab.N_BOOT, "secondary": N_BOOT_SECONDARY},
        "hardware": "RTX PRO 5000 Blackwell, shard i on card i for every arm",
    }, indent=2))
    (out / "metrics.json").write_text(json.dumps(res, indent=2))

    lines = [f"camera timing, test2000 open loop -- complete arms {len(done)}/{len(tags) - 1}",
             f"rows {counts}",
             f"G0a cache READY={res['G0a_cache']['ready']}  G0b {res['G0b_sync_vs_bw_baseline']}",
             (f"sync minADE@6 6.4 s {res['sync']['minADE6_6.4s_mean']:.4f}, "
              f"1.6 s {res['sync']['minADE6_1.6s_mean']:.4f}"), "",
             (f"{'arm':14s} {'d6.4s (m)':>10s} {'95% CI':>20s} {'rel%':>7s} {'d1.6s':>8s} "
              f"{'rel%1.6':>8s} {'med':>8s} {'p':>8s} {'CoC same':>8s} G1")]
    for n in done:
        a = res["arms"][n]
        p6, p1 = a["horizons"]["6.4s"]["ade"], a["horizons"]["1.6s"]["ade"]
        lines.append(f"{n:14s} {p6['delta_mean']:+10.4f} [{p6['ci'][0]:+.4f},{p6['ci'][1]:+.4f}] "
                     f"{p6['rel_pct']:+6.1f}% {p1['delta_mean']:+8.4f} {p1['rel_pct']:+7.1f}% "
                     f"{p6['delta_median']:+8.4f} {p6['wilcoxon_p']:8.1e} "
                     f"{a['coc']['identical_text']:8.1%} {'Y' if a['G1_ci_excludes_0'] else 'n'}")
    if "H2_tele_dose" in res:
        means = np.round(res["H2_tele_dose"]["delta_means_6.4s"], 4)
        lines += ["", f"H2 tele dose 6.4 s means {means} monotone={res['H2_tele_dose']['monotone']}"]
    jit = res["jitter"]
    if jit:
        lines += ["", f"jitter: t0 frames with dt>0 per clip {jit['n_future_t0_hist']}"]
        for n, r in jit["by_n_future_t0"].items():
            lines.append(f"  {n:20s} d6.4s by #future t0 0..4: "
                         + " ".join(f"{r[k]['6.4s']['delta_mean']:+.3f}" for k in range(5)))
        if "additivity" in jit:
            a = jit["additivity"]["6.4s"]
            lines.append(f"  additivity 6.4 s: design {a['design']:+.4f} = jitter {a['jitter_alone']:+.4f}"
                         f" + blackout {a['blackout_alone']:+.4f} + interaction {a['interaction']:+.4f}"
                         f" [{a['interaction_ci'][0]:+.4f},{a['interaction_ci'][1]:+.4f}]")
        if "drop_vs_black" in jit:
            for h in ("1.6s", "6.4s"):
                v = jit["drop_vs_black"]["all"][h]
                lines.append(f"  drop - black {h}: black {v['base_mean']:.4f} drop {v['arm_mean']:.4f} "
                             f"d {v['delta_mean']:+.4f} [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]")
    lines += ["", f"blackout context (test500, Ada, rel% 6.4 s): {res['blackout_context_test500']}"]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
