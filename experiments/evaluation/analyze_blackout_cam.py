"""Per-camera t0 blackout vs baseline, open loop (test500 + OOD-val262).

Plan: plans/2026-10-18_camera-blackout-openloop.md ("추가: 카메라별 고정 블랙아웃").

Each condition blacks out the t0 frame of ONE fixed camera on every clip
(`run_baseline.py --blackout-cam {0,1,2,6}`). Every camera runs on the same clips with the
same sampling seeds, so cameras are compared to the baseline and to each other paired:

  G0'  clips where the random-camera run (`blackout_test` / `blackout_oodval`) happened to
       black out the same camera are the same computation and must be bitwise identical.
  H2   front-wide costs more than each other camera: paired camera-vs-camera delta of
       minADE@6 (6.4 s) has CI > 0 and Holm-corrected Wilcoxon p < 0.05, on both sets.

Usage:
  python experiments/evaluation/analyze_blackout_cam.py
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

OUT = ab.OUT
CAMS = {0: "cross_left_120", 1: "front_wide_120", 2: "cross_right_120", 6: "front_tele_30"}
SETS = {"test500": ("baseline_pred_test", "test", "blackout_test"),
        "oodval262": ("baseline_pred_oodval", "oodval", "blackout_oodval")}
COLORS = {0: ab.C1, 1: ab.C4, 2: ab.C2, 6: ab.C3}
FRONT = 1


def holm(ps):
    """Holm step-down adjusted p-values."""
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(run, 1.0)
    return adj.tolist()


def g0_overlap(cam_rows, rnd_rows, cam):
    ov = [c for c, r in rnd_rows.items() if r["blackout_cam"] == cam and c in cam_rows]
    same = [cam_rows[c]["ade_rollout_k"] == rnd_rows[c]["ade_rollout_k"]
            and cam_rows[c]["gen_coc"] == rnd_rows[c]["gen_coc"] for c in ov]
    return {"n_overlap": len(ov), "identical": int(sum(same)),
            "status": "PASS" if ov and all(same) else ("MISSING" if not ov else "FAIL")}


def analyse_set(base, ref, rnd, cam_rows):
    cams = [c for c in CAMS if cam_rows.get(c)]
    common = [c for c in base if all(c in cam_rows[k] for k in cams)]
    B = [base[c] for c in common]
    res = {"n_paired": len(common), "cameras": {}, "pairwise": {}}
    for cam in cams:
        X = [cam_rows[cam][c] for c in common]
        r = {"G0": g0_overlap(cam_rows[cam], rnd, cam), "horizons": {}}
        for h in ab.HORIZONS:
            r["horizons"][h] = {kind: ab.paired([ab.metric(b, kind, h) for b in B],
                                                [ab.metric(x, kind, h) for x in X])
                                for kind in ("ade", "fde")}
        r["tail"] = {kind: ab.tail([ab.metric(x, kind, "6.4s") - ab.metric(b, kind, "6.4s")
                                    for b, x in zip(B, X)]) for kind in ("ade", "fde")}
        r["coc"] = {"degenerate_base": float(np.mean([b["coc_degenerate"] for b in B])),
                    "degenerate_blackout": float(np.mean([x["coc_degenerate"] for x in X])),
                    "identical_text": float(np.mean([b["gen_coc"] == x["gen_coc"]
                                                     for b, x in zip(B, X)]))}
        res["cameras"][CAMS[cam]] = r
    # camera vs camera on the same clips: delta = cam_a - cam_b
    for h in ("1.6s", "6.4s"):
        for kind in ("ade", "fde"):
            fam = {}
            for a, b in itertools.combinations(cams, 2):
                xa = np.array([ab.metric(cam_rows[a][c], kind, h) for c in common])
                xb = np.array([ab.metric(cam_rows[b][c], kind, h) for c in common])
                d = xa - xb
                lo, hi = ab.boot_ci(d)
                fam[f"{CAMS[a]}-{CAMS[b]}"] = {
                    "delta_mean": float(d.mean()), "ci": [lo, hi],
                    "delta_median": float(np.median(d)),
                    "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}
            for k, p in zip(fam, holm([v["wilcoxon_p"] for v in fam.values()])):
                fam[k]["holm_p"] = p
            res["pairwise"][f"{kind}_{h}"] = fam
    if len(cams) == len(CAMS):
        fam = res["pairwise"]["ade_6.4s"]
        wins = []
        for o in cams:
            if o == FRONT:
                continue
            key = f"{CAMS[min(FRONT, o)]}-{CAMS[max(FRONT, o)]}"
            v = fam[key]
            sign = 1 if FRONT < o else -1           # orient as front_wide - other
            lo, hi = sorted([sign * v["ci"][0], sign * v["ci"][1]])
            wins.append(lo > 0 and v["holm_p"] < 0.05)
        res["H2_front_wide_largest"] = bool(all(wins))
    if ref:
        rc = [c for c in common if c in ref]
        res["reference_dual"] = {
            "n": len(rc),
            **{h: {kind: ab.paired([ab.metric(base[c], kind, h) for c in rc],
                                   [ab.metric(ref[c], kind, h) for c in rc])
                   for kind in ("ade", "fde")} for h in ab.HORIZONS}}
    return res


def plot_relative(res, out):
    """Change vs baseline (%) per camera, three horizons, ADE and FDE."""
    fig, axes = plt.subplots(2, len(res), figsize=(6.2 * len(res), 8), squeeze=False)
    for j, (s, r) in enumerate(res.items()):
        for i, kind in enumerate(("ade", "fde")):
            ax = axes[i][j]
            x = np.arange(len(ab.HORIZONS))
            n = len(r["cameras"]) + ("reference_dual" in r)
            offs = np.linspace(-0.3, 0.3, n)
            series = [(CAMS_INV[name], name, v["horizons"]) for name, v in r["cameras"].items()]
            for k, (cam, name, src) in enumerate(series):
                base = [src[h][kind]["base_mean"] for h in ab.HORIZONS]
                m = [100 * src[h][kind]["delta_mean"] / b for h, b in zip(ab.HORIZONS, base)]
                lo = [m[q] - 100 * src[h][kind]["ci"][0] / base[q] for q, h in enumerate(ab.HORIZONS)]
                hi = [100 * src[h][kind]["ci"][1] / base[q] - m[q] for q, h in enumerate(ab.HORIZONS)]
                ax.errorbar(x + offs[k], m, yerr=[lo, hi], fmt="o", color=COLORS[cam], capsize=3,
                            label=name)
            if "reference_dual" in r:
                src = r["reference_dual"]
                base = [src[h][kind]["base_mean"] for h in ab.HORIZONS]
                m = [100 * src[h][kind]["delta_mean"] / b for h, b in zip(ab.HORIZONS, base)]
                lo = [m[q] - 100 * src[h][kind]["ci"][0] / base[q] for q, h in enumerate(ab.HORIZONS)]
                hi = [100 * src[h][kind]["ci"][1] / base[q] - m[q] for q, h in enumerate(ab.HORIZONS)]
                ax.errorbar(x + offs[-1], m, yerr=[lo, hi], fmt="s", color=ab.MUTED, capsize=3,
                            label="ref: 24% pruning (dual)")
            ax.axhline(0, color=ab.MUTED, lw=0.8)
            ax.set_xticks(x, list(ab.HORIZONS))
            ax.set_ylabel(f"min{kind.upper()}@6 change vs baseline (%), 95% CI")
            ax.set_title(f"{s} (n={r['n_paired']}): min{kind.upper()}@6")
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "relative_by_camera.png", dpi=150)
    plt.close(fig)


def plot_absolute(res, out):
    """Absolute delta (m) per camera at 1.6 s and 6.4 s."""
    fig, axes = plt.subplots(2, len(res), figsize=(6.2 * len(res), 8), squeeze=False)
    for j, (s, r) in enumerate(res.items()):
        for i, h in enumerate(("1.6s", "6.4s")):
            ax = axes[i][j]
            names = list(r["cameras"])
            x = np.arange(len(names))
            for off, kind, col in ((-0.12, "ade", ab.C1), (0.12, "fde", ab.C4)):
                m = [r["cameras"][n]["horizons"][h][kind]["delta_mean"] for n in names]
                lo = [m[q] - r["cameras"][n]["horizons"][h][kind]["ci"][0] for q, n in enumerate(names)]
                hi = [r["cameras"][n]["horizons"][h][kind]["ci"][1] - m[q] for q, n in enumerate(names)]
                ax.errorbar(x + off, m, yerr=[lo, hi], fmt="o", color=col, capsize=4,
                            label=f"min{kind.upper()}@6")
            ax.axhline(0, color=ab.MUTED, lw=0.8)
            ax.set_xticks(x, names)
            ax.set_ylabel("blackout - baseline (m), mean, 95% CI")
            ax.set_title(f"{s}: {h}")
            ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "delta_by_camera_fixed.png", dpi=150)
    plt.close(fig)


CAMS_INV = {v: k for k, v in CAMS.items()}


def main():
    out = OUT / "blackout_cam_analysis"
    (out / "plots").mkdir(parents=True, exist_ok=True)
    res = {}
    for s, (base_id, suffix, rnd_id) in SETS.items():
        cam_rows = {cam: ab.load_rows(OUT / f"blackout_cam{cam}_{suffix}", f"baseline_bkcam{cam}")
                    for cam in CAMS}
        if not any(cam_rows.values()):
            print(f"{s}: no per-camera rows yet")
            continue
        res[s] = analyse_set(ab.load_rows(OUT / base_id, "baseline"),
                             ab.load_rows(OUT / ab.REF[s], ab.REF_TAG),
                             ab.load_rows(OUT / rnd_id, "baseline_bk1"), cam_rows)
        res[s]["n_rows_per_camera"] = {CAMS[c]: len(v) for c, v in cam_rows.items()}
    if not res:
        sys.exit("nothing to analyse")
    plot_relative(res, out / "plots")
    plot_absolute(res, out / "plots")

    (out / "config.json").write_text(json.dumps({
        "sets": {s: {"baseline": v[0], "per_camera": f"blackout_cam{{idx}}_{v[1]}",
                     "random_run": v[2]} for s, v in SETS.items()},
        "cameras": CAMS, "tag": "baseline_bkcam{idx}", "k": ab.K, "n_boot": ab.N_BOOT,
        "reference": {s: [ab.REF[s], ab.REF_TAG] for s in ab.REF}, "multiple_testing": "Holm",
        "plan": "plans/2026-10-18_camera-blackout-openloop.md",
    }, indent=2))
    (out / "metrics.json").write_text(json.dumps(res, indent=2))

    lines = ["per-camera t0 blackout (one fixed camera on every clip) vs baseline, open loop", ""]
    for s, r in res.items():
        lines.append(f"== {s}: {r['n_paired']} clips common to all cameras | rows {r['n_rows_per_camera']}"
                     + (f" | H2 front_wide largest: {r['H2_front_wide_largest']}"
                        if "H2_front_wide_largest" in r else ""))
        for name, v in r["cameras"].items():
            lines.append(f"  [{name}] G0' {v['G0']} | CoC identical {v['coc']['identical_text']:.3f}, "
                         f"degenerate {v['coc']['degenerate_base']:.3f} -> {v['coc']['degenerate_blackout']:.3f}")
            for h in ab.HORIZONS:
                for kind in ("ade", "fde"):
                    lines.append(f"    min{kind.upper()}@6 {h}: {ab.fmt_p(v['horizons'][h][kind])}")
            for kind, t in v["tail"].items():
                lines.append(f"    6.4s {kind} 10% trimmed {t['trim10_mean']:+.4f} "
                             f"[{t['trim10_ci'][0]:+.4f}, {t['trim10_ci'][1]:+.4f}]")
        for fam, d in r["pairwise"].items():
            lines.append(f"  camera vs camera, {fam} (a - b):")
            for k, v in d.items():
                lines.append(f"    {k:32s} {v['delta_mean']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] "
                             f"median {v['delta_median']:+.4f} p={v['wilcoxon_p']:.3g} holm={v['holm_p']:.3g}")
        if "reference_dual" in r:
            for h in ab.HORIZONS:
                for kind in ("ade", "fde"):
                    lines.append(f"  ref dual min{kind.upper()}@6 {h}: "
                                 f"{ab.fmt_p(r['reference_dual'][h][kind])}")
        lines.append("")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
