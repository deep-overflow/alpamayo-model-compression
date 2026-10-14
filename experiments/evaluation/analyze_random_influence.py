"""Are the clips whose random draws move the selection the multimodal ones?

Hypothesis (user, 2026-10-14): the importance measurement's random draws -- which CoC is
sampled, which flow-matching noise is drawn -- matter because of multimodality with a single
GT. On a clip where the model has several plausible futures but only one is the GT, the
flow-matching loss against that GT, and so its gradient, depends on which noise draw lands
near which mode; such clips would inject the noise that flips the layer-0 head choice.

Data, no GPU: `run_coc_importance.py` scored 124 clips (6 calibration sets; 62 picked for
large between-CoC trajectory spread, 62 drawn at random) under the shipped seeds (C0), five
other CoC seeds (C1-C5, noise held) and three other noise seeds (N1-N3, CoC held).
Multimodality comes from `run_coc_multimode.py` on the same clips (6 CoC x 6 paths).

Influence of clip c under condition k = what its own redraw does to ITS set's selection:
    I_set' = I_set + (I_c[k] - I_c[C0]) / 100        (exact: a set's importance is the mean)
  flips0    layer-0 Q heads whose kept status changes (the place collapses come from)
  risky     change in how many of the heads that hurt when dropped (4, 7, 24, 30) are cut
  lever0    sum_h |I_c[k] - I_c[C0]| at layer 0, as a share of the set's layer-0 total --
            the clip's pull on layer 0 regardless of how close that set's cut happens to be
noise influence = mean over N1-N3; CoC influence = mean over C1-C5.

calib_100 also gets a direct decomposition of the seed arm that collapsed (`n1`, noise seed 1
drops layer-0 head 30): the 25 calib_100 clips measured here are compared with the whole
change between importance_seed_n1 and importance_seed_ref (both Ada, same seed rule).

Usage:
  python experiments/evaluation/analyze_random_influence.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

REPO = Path(__file__).resolve().parents[2]
O = REPO / "outputs"
sys.path.insert(0, str(Path(__file__).parent))

from analyze_coc_multimode import SETS, load
from analyze_coc_multimode import clip_metrics as mm_metrics

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C3, C4 = "#2a78d6", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})
BASE = {"calib_100": "importance_seed_ref", **{x: f"importance_nt500_{x}" for x in "abcde"}}
RISKY = (4, 7, 24, 30)          # heads whose removal hurt (single-head map, census)
DROP_Q = 13
NC, NN = 5, 3                   # C1-C5, N1-N3; index 0 is C0


def rank_norm(x):
    """(..., U) -> within-row rank in [0, 1]; verbatim rule of run_cocsafe.rank_norm"""
    return np.argsort(np.argsort(x, axis=-1), axis=-1) / (x.shape[-1] - 1)


def kept(traj, coc):
    """dual selection on (L, H) Q scores -> list of kept sets, 13 dropped per layer"""
    s = np.maximum(rank_norm(traj), rank_norm(coc))
    return [set(np.argsort(row)[DROP_Q:].tolist()) for row in s]


def main():
    out = O / "coc_importance"
    (out / "plots").mkdir(parents=True, exist_ok=True)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    recs, A = [], {"traj": [], "coc": []}
    for i in range(4):
        recs += json.loads((out / f"records_s{i}of4.json").read_text())
        z = np.load(out / f"imp_s{i}of4.npz")
        A["traj"].append(z["traj_vlm_q"])
        A["coc"].append(z["coc_vlm_q"])
    A = {k: np.concatenate(v).astype(np.float64) for k, v in A.items()}   # (124, 9, 36, 32)
    n = len(recs)
    base = {s: dict(np.load(O / d / "importance.npz")) for s, d in BASE.items()}
    base_keep = {s: kept(b["traj_vlm_q"], b["coc_vlm_q"]) for s, b in base.items()}

    # ---- influence of each clip's own redraw on its set's selection ----
    flips0 = np.zeros((n, 9))
    flips_all = np.zeros((n, 9))
    risky = np.zeros((n, 9))
    lever_t = np.zeros((n, 9))
    lever_c = np.zeros((n, 9))
    drops30 = np.zeros((n, 9))
    for i, r in enumerate(recs):
        b = base[r["set"]]
        k0 = base_keep[r["set"]]
        for k in range(1, 9):
            dt = (A["traj"][i, k] - A["traj"][i, 0]) / 100
            dc = (A["coc"][i, k] - A["coc"][i, 0]) / 100
            kk = kept(b["traj_vlm_q"] + dt, b["coc_vlm_q"] + dc)
            flips0[i, k] = len(k0[0] - kk[0])
            flips_all[i, k] = sum(len(a - c) for a, c in zip(k0, kk))
            risky[i, k] = sum(h not in kk[0] for h in RISKY) - sum(h not in k0[0] for h in RISKY)
            drops30[i, k] = float(30 in k0[0] and 30 not in kk[0])
            lever_t[i, k] = np.abs(dt[0]).sum() / b["traj_vlm_q"][0].sum()
            lever_c[i, k] = np.abs(dc[0]).sum() / b["coc_vlm_q"][0].sum()
    cs, ns = slice(1, 1 + NC), slice(1 + NC, 9)
    infl = {
        "noise_flips0": flips0[:, ns].mean(1), "coc_flips0": flips0[:, cs].mean(1),
        "noise_lever_traj": lever_t[:, ns].mean(1), "coc_lever_traj": lever_t[:, cs].mean(1),
        "coc_lever_coc": lever_c[:, cs].mean(1),
        "noise_risky": risky[:, ns].mean(1), "coc_risky": risky[:, cs].mean(1),
        "noise_flips_all": flips_all[:, ns].mean(1), "coc_flips_all": flips_all[:, cs].mean(1),
    }
    fm = np.array([[c["fm_loss"] for c in r["conds"]] for r in recs])          # (n, 9)
    fm_noise_cv = fm[:, [0, 6, 7, 8]].std(1, ddof=1) / fm[:, [0, 6, 7, 8]].mean(1)

    # ---- multimodality of the same clips ----
    rows = {}
    for _, stem in SETS:
        for row in load(stem):
            rows[row["clip_id"]] = row
    mmk = ["end_rms_within", "end_rms_between", "ade_sd", "fde_sd", "n_head", "n_text", "nll",
           "ade_mean36", "minade36", "minade6"]
    mm = {k: [] for k in mmk}
    for r in recs:
        m = mm_metrics(rows[r["clip_id"]])
        for k in mmk:
            mm[k].append(float(m[k]))
    mm = {k: np.array(v) for k, v in mm.items()}
    mm["gt_gap"] = mm["ade_mean36"] - mm["minade36"]     # typical sample vs best sample to the GT
    mm["fm_noise_cv"] = fm_noise_cv
    mmk = mmk + ["gt_gap", "fm_noise_cv"]
    group = np.array([r["group"] for r in recs])
    sets = np.array([r["set"] for r in recs])
    bucket = np.array([rows[r["clip_id"]]["bucket"] for r in recs])

    say(f"{n} clips ({(group == 'high').sum()} picked for large between-CoC spread, "
        f"{(group == 'control').sum()} random), 6 sets; base selections from {BASE}")
    say("\n== how often does one clip's own redraw change its set's selection?")
    for lab, src, sl in (("noise seed", flips0, ns), ("CoC seed", flips0, cs)):
        f = src[:, sl]
        say(f"{lab:10s}: layer-0 heads swapped per redraw mean {f.mean():.3f}; redraws that swap any "
            f"layer-0 head {np.mean(f > 0):.1%}; clips with >= 1 such redraw "
            f"{np.mean((f > 0).any(1)):.1%}; redraws that newly cut head 30 "
            f"{drops30[:, sl].mean():.1%}")
    for lab, sl in (("noise seed", ns), ("CoC seed", cs)):
        say(f"{lab:10s}: Q heads swapped over all 36 layers per redraw mean {flips_all[:, sl].mean():.2f}; "
            f"layer 0's share {flips0[:, sl].sum() / max(flips_all[:, sl].sum(), 1):.1%}")
    say("\nconcentration: share of the total layer-0 pull (lever0) carried by the top 10% of clips")
    for k in ("noise_lever_traj", "coc_lever_traj", "coc_lever_coc"):
        v = np.sort(infl[k])[::-1]
        say(f"  {k:18s} top 12 of {n}: {v[:12].sum() / v.sum():.1%}   (median clip {np.median(v):.2e}, max {v[0]:.2e})")

    metrics = {"n": n, "rho": {}, "top_vs_rest": {}}
    say("\n== Spearman(influence, multimodality); pooled n=124 | random clips only n=62")
    head = f"{'influence':18s}" + "".join(f"{k:>16s}" for k in mmk)
    for subset, sel in (("pooled", np.ones(n, bool)), ("random only", group == "control")):
        say(f"\n[{subset}]")
        say(head)
        for ik in ("noise_lever_traj", "noise_flips0", "coc_lever_traj", "coc_lever_coc", "coc_flips0"):
            cells = []
            for mk in mmk:
                r_, p_ = spearmanr(infl[ik][sel], mm[mk][sel])
                metrics["rho"][f"{subset}|{ik}|{mk}"] = [float(r_), float(p_)]
                star = "**" if p_ < 0.001 else "*" if p_ < 0.05 else ""
                cells.append(f"{r_:+.2f}{star:2s}")
            say(f"{ik:18s}" + "".join(f"{c:>16s}" for c in cells))
    say("(* p<0.05, ** p<0.001; 12 multimodality measures x 5 influence measures per block)")

    # top-influence clips vs the rest
    say("\n== the 12 clips with the largest noise pull on layer 0 vs the other 112 (medians)")
    top = np.argsort(infl["noise_lever_traj"])[::-1][:12]
    rest = np.setdiff1d(np.arange(n), top)
    for mk in mmk:
        p_ = mannwhitneyu(mm[mk][top], mm[mk][rest]).pvalue
        metrics["top_vs_rest"][mk] = [float(np.median(mm[mk][top])), float(np.median(mm[mk][rest])), float(p_)]
        say(f"  {mk:16s} top {np.median(mm[mk][top]):8.3f}   rest {np.median(mm[mk][rest]):8.3f}   p={p_:.3f}")
    say(f"  group of the top 12: {dict(zip(*np.unique(group[top], return_counts=True)))}; "
        f"bucket: {dict(zip(*np.unique(bucket[top], return_counts=True)))}")

    # does the noise pull follow the GT-relative spread more than the CoC-level spread? partial view
    say("\n== by bucket (median noise pull on layer 0, median within-CoC endpoint spread)")
    for bk in ("cruise", "decel_stop", "accel", "turn"):
        sel = bucket == bk
        say(f"  {bk:11s} n={int(sel.sum()):3d}  noise_lever_traj {np.median(infl['noise_lever_traj'][sel]):.2e}"
            f"  end_rms_within {np.median(mm['end_rms_within'][sel]):.2f} m  fm_noise_cv {np.median(fm_noise_cv[sel]):.3f}")

    # ---- is the noise pull scale or instability? lever0 = weight x relative change, exactly ----
    # weight: the clip's own share of its set's layer-0 I_traj; relative: how much of its OWN
    # layer-0 I_traj a noise redraw moves; rank_rho: Spearman of its layer-0 map, C0 vs redraw
    w = np.array([A["traj"][i, 0, 0].sum() / 100 / base[r["set"]]["traj_vlm_q"][0].sum()
                  for i, r in enumerate(recs)])
    rel = np.array([np.mean([np.abs(A["traj"][i, k, 0] - A["traj"][i, 0, 0]).sum()
                             / A["traj"][i, 0, 0].sum() for k in range(1 + NC, 9)])
                    for i in range(n)])
    rrho = np.array([np.mean([spearmanr(A["traj"][i, 0, 0], A["traj"][i, k, 0])[0]
                              for k in range(1 + NC, 9)]) for i in range(n)])
    say("\n== the noise pull split into the clip's weight and its own relative instability")
    say(f"  check: lever0 == weight x relative, max abs diff "
        f"{np.abs(infl['noise_lever_traj'] - w * rel).max():.1e}")
    r_, p_ = spearmanr(w, rel)
    say(f"  Spearman(weight, relative) {r_:+.2f} (p={p_:.1e}); median weight {np.median(w):.4f} "
        f"(1/100 = 0.0100), median relative change {np.median(rel):.3f}, median layer-0 rank rho "
        f"{np.median(rrho):.3f}")
    say(f"{'':22s}" + "".join(f"{k:>16s}" for k in mmk))
    metrics["scale_split"] = {}
    for lab, v in (("weight (scale)", w), ("relative change", rel), ("1 - rank rho", 1 - rrho)):
        for subset, sel in (("pooled", np.ones(n, bool)), ("random", group == "control")):
            cells = []
            for mk in mmk:
                r_, p_ = spearmanr(v[sel], mm[mk][sel])
                metrics["scale_split"][f"{subset}|{lab}|{mk}"] = [float(r_), float(p_)]
                star = "**" if p_ < 0.001 else "*" if p_ < 0.05 else ""
                cells.append(f"{r_:+.2f}{star:2s}")
            say(f"{lab[:14] + ' ' + subset:22s}" + "".join(f"{c:>16s}" for c in cells))

    # ---- calib_100: decompose the collapse of seed arm n1 ----
    say("\n== calib_100: what carried n1's layer-0 change (noise seed 1 cuts head 30, keeps head 20)?")
    ref = dict(np.load(O / "importance_seed_ref" / "importance.npz"))
    n1 = dict(np.load(O / "importance_seed_n1" / "importance.npz"))
    tot = n1["traj_vlm_q"][0] - ref["traj_vlm_q"][0]                          # (32,)
    idx = np.flatnonzero(sets == "calib_100")
    part = ((A["traj"][idx, 6, 0] - A["traj"][idx, 0, 0]) / 100)              # (m, 32), N1 = column 6
    rt_ref, rt_n1 = rank_norm(ref["traj_vlm_q"][0]), rank_norm(n1["traj_vlm_q"][0])
    say(f"  rank I_traj of head 30: ref {rt_ref[30]:.2f} -> n1 {rt_n1[30]:.2f}; head 20: {rt_ref[20]:.2f} -> {rt_n1[20]:.2f}")
    gap_tot = tot[30] - tot[20]
    say(f"  change in I_traj(30) - I_traj(20), all 100 clips: {gap_tot:+.3e}")
    contrib = part[:, 30] - part[:, 20]
    say(f"  carried by the {len(idx)} calib_100 clips measured here: {contrib.sum():+.3e} "
        f"({contrib.sum() / gap_tot:.0%} of the total; they are {len(idx)}% of the clips)")
    order = np.argsort(contrib)
    say("  clips pushing head 30 below head 20 the most (most negative), with their multimodality:")
    say(f"  {'clip':10s}{'group':>9s}{'bucket':>11s}{'push':>11s}{'within':>8s}{'between':>9s}{'ade_sd':>8s}"
        f"{'n_head':>7s}{'nll':>7s}{'fm cv':>7s}")
    for j in order[:8]:
        g = idx[j]
        say(f"  {recs[g]['clip_id'][:8]:10s}{group[g]:>9s}{bucket[g]:>11s}{contrib[j]:+11.2e}"
            f"{mm['end_rms_within'][g]:8.2f}{mm['end_rms_between'][g]:9.2f}{mm['ade_sd'][g]:8.2f}"
            f"{int(mm['n_head'][g]):7d}{mm['nll'][g]:7.3f}{fm_noise_cv[g]:7.3f}")
    r_, p_ = spearmanr(-contrib, mm["end_rms_within"][idx])
    r2, p2 = spearmanr(-contrib, mm["end_rms_between"][idx])
    r3, p3 = spearmanr(np.abs(contrib), fm_noise_cv[idx])
    say(f"  Spearman over these {len(idx)} clips: push vs within-CoC spread {r_:+.2f} (p={p_:.2f}), "
        f"vs between-CoC spread {r2:+.2f} (p={p2:.2f}); |push| vs FM-loss noise CV {r3:+.2f} (p={p3:.2f})")
    metrics["n1_decomposition"] = {"gap_total": float(gap_tot), "gap_measured": float(contrib.sum()),
                                   "n_measured": len(idx)}

    (out / "metrics_influence.json").write_text(json.dumps(metrics, indent=2))
    (out / "summary_influence.txt").write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    col = np.where(group == "high", C3, C1)
    for ax, (mk, lab) in zip(axes, (("end_rms_within", "within-CoC endpoint spread [m]"),
                                    ("end_rms_between", "between-CoC endpoint spread [m]"),
                                    ("fm_noise_cv", "FM-loss CV over noise seeds"))):
        ax.scatter(mm[mk], infl["noise_lever_traj"], c=col, s=16, alpha=0.8)
        r_, p_ = spearmanr(mm[mk], infl["noise_lever_traj"])
        ax.set_xlabel(lab)
        ax.set_yscale("log")
        ax.set_title(f"rho {r_:+.2f} (p={p_:.1e})")
    axes[0].set_ylabel("noise pull on layer 0 (share of set's layer-0 I_traj)")
    axes[0].scatter([], [], c=C3, label="picked: large between-CoC spread")
    axes[0].scatter([], [], c=C1, label="random")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plots" / "influence_vs_multimodality.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
