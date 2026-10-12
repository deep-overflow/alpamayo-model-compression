"""How much does a clip's Taylor importance depend on which CoC it sampled?

Reads `run_coc_importance.py`: per clip, the shipped dual-objective importance under 6
CoCs (C0..C5, FM noise held) and under 3 FM-noise reseeds of CoC 0 (N1..N3).

Three levels, each read against the N floor:

  clip, score      within-layer Spearman of the (L, U) map between C0 and each other
                   condition, averaged over layers that carry signal.
  clip, selection  the dual kept set a clip would choose alone -- max(rank I_traj,
                   rank I_CoC), 19/32 Q heads and 7390/12288 MLP channels per layer --
                   and its overlap between C0 and each other condition.
  set, selection   the calibration set's real selection. A set's importance is the mean
                   of its 100 clips, so swapping one clip's condition is exactly
                       I_set' = I_set + (I_clip[cond] - I_clip[C0]) / 100
                   with I_set the stored importance.npz of that set. Each measured clip
                   draws a random alternative CoC (or a random noise reseed, the floor),
                   and the overlap of the resulting kept set with the stored one is
                   read against the two scales this repo has measured: 97.2% (same
                   clips, other GPU architecture -- harmless) and 91.3% / 85.3% (two
                   disjoint n=100 draws).

The base selection is checked first: dual from importance_v2 must reproduce
slim_dual_u40_v2's kept indices unit for unit.

Usage:
  python experiments/evaluation/analyze_coc_importance.py [--out outputs/coc_importance]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

from analyze_coc_multimode import SETS, load

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

KEYS = ["coc_vlm_q", "coc_vlm_mlp", "traj_vlm_q", "traj_vlm_mlp"]
N_COC, N_NOISE = 6, 3
DROP = {"q": 13, "mlp": 4898}           # per layer, dual_u40_v2's realized cut
IMP = {"calib_100": "importance_v2", **{x: f"importance_nt500_{x}" for x in "abcde"}}
DRAWS = 200
REF = {"GPU architecture, same clips": (0.9722, 0.9729),
       "two disjoint n=100 draws": (0.9134, 0.8530)}


def rank_norm(s):
    """Per-layer rank in [0, 1]. (L, U) -> (L, U); verbatim from run_cocsafe."""
    return np.argsort(np.argsort(s, axis=1), axis=1) / max(s.shape[1] - 1, 1)


def dual_mask(traj, coc, axis):
    """Keep-mask of max(rank I_traj, rank I_CoC), dropping DROP[axis] units per layer."""
    s = np.maximum(rank_norm(traj), rank_norm(coc))
    m = np.ones(s.shape, bool)
    np.put_along_axis(m, np.argsort(s, axis=1)[:, :DROP[axis]], False, axis=1)
    return m


def overlap(a, b):
    return float((a & b).sum() / a.sum())


def layer_rho(a, b):
    """Mean within-layer Spearman over layers where both maps vary."""
    r = [spearmanr(a[li], b[li])[0] for li in range(a.shape[0])
         if a[li].std() > 0 and b[li].std() > 0]
    return float(np.mean(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/coc_importance")
    args = ap.parse_args()
    out = Path(args.out) if Path(args.out).is_absolute() else REPO / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    recs, A = [], {k: [] for k in KEYS}
    for p in sorted(out.glob("records_s*of*.json")):
        recs += json.loads(p.read_text())
        z = np.load(out / p.name.replace("records_", "imp_").replace(".json", ".npz"))
        for k in KEYS:
            A[k].append(z[k])
    A = {k: np.concatenate(v) for k, v in A.items()}      # (n_clip, 9, L, U)
    n = len(recs)
    assert A[KEYS[0]].shape[0] == n and A[KEYS[0]].shape[1] == N_COC + N_NOISE
    group = np.array([r["group"] for r in recs])
    sets = np.array([r["set"] for r in recs])
    say(f"{n} clips: {int((group == 'high').sum())} high, {int((group == 'control').sum())} control")

    # G0: the CoC regenerated here is the one run_coc_multimode stored
    ref = {r["clip_id"]: r for _, stem in SETS for r in load(stem)}
    same = sum(c["text"] == ref[r["clip_id"]]["cocs"][j]["text"]
               for r in recs for j, c in enumerate(r["conds"][:N_COC]))
    say(f"G0 CoC text reproduced: {same}/{n * N_COC}")
    # which alternatives are a genuinely different sentence from C0
    diff = np.array([[r["conds"][j]["text"] != r["conds"][0]["text"] for j in range(1, N_COC)]
                     for r in recs])                       # (n, 5)
    say(f"alternative CoCs that differ from C0 in text: high {diff[group == 'high'].mean():.2f}, "
        f"control {diff[group == 'control'].mean():.2f}")

    # base selection check
    base = {s: dict(np.load(REPO / "outputs" / IMP[s] / "importance.npz")) for s, _ in SETS}
    meta = json.loads((REPO / "outputs" / "slim_dual_u40_v2" / "slim_meta.json").read_text())
    bm = {s: {ax: dual_mask(base[s][f"traj_vlm_{ax}"], base[s][f"coc_vlm_{ax}"], ax)
              for ax in ("q", "mlp")} for s, _ in SETS}
    ok = all(sorted(np.flatnonzero(bm["calib_100"][ax][li]).tolist()) == meta["vlm"][li][ax]
             for ax in ("q", "mlp") for li in range(36))
    say(f"base selection reproduces slim_dual_u40_v2: {ok}")
    metrics = {"n_clips": n, "g0_text": [same, n * N_COC], "base_selection_reproduced": bool(ok),
               "clip": {}, "set": {}}

    # ---- clip level ----
    conds_c = list(range(1, N_COC))
    conds_n = list(range(N_COC, N_COC + N_NOISE))
    rho = {k: np.zeros((n, N_COC + N_NOISE)) for k in KEYS}
    ov = {ax: np.zeros((n, N_COC + N_NOISE)) for ax in ("q", "mlp")}
    for i in range(n):
        for k in KEYS:
            for c in range(1, N_COC + N_NOISE):
                rho[k][i, c] = layer_rho(A[k][i, 0], A[k][i, c])
        for ax in ("q", "mlp"):
            m0 = dual_mask(A[f"traj_vlm_{ax}"][i, 0], A[f"coc_vlm_{ax}"][i, 0], ax)
            for c in range(1, N_COC + N_NOISE):
                ov[ax][i, c] = overlap(m0, dual_mask(A[f"traj_vlm_{ax}"][i, c],
                                                     A[f"coc_vlm_{ax}"][i, c], ax))

    def cstat(x, g):
        """per-clip mean over CoC alternatives, over CoC alternatives whose text differs,
        and over noise reseeds -- then the group median of each"""
        sel = group == g
        coc = x[sel][:, conds_c].mean(1)
        d = diff[sel]
        cd = np.array([x[sel][i, conds_c][d[i]].mean() if d[i].any() else np.nan
                       for i in range(sel.sum())])
        noi = x[sel][:, conds_n].mean(1)
        p = wilcoxon(coc, noi).pvalue
        return {"coc": float(np.median(coc)), "coc_textdiff": float(np.nanmedian(cd)),
                "noise": float(np.median(noi)), "p_coc_vs_noise": float(p),
                "frac_below_noise": float((coc < noi).mean())}, coc, noi

    say("\n== clip level: agreement with the C0 map (group median of per-clip means)")
    say(f"{'':26s}{'group':>9s}{'other CoC':>11s}{'text differs':>14s}{'noise only':>12s}"
        f"{'p':>10s}{'CoC<noise':>11s}")
    per_clip = {}
    for label, x in ([(f"rho {k}", rho[k]) for k in KEYS]
                     + [(f"dual kept overlap {ax}", ov[ax]) for ax in ("q", "mlp")]):
        metrics["clip"][label] = {}
        for g in ("high", "control"):
            s, coc, noi = cstat(x, g)
            metrics["clip"][label][g] = s
            per_clip[(label, g)] = (coc, noi)
            say(f"{label:26s}{g:>9s}{s['coc']:11.4f}{s['coc_textdiff']:14.4f}{s['noise']:12.4f}"
                f"{s['p_coc_vs_noise']:10.1e}{s['frac_below_noise']:11.2f}")
        p = mannwhitneyu(per_clip[(label, "high")][0], per_clip[(label, "control")][0]).pvalue
        metrics["clip"][label]["p_high_vs_control"] = float(p)
        say(f"{'':26s}{'':>9s}high vs control (other CoC) p = {p:.1e}")

    # is the change tied to how far the CoC moved the trajectory?
    clips_meta = {c["clip_id"]: c for c in json.loads((out / "clips.json").read_text())}
    btw = np.array([clips_meta[r["clip_id"]]["end_rms_between"] for r in recs])
    say("\n== pooled over clips: Spearman(between-CoC endpoint RMS, agreement with C0)")
    metrics["rho_vs_between"] = {}
    for label, x in ([(f"rho {k}", rho[k]) for k in KEYS]
                     + [(f"dual kept overlap {ax}", ov[ax]) for ax in ("q", "mlp")]):
        r, p = spearmanr(btw, x[:, conds_c].mean(1))
        metrics["rho_vs_between"][label] = [float(r), float(p)]
        say(f"{label:26s} rho {r:+.3f}  p {p:.1e}")

    # ---- set level ----
    def substituted(s, idx, cond):
        """Set importance with clip idx[i] moved from C0 to cond[i]; exact, see docstring."""
        m = {}
        for ax in ("q", "mlp"):
            t = base[s][f"traj_vlm_{ax}"].astype(np.float64).copy()
            c = base[s][f"coc_vlm_{ax}"].astype(np.float64).copy()
            for i, cd in zip(idx, cond):
                t += (A[f"traj_vlm_{ax}"][i, cd].astype(np.float64)
                      - A[f"traj_vlm_{ax}"][i, 0]) / 100
                c += (A[f"coc_vlm_{ax}"][i, cd].astype(np.float64)
                      - A[f"coc_vlm_{ax}"][i, 0]) / 100
            m[ax] = dual_mask(t, c, ax)
        return m

    def mean_substituted(s, idx):
        """Every measured clip scored as the mean over its 6 CoCs instead of C0 alone."""
        m = {}
        for ax in ("q", "mlp"):
            t = base[s][f"traj_vlm_{ax}"].astype(np.float64).copy()
            c = base[s][f"coc_vlm_{ax}"].astype(np.float64).copy()
            for i in idx:
                t += (A[f"traj_vlm_{ax}"][i, :N_COC].astype(np.float64).mean(0)
                      - A[f"traj_vlm_{ax}"][i, 0]) / 100
                c += (A[f"coc_vlm_{ax}"][i, :N_COC].astype(np.float64).mean(0)
                      - A[f"coc_vlm_{ax}"][i, 0]) / 100
            m[ax] = dual_mask(t, c, ax)
        return m

    say(f"\n== set level: kept-set overlap with the stored selection after swapping the "
        f"measured clips ({DRAWS} random assignments; mean [min])")
    say("reference scales: " + "; ".join(f"{k} Q {q:.1%} / MLP {m:.1%}" for k, (q, m) in REF.items()))
    say(f"{'set':10s}{'clips':>7s}{'which':>10s}" + "".join(
        f"{h:>20s}" for h in ("CoC swap Q", "noise swap Q", "CoC swap MLP", "noise swap MLP"))
        + f"{'6-CoC mean Q':>14s}{'MLP':>8s}")
    for s, _ in SETS:
        metrics["set"][s] = {}
        for which, sel in (("high", (sets == s) & (group == "high")), ("all", sets == s)):
            idx = np.flatnonzero(sel)
            res = {"coc": {"q": [], "mlp": []}, "noise": {"q": [], "mlp": []}}
            for _ in range(DRAWS):
                for kind, pool in (("coc", conds_c), ("noise", conds_n)):
                    m = substituted(s, idx, rng.choice(pool, len(idx)))
                    for ax in ("q", "mlp"):
                        res[kind][ax].append(overlap(bm[s][ax], m[ax]))
            mm = mean_substituted(s, idx)
            row = {f"{kind}_{ax}": {"mean": float(np.mean(res[kind][ax])),
                                    "min": float(np.min(res[kind][ax]))}
                   for kind in res for ax in ("q", "mlp")}
            row["mean6_q"], row["mean6_mlp"] = overlap(bm[s]["q"], mm["q"]), overlap(bm[s]["mlp"], mm["mlp"])
            row["n_swapped"] = len(idx)
            metrics["set"][s][which] = row
            say(f"{s:10s}{len(idx):7d}{which:>10s}" + "".join(
                f"{row[k]['mean']:11.4f} [{row[k]['min']:.4f}]"
                for k in ("coc_q", "noise_q", "coc_mlp", "noise_mlp"))
                + f"{row['mean6_q']:14.4f}{row['mean6_mlp']:8.4f}")

    # extrapolation: per swapped clip, how much of the kept set moves
    say("\n== units moved per swapped clip (set level, 'all' rows): (1 - overlap) / n_swapped")
    for kind in ("coc", "noise"):
        q = np.mean([(1 - metrics["set"][s]["all"][f"{kind}_q"]["mean"])
                     / metrics["set"][s]["all"]["n_swapped"] for s, _ in SETS])
        m = np.mean([(1 - metrics["set"][s]["all"][f"{kind}_mlp"]["mean"])
                     / metrics["set"][s]["all"]["n_swapped"] for s, _ in SETS])
        metrics[f"per_clip_move_{kind}"] = {"q": float(q), "mlp": float(m)}
        say(f"{kind:6s} Q {q:.5f}  MLP {m:.5f}  per clip")

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "config.json").write_text(json.dumps({
        "keys": KEYS, "n_coc": N_COC, "n_noise": N_NOISE, "drop_per_layer": DROP,
        "base_importance": IMP, "draws": DRAWS, "reference_overlaps": REF}, indent=2))

    # plots
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    for ax_, k in zip(axes, KEYS):
        data, cols, labs = [], [], []
        for g, c in (("high", C3), ("control", C1)):
            coc, noi = per_clip[(f"rho {k}", g)]
            data += [coc, noi]
            cols += [c, MUTED]
            labs += [f"{g}\nother CoC", f"{g}\nnoise"]
        bp = ax_.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                         medianprops={"color": INK})
        for b, c in zip(bp["boxes"], cols):
            b.set_facecolor(c)
            b.set_alpha(0.8)
        ax_.set_xticks(range(1, 5), labs, fontsize=7.5)
        ax_.set_title(k)
    axes[0].set_ylabel("within-layer Spearman vs C0")
    fig.tight_layout()
    fig.savefig(out / "plots" / "clip_rho.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    xs = np.arange(len(SETS))
    for ax_, axn, ri in zip(axes, ("q", "mlp"), (0, 1)):
        for off, kind, c in ((-0.2, "coc", C3), (0.2, "noise", MUTED)):
            ax_.bar(xs + off, [1 - metrics["set"][s]["all"][f"{kind}_{axn}"]["mean"] for s, _ in SETS],
                    width=0.38, color=c, label=f"{kind} swap, all measured clips")
            ax_.plot(xs + off, [1 - metrics["set"][s]["high"][f"{kind}_{axn}"]["mean"] for s, _ in SETS],
                     "o", color=INK, ms=4, label="high clips only" if kind == "coc" else None)
        for (lab, v), ls in zip(REF.items(), ("--", ":")):
            ax_.axhline(1 - v[ri], color=INK, lw=0.8, ls=ls, label=lab)
        ax_.set_xticks(xs, [s for s, _ in SETS])
        ax_.set_title(f"{'Q heads' if axn == 'q' else 'MLP channels'}: share of the kept set that moves")
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "plots" / "set_churn.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax_, k in zip(axes, ("coc_vlm_mlp", "traj_vlm_mlp")):
        y = rho[k][:, conds_c].mean(1)
        for g, c in (("high", C3), ("control", C1)):
            ax_.scatter(btw[group == g], y[group == g], s=14, color=c, alpha=0.8, label=g)
        ax_.set_xscale("log")
        ax_.set_xlabel("between-CoC endpoint RMS [m]")
        ax_.set_ylabel("within-layer Spearman vs C0")
        ax_.set_title(k)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "plots" / "rho_vs_between.png", dpi=150)
    plt.close(fig)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
