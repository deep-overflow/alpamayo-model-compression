"""Layer-0 head choice as the source of collapses: reads the `make_l0_masks.py` variants.

  A  nt<x>_L0ship on test_cruise120: calibration block x's kept set with only layer 0's
     Q heads replaced by the shipped arm's. Paired with the block's own stored test500
     rows, the shipped arm and the unpruned model on the same 120 cruise clips.
  B  ship_L0drop<h> on test_cruise60: the shipped kept set with layer-0 head h dropped
     (head 20 restored in its place), one per kept head. Paired with the shipped arm.
     Joined with each head's rank under I_traj and I_CoC in importance_v2, so the map can
     be read against what the criterion believed about the head.

Usage:
  python experiments/evaluation/analyze_l0_masks.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

REPO = Path(__file__).resolve().parents[2]
BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C3, C4 = "#2a78d6", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})
BOOT, K = 20000, 6
# test500 mean minADE@6 of each block arm (2026-09-03 report), for orientation
ARM500 = {"a": 1.0219, "b": 1.0602, "c": 1.3479, "d": 1.7742, "e": 1.7178}


def load_run(exp):
    rows = {}
    for p in sorted((REPO / "outputs" / exp).glob("*.json")):
        if not p.name.startswith("config"):
            for r in json.loads(p.read_text()):
                rows[r["clip_id"]] = r
    return rows


def main():
    out = REPO / "outputs" / "swap_masks"
    (out / "plots").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    def ids_of(stem):
        return pd.read_parquet(REPO / "outputs" / "eval_sets" / f"{stem}.parquet").clip_id.tolist()

    def vec(rows, ids):
        return np.array([min(rows[c]["ade_rollout_k"][:K]) for c in ids])

    def ci(d):
        idx = rng.integers(0, len(d), (BOOT, len(d)))
        lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
        nz = d[d != 0]
        return (float(d.mean()), float(lo), float(hi), float(np.median(d)),
                float(wilcoxon(nz).pvalue) if len(nz) > 10 else 1.0)

    base, ship = load_run("baseline_ada_ps_test"), load_run("dual_u40_v2_ps_test")
    metrics = {"blocks": {}, "l0_map": {}}

    # ---- A: do the bad-draw collapses live in layer 0? ----
    ids = ids_of("test_cruise120")
    b, s = vec(base, ids), vec(ship, ids)
    say(f"== A. calibration blocks with layer 0's heads from the shipped arm ({len(ids)} cruise clips)")
    say(f"unpruned {b.mean():.4f}   shipped {s.mean():.4f}")
    say(f"{'block':6s}{'test500':>9s}{'own':>9s}{'L0 fixed':>10s}   fixed - own [95% CI]            "
        f"fixed - shipped [95% CI]         gap closed")
    for x in "abcde":
        own = vec(load_run(f"dual_nt_{x}_test"), ids)
        rows = load_run(f"swap_nt{x}_L0ship_cruise120")
        if not all(c in rows for c in ids):
            say(f"nt_{x}: missing ({len(rows)}/{len(ids)})")
            continue
        fx = vec(rows, ids)
        d1, d2 = ci(fx - own), ci(fx - s)
        closed = float((own.mean() - fx.mean()) / (own.mean() - s.mean()))
        metrics["blocks"][x] = {"own": float(own.mean()), "fixed": float(fx.mean()),
                                "fixed_minus_own": d1, "fixed_minus_shipped": d2,
                                "gap_closed": closed, "test500": ARM500[x],
                                "degen_fixed": float(np.mean([rows[c]["coc_degenerate"] for c in ids]))}
        say(f"nt_{x:3s}{ARM500[x]:9.4f}{own.mean():9.4f}{fx.mean():10.4f}   "
            f"{d1[0]:+.4f} [{d1[1]:+.4f}, {d1[2]:+.4f}] p={d1[4]:.1e}   "
            f"{d2[0]:+.4f} [{d2[1]:+.4f}, {d2[2]:+.4f}] p={d2[4]:.1e}   {closed:+.2f}")
    say("gap closed: share of (own - shipped) removed by swapping layer 0's 19 kept heads")

    # ---- B: single-head sensitivity of layer 0 under the shipped mask ----
    ids = ids_of("test_cruise60")
    b, s = vec(base, ids), vec(ship, ids)
    z = dict(np.load(REPO / "outputs" / "importance_v2" / "importance.npz"))

    def rn(x):
        return np.argsort(np.argsort(x)) / (len(x) - 1)

    rt, rc = rn(z["traj_vlm_q"][0]), rn(z["coc_vlm_q"][0])
    kept0 = sorted(json.loads((REPO / "outputs" / "slim_dual_u40_v2" / "slim_meta.json")
                              .read_text())["vlm"][0]["q"])
    say(f"\n== B. shipped mask, one layer-0 head dropped at a time ({len(ids)} cruise clips)")
    say(f"unpruned {b.mean():.4f}   shipped {s.mean():.4f}")
    say(f"{'head':5s}{'mean':>8s}   minus shipped [95% CI]              median      p   "
        f"rank I_traj  rank I_CoC   dual")
    rows_b = []
    for h in kept0:
        rows = load_run(f"swap_ship_L0drop{h}_cruise60")
        if not all(c in rows for c in ids):
            say(f"H{h}: missing ({len(rows)}/{len(ids)})")
            continue
        v = vec(rows, ids)
        d = ci(v - s)
        sig = "*" if d[1] > 0 or d[2] < 0 else " "
        metrics["l0_map"][str(h)] = {"mean": float(v.mean()), "minus_shipped": d,
                                     "rank_traj": float(rt[h]), "rank_coc": float(rc[h])}
        rows_b.append((h, d[0], d[1], d[2], rt[h], rc[h]))
        say(f"H{h:<4d}{v.mean():8.4f}   {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]{sig}  "
            f"{d[3]:+.4f}  {d[4]:.1e}   {rt[h]:8.2f}  {rc[h]:10.2f}  {max(rt[h], rc[h]):5.2f}")
    if len(rows_b) > 3:
        a = np.array(rows_b)
        for lab, col in (("rank I_traj", 4), ("rank I_CoC", 5)):
            r, p = spearmanr(a[:, col], a[:, 1])
            metrics[f"rho_damage_{lab}"] = [float(r), float(p)]
            say(f"Spearman(damage, {lab}) over {len(a)} heads: {r:+.3f} p={p:.3f}")
        r, p = spearmanr(np.maximum(a[:, 4], a[:, 5]), a[:, 1])
        metrics["rho_damage_dual"] = [float(r), float(p)]
        say(f"Spearman(damage, dual score): {r:+.3f} p={p:.3f}")
        n_bad = int((a[:, 2] > 0).sum())
        say(f"heads whose removal hurts (CI above 0): {n_bad} of {len(a)}: "
            f"{[int(h) for h in a[a[:, 2] > 0, 0]]}")

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
        order = np.argsort(-a[:, 1])
        ax = axes[0]
        ax.bar(range(len(a)), a[order, 1], color=[C3 if a[i, 2] > 0 else C1 for i in order], width=0.7)
        ax.errorbar(range(len(a)), a[order, 1],
                    yerr=[a[order, 1] - a[order, 2], a[order, 3] - a[order, 1]],
                    fmt="none", ecolor=INK, capsize=2, lw=1)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(range(len(a)), [f"H{int(h)}" for h in a[order, 0]], rotation=60, fontsize=8)
        ax.set_ylabel("minADE@6 minus shipped [m], cruise")
        ax.set_title("Dropping one kept layer-0 head from the shipped mask")
        ax = axes[1]
        sc = ax.scatter(a[:, 4], a[:, 5], c=a[:, 1], cmap="RdBu_r", s=70,
                        vmin=-np.abs(a[:, 1]).max(), vmax=np.abs(a[:, 1]).max(), edgecolor=INK)
        for h, dm, _, _, x, y in a:
            ax.annotate(f"{int(h)}", (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax.set_xlabel("rank I_traj in layer 0")
        ax.set_ylabel("rank I_CoC in layer 0")
        ax.set_title("What the criterion believed vs what dropping costs")
        fig.colorbar(sc, ax=ax, label="damage [m]")
        fig.tight_layout()
        fig.savefig(out / "plots" / "l0_map.png", dpi=150)
        plt.close(fig)

    (out / "metrics_l0.json").write_text(json.dumps(metrics, indent=2))
    (out / "summary_l0.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
