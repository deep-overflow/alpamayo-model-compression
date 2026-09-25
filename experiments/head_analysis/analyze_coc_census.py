"""Do the heads that I_traj and I_CoC rank differently at CoC positions attend to different keys?

plans/2026-09-25_coc-position-disagreement.md, test 2. Joins the CoC-query attention census
(run_coc_census.py shards) with the same-position scores of the calib_100 anatomy
(outputs/gradanat_v1/anatomy_perclip_q.npz): per layer, T-fav = the 6 heads with the largest
rank(I_traj@CoC) - rank(I_CoC@CoC), C-fav = the 6 smallest.

  G-A  T-fav heads put more of their CoC-query attention on vision + ego-history keys than
       C-fav heads (one-sided Mann-Whitney over heads, per band, p < 0.01)
  G-B  C-fav heads put more on prompt + earlier-CoC keys than T-fav heads (p < 0.01)
  continuous: within-layer Spearman of each head's key-group masses with I_traj@CoC and
  I_CoC@CoC, band means

Usage:
  python experiments/head_analysis/analyze_coc_census.py --shards coc_census_v1_s0 coc_census_v1_s1 \
      --out coc_census_v1
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

REPO = Path(__file__).resolve().parents[2]
GROUPS = ("sink", "vision", "hist", "prompt", "coc_prev", "self")
VIS, HIST, PT, COC, SINK = range(5)  # anatomy type rows
BANDS = {"0-21": range(22), "22-34": range(22, 35)}
K = 6
rng = np.random.default_rng(0)


def load_census(shards):
    clips, arrs = [], []
    for s in shards:
        m = json.loads((REPO / "outputs" / s / "metrics.json").read_text())
        z = np.load(REPO / "outputs" / s / "census.npz")
        clips += m["clip_ids"]
        arrs.append(z["mass_by_clip"])
    return clips, np.concatenate(arrs)  # (N, L, H, 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--anatomy", default="gradanat_v1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = REPO / "outputs" / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)
    clips, mass = load_census(args.shards)
    pq = np.load(REPO / "outputs" / args.anatomy / "anatomy_perclip_q.npz")
    met = json.loads((REPO / "outputs" / args.anatomy / "metrics.json").read_text())
    anat_ids = [r["clip_id"] for r in met["per_clip"]]
    fm, ce = pq["fm_full"].astype(np.float64), pq["ce"].astype(np.float64)  # (N, 5, 36, 32)
    Tc, Cc = np.abs(fm[:, COC]).mean(0), np.abs(ce[:, COC]).mean(0)  # (36, 32)
    common = sorted(set(clips) & set(anat_ids))
    lines = [f"CoC-query attention census vs same-position scores -- {len(clips)} census clips, {len(common)} shared with {args.anatomy}", ""]
    M = mass.mean(0)  # (L, H, 6) mean over clips
    lay = M.mean(1)
    lines.append("attention of CoC queries by key group, layer means (0-6 / 7-21 / 22-34): " + " | ".join(
        f"{g} {lay[0:7, i].mean():.3f} / {lay[7:22, i].mean():.3f} / {lay[22:35, i].mean():.3f}" for i, g in enumerate(GROUPS)))

    def rk(x):
        return np.argsort(np.argsort(x, axis=1), axis=1) / (x.shape[1] - 1)

    diff = rk(Tc) - rk(Cc)
    gates, res = {}, {}
    for bname, layers in BANDS.items():
        tf, cf = [], []
        for l in layers:
            order = np.argsort(diff[l])
            cf += [(l, h) for h in order[:K]]
            tf += [(l, h) for h in order[-K:]]
        g = {}
        for name, hs in (("T-fav", tf), ("C-fav", cf)):
            L_, H_ = np.array([l for l, _ in hs]), np.array([h for _, h in hs])
            g[name] = M[L_, H_]  # (n, 6)
        vh = {n: g[n][:, 1] + g[n][:, 2] for n in g}  # vision + hist
        pc = {n: g[n][:, 3] + g[n][:, 4] for n in g}  # prompt + earlier CoC
        pA = float(mannwhitneyu(vh["T-fav"], vh["C-fav"], alternative="greater")[1])
        pB = float(mannwhitneyu(pc["C-fav"], pc["T-fav"], alternative="greater")[1])
        # clip bootstrap of the group-mean difference
        Lt, Ht = np.array([l for l, _ in tf]), np.array([h for _, h in tf])
        Lc, Hc = np.array([l for l, _ in cf]), np.array([h for _, h in cf])
        d_vh, d_pc = [], []
        for _ in range(2000):
            idx = rng.integers(0, len(mass), len(mass))
            mb = mass[idx].mean(0)
            d_vh.append((mb[Lt, Ht][:, 1:3].sum(1)).mean() - (mb[Lc, Hc][:, 1:3].sum(1)).mean())
            d_pc.append((mb[Lc, Hc][:, 3:5].sum(1)).mean() - (mb[Lt, Ht][:, 3:5].sum(1)).mean())
        gates[bname] = {"G-A": pA < 0.01, "G-A_p": pA, "G-B": pB < 0.01, "G-B_p": pB,
                        "vision+hist T-fav": float(vh["T-fav"].mean()), "vision+hist C-fav": float(vh["C-fav"].mean()),
                        "prompt+coc T-fav": float(pc["T-fav"].mean()), "prompt+coc C-fav": float(pc["C-fav"].mean()),
                        "d_vh_ci": [float(np.percentile(d_vh, 2.5)), float(np.percentile(d_vh, 97.5))],
                        "d_pc_ci": [float(np.percentile(d_pc, 2.5)), float(np.percentile(d_pc, 97.5))]}
        lines.append(f"\n[{bname}] mean attention of CoC queries by key group (T-fav = I_traj-favoured at CoC, C-fav = I_CoC-favoured; {len(tf)} heads each)")
        lines.append(f"{'group':7s} " + " ".join(f"{gname:>9s}" for gname in GROUPS))
        for name in ("T-fav", "C-fav"):
            lines.append(f"{name:7s} " + " ".join(f"{v:9.3f}" for v in g[name].mean(0)))
        lines.append("  all heads " + " ".join(f"{v:9.3f}" for v in M[list(layers)].mean((0, 1))))
        lines.append(f"  G-A vision+hist: T-fav {vh['T-fav'].mean():.3f} vs C-fav {vh['C-fav'].mean():.3f}, diff CI [{gates[bname]['d_vh_ci'][0]:+.3f}, {gates[bname]['d_vh_ci'][1]:+.3f}], MWU p={pA:.2g} -> {'PASS' if pA < 0.01 else 'FAIL'}")
        lines.append(f"  G-B prompt+coc_prev: C-fav {pc['C-fav'].mean():.3f} vs T-fav {pc['T-fav'].mean():.3f}, diff CI [{gates[bname]['d_pc_ci'][0]:+.3f}, {gates[bname]['d_pc_ci'][1]:+.3f}], MWU p={pB:.2g} -> {'PASS' if pB < 0.01 else 'FAIL'}")
        # continuous, within layer
        rows = []
        for i, gname in enumerate(GROUPS):
            rt = np.nanmean([spearmanr(M[l, :, i], Tc[l])[0] for l in layers])
            rc = np.nanmean([spearmanr(M[l, :, i], Cc[l])[0] for l in layers])
            rd = np.nanmean([spearmanr(M[l, :, i], diff[l])[0] for l in layers])
            rows.append((gname, rt, rc, rd))
            res[f"{bname}:{gname}"] = {"rho_traj": float(rt), "rho_coc": float(rc), "rho_diff": float(rd)}
        lines.append("  within-layer Spearman of a head's key-group mass with I_traj@CoC / I_CoC@CoC / their rank difference:")
        lines.append("    " + " | ".join(f"{gname} {rt:+.2f} / {rc:+.2f} / {rd:+.2f}" for gname, rt, rc, rd in rows))
    # figure
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    for name, col in (("T-fav", "#2a78d6"), ("C-fav", "#e87ba4")):
        prof = []
        for l in range(35):
            order = np.argsort(diff[l])
            hs = order[-K:] if name == "T-fav" else order[:K]
            prof.append(M[l, hs][:, 1:3].sum(1).mean())
        ax.plot(range(35), prof, color=col, lw=1.4, label=f"{name} heads")
    ax.plot(range(35), M[:35, :, 1:3].sum(2).mean(1), color="grey", lw=1.0, ls="--", label="all heads")
    ax.axvspan(21.5, 35.5, color="#777777", alpha=0.13, lw=0)
    ax.set_xlabel("VLM layer"); ax.set_ylabel("CoC-query attention on vision + ego-history keys"); ax.legend(fontsize=7)
    ax = axes[1]
    for bname, layers in BANDS.items():
        for name, col, mk in (("T-fav", "#2a78d6", "o"), ("C-fav", "#e87ba4", "s")):
            xs, ys = [], []
            for l in layers:
                order = np.argsort(diff[l])
                hs = order[-K:] if name == "T-fav" else order[:K]
                xs += list(M[l, hs][:, 1:3].sum(1)); ys += list(M[l, hs][:, 3:5].sum(1))
            ax.scatter(xs, ys, s=10, color=col, marker=mk, alpha=0.5 if bname == "0-21" else 0.9, label=f"{name} {bname}")
    ax.set_xlabel("vision + ego-history mass"); ax.set_ylabel("prompt + earlier-CoC mass"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "plots" / "coc_census_groups.png", dpi=150); plt.close(fig)
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "metrics.json").write_text(json.dumps({"gates": gates, "continuous": res, "n_clips": len(clips)}, indent=1))
    (out / "config.json").write_text(json.dumps({"shards": args.shards, "anatomy": args.anatomy, "plan": "plans/2026-09-25_coc-position-disagreement.md"}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
