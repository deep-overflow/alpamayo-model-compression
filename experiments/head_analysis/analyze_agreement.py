"""Channel-agreement analysis of the dual-objective importance (post-hoc, no GPU).

Reads outputs/<importance_exp>/importance.npz and asks: how much of each layer must be kept
if BOTH objectives are to keep their top-q units, and how does that compare to the layerwise
budget an existing slim config actually spent there?

The prunable headroom of a layer is 1 - |top_q(I_coc) union top_q(I_traj)| / n. Where the two
objectives agree the union is small (cheap to prune); where they diverge the union is large
(expensive). A budget allocated against that headroom is the "agreement-proportional" config
computed here at matched total compression.

Usage: python analyze_agreement.py [--importance-exp importance_v1]
                                   [--ref-slim slim_integrated_mag] [--exp-id agreement_alloc]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]

BG = "#FAF9F5"
INK = "#29261B"
MUTED = "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

BANDS = [(0, 5), (6, 17), (18, 29), (30, 35)]

# Alpamayo-1.5-10B VLM text tower (Qwen3-8B shape)
HIDDEN, HEADS, HEAD_DIM, INTER = 4096, 32, 128, 12288
P_HEAD = 2 * HIDDEN * HEAD_DIM   # q_proj + o_proj per Q head
P_MLPC = 3 * HIDDEN              # gate/up rows + down cols per MLP channel


def per_layer_agreement(coc, traj, keep):
    """(rho, union) per layer; nan where an objective carries no gradient."""
    n_layers = coc.shape[0]
    rho = np.full(n_layers, np.nan)
    union = np.full(n_layers, np.nan)
    for l in range(n_layers):
        c, t = coc[l], traj[l]
        if np.allclose(c, 0) or np.allclose(t, 0):
            continue
        rho[l] = spearmanr(c, t).statistic
        k = round(keep * len(c))
        union[l] = len(set(np.argsort(-c)[:k]) | set(np.argsort(-t)[:k])) / len(c)
    return rho, union


def fill_tail(u):
    """Layer 35 has no usable gradient on one objective -> use the late-band mean."""
    out = u.copy()
    out[np.isnan(out)] = np.nanmean(out[30:35])
    return out


def removed_fraction(alpha, u_q, u_mlp, n_layers):
    kq = np.clip(alpha * u_q, 0, 1)
    km = np.clip(alpha * u_mlp, 0, 1)
    full = n_layers * (HEADS * P_HEAD + INTER * P_MLPC)
    return 1 - (kq * HEADS * P_HEAD + km * INTER * P_MLPC).sum() / full, kq, km


def match_budget(target, u_q, u_mlp, n_layers):
    """Solve alpha so the agreement-proportional keep profile removes `target` of q/o+MLP params."""
    lo, hi = 0.05, 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        r, _, _ = removed_fraction(mid, u_q, u_mlp, n_layers)
        if r > target:
            lo = mid
        else:
            hi = mid
    alpha = 0.5 * (lo + hi)
    r, kq, km = removed_fraction(alpha, u_q, u_mlp, n_layers)
    return alpha, r, kq, km


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--importance-exp", default="importance_v1")
    ap.add_argument("--ref-slim", default="slim_integrated_mag")
    ap.add_argument("--exp-id", default="agreement_alloc")
    ap.add_argument("--keep", type=float, default=0.5, help="top-q of each objective treated as must-keep")
    args = ap.parse_args()

    out = REPO / "outputs" / args.exp_id
    (out / "plots").mkdir(parents=True, exist_ok=True)
    z = np.load(REPO / "outputs" / args.importance_exp / "importance.npz")
    ref = json.loads((REPO / "outputs" / args.ref_slim / "slim_meta.json").read_text())
    n_layers = z["coc_vlm_q"].shape[0]

    units = {
        "vlm_q": ("VLM Q head", z["coc_vlm_q"], z["traj_vlm_q"]),
        "vlm_mlp": ("VLM MLP channel", z["coc_vlm_mlp"], z["traj_vlm_mlp"]),
        "kv_k": ("KV group (K)", z["coc_kv_k"], z["traj_kv_k"]),
        "exp_q": ("Expert Q head", z["coc_exp_q"], z["traj_exp_q"]),
        "exp_mlp": ("Expert MLP channel", z["coc_exp_mlp"], z["traj_exp_mlp"]),
    }
    metrics = {"keep_q": args.keep, "importance_exp": args.importance_exp,
               "ref_slim": args.ref_slim, "units": {}}
    for key, (label, coc, traj) in units.items():
        rho, union = per_layer_agreement(coc, traj, args.keep)
        metrics["units"][key] = {
            "label": label,
            "scored_layers": int(np.isfinite(rho).sum()),
            "rho": rho.tolist(),
            "union": union.tolist(),
            "bands": {f"{a}-{b}": {"rho": float(np.nanmean(rho[a:b + 1])),
                                   "union": float(np.nanmean(union[a:b + 1]))}
                      for a, b in BANDS},
        }

    u_q = fill_tail(per_layer_agreement(z["coc_vlm_q"], z["traj_vlm_q"], args.keep)[1])
    u_m = fill_tail(per_layer_agreement(z["coc_vlm_mlp"], z["traj_vlm_mlp"], args.keep)[1])

    kq_ref = np.array([len(l["q"]) / HEADS for l in ref["vlm"]])
    km_ref = np.array([len(l["mlp"]) / INTER for l in ref["vlm"]])
    full = n_layers * (HEADS * P_HEAD + INTER * P_MLPC)
    ref_removed = 1 - (kq_ref * HEADS * P_HEAD + km_ref * INTER * P_MLPC).sum() / full
    alpha, got, kq, km = match_budget(ref_removed, u_q, u_m, n_layers)

    metrics["allocation"] = {
        "ref_removed_vlm_qo_mlp": float(ref_removed),
        "alpha": float(alpha), "matched_removed": float(got),
        "ref_keep_q": kq_ref.tolist(), "ref_keep_mlp": km_ref.tolist(),
        "alloc_keep_q": kq.tolist(), "alloc_keep_mlp": km.tolist(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    lines = [
        "Channel-agreement analysis of dual-objective importance",
        f"  importance: {args.importance_exp}   reference config: {args.ref_slim}   must-keep top-{args.keep:.0%}",
        "",
        "Within-layer agreement between the two objectives, and the resulting prunable headroom",
        f"  {'band':8s} " + "".join(f"{u[0]:>26s}" for u in units.values()),
    ]
    for a, b in BANDS:
        row = f"  {a:2d}-{b:2d}    "
        for key in units:
            bd = metrics["units"][key]["bands"][f"{a}-{b}"]
            row += (f"{'rho  n/a':>26s}" if not np.isfinite(bd["rho"])
                    else f"{'rho ' + format(bd['rho'], '+.3f') + '  free ' + format(1 - bd['union'], '.3f'):>26s}")
        lines.append(row)
    lines += [
        "",
        "  (expert columns are n/a by construction: CoC tokens come from the VLM lm_head, so the",
        "   CoC loss carries no gradient into the expert -- the expert is a single-objective module)",
        "",
        f"Budget allocation at matched compression ({ref_removed:.1%} of VLM q/o+MLP params removed)",
        f"  agreement-proportional keep = clip({alpha:.3f} x union_l, 0, 1)  -> removed {got:.1%}",
        "",
        (f"  {'band':8s}{'ref keep Q':>12s}{'alloc keep Q':>14s}{'delta':>9s}"
         f"{'ref keep MLP':>14s}{'alloc keep MLP':>16s}{'delta':>9s}"),
    ]
    for a, b in BANDS:
        s = slice(a, b + 1)
        lines.append(f"  {a:2d}-{b:2d}    {kq_ref[s].mean():>12.3f}{kq[s].mean():>14.3f}"
                     f"{kq[s].mean() - kq_ref[s].mean():>+9.3f}"
                     f"{km_ref[s].mean():>14.3f}{km[s].mean():>16.3f}"
                     f"{km[s].mean() - km_ref[s].mean():>+9.3f}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- figure: agreement vs. what was actually spent
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    layers = np.arange(n_layers)

    ax = axes[0]
    for key, c in [("vlm_q", C1), ("vlm_mlp", C2), ("kv_k", C4)]:
        ax.plot(layers, metrics["units"][key]["rho"], color=c, lw=1.6,
                label=metrics["units"][key]["label"])
    ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("layer"); ax.set_ylabel("within-layer Spearman $\\rho$")
    ax.set_title("CoC vs trajectory importance agreement")
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1]
    ax.plot(layers, 1 - u_q, color=C1, lw=1.6, label="prunable headroom (Q head)")
    ax.plot(layers, 1 - u_m, color=C2, lw=1.6, label="prunable headroom (MLP ch)")
    ax.plot(layers, 1 - kq_ref, color=C1, lw=1.6, ls="--", label="integrated_mag pruned (Q)")
    ax.plot(layers, 1 - km_ref, color=C2, lw=1.6, ls="--", label="integrated_mag pruned (MLP)")
    ax.set_xlabel("layer"); ax.set_ylabel("fraction of units")
    ax.set_title("headroom vs what was actually pruned")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    w = 0.38
    bx = np.arange(len(BANDS))
    ref_b = [1 - kq_ref[a:b + 1].mean() for a, b in BANDS]
    alc_b = [1 - kq[a:b + 1].mean() for a, b in BANDS]
    ax.bar(bx - w / 2, ref_b, w, color=C3, label="integrated_mag")
    ax.bar(bx + w / 2, alc_b, w, color=C1, label="agreement-proportional")
    ax.set_xticks(bx); ax.set_xticklabels([f"{a}-{b}" for a, b in BANDS])
    ax.set_xlabel("layer band"); ax.set_ylabel("Q heads pruned (fraction)")
    ax.set_title(f"same total budget ({ref_removed:.1%}), inverted profile")
    ax.legend(frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out / "plots" / "agreement_allocation.png", dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
