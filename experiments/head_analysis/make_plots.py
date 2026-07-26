"""Generate analysis plots from head_stats.npz.

Usage: python make_plots.py --exp-id head_analysis_v1
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/workspace/alpamayo-model-compression")

BG = "#FAF9F5"
INK = "#29261B"
MUTED = "#6B6555"
# categorical slots (validated reference palette, light mode)
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": INK,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def heatmap(ax, data, title, cmap="Oranges", vmin=None, vmax=None, xlabel="head", cbar=True):
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("layer")
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", type=str, default="head_analysis_v1")
    args = ap.parse_args()
    out_dir = REPO / "outputs" / args.exp_id
    plots = out_dir / "plots"
    plots.mkdir(exist_ok=True)
    d = np.load(out_dir / "head_stats.npz")

    # 1) VLM Q-head Taylor importance (CoC NLL)
    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap(ax, d["taylor_vlm_q"], "VLM Q-head Taylor importance (CoC NLL)")
    fig.tight_layout(); fig.savefig(plots / "taylor_vlm_q.png", dpi=150); plt.close(fig)

    # 2) KV-head importance: CoC part, FM part, combined
    coc = d["taylor_vlm_kv_coc"]
    fm = d["taylor_cache_k_fm"] + d["taylor_cache_v_fm"]
    comb = coc / coc.sum() + fm / fm.sum()
    fig, axes = plt.subplots(1, 3, figsize=(11, 6))
    heatmap(axes[0], coc, "KV-head: CoC NLL (v-gate)", xlabel="kv head")
    heatmap(axes[1], fm, "KV-head: FM cache Taylor", xlabel="kv head")
    heatmap(axes[2], comb, "KV-head: combined (norm.)", xlabel="kv head")
    fig.tight_layout(); fig.savefig(plots / "taylor_kv.png", dpi=150); plt.close(fig)

    # 3) Expert Q-head Taylor
    fig, ax = plt.subplots(figsize=(6, 6))
    heatmap(ax, d["taylor_expert_q"], "Expert Q-head Taylor importance (FM MSE)")
    fig.tight_layout(); fig.savefig(plots / "taylor_expert_q.png", dpi=150); plt.close(fig)

    # 4) VLM modality attention mass (CoC queries)
    fig, axes = plt.subplots(1, 4, figsize=(14, 6))
    for ax, key, title in zip(
        axes,
        ["vlm_mass_vision_coc", "vlm_mass_text_coc", "vlm_mass_sink_coc", "vlm_mass_hist_coc"],
        ["vision", "text/CoC", "sink (pos 0)", "traj history"],
    ):
        heatmap(ax, d[key], f"mass → {title}", cmap="Blues", vmin=0, vmax=1, cbar=(title == "traj history"))
    fig.suptitle("VLM attention mass by key region (CoC queries)", y=1.0)
    fig.tight_layout(); fig.savefig(plots / "vlm_modality_mass.png", dpi=150); plt.close(fig)

    # 5) Expert cache usage: per-layer means + FM cache Taylor per layer
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    layers = np.arange(d["expert_mass_own"].shape[0])
    series = [
        ("vision", d["expert_mass_vision"].mean(1), C1),
        ("text/CoC", d["expert_mass_text"].mean(1), C2),
        ("own 64 tokens", d["expert_mass_own"].mean(1), C3),
        ("traj history", d["expert_mass_hist"].mean(1), C4),
    ]
    for name, y, c in series:
        axes[0].plot(layers, y, color=c, lw=2, label=name)
    axes[0].set_xlabel("expert layer"); axes[0].set_ylabel("mean attention mass")
    axes[0].set_title("Expert attention mass by region")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(axis="y", color="#E8E6DC", lw=0.7)
    per_layer_fm = fm.sum(1)
    axes[1].bar(layers, per_layer_fm, color=C1, width=0.7)
    axes[1].set_xlabel("VLM layer (cache)"); axes[1].set_ylabel("|k·∇k| + |v·∇v|")
    axes[1].set_title("FM-loss Taylor on VLM KV cache, per layer")
    axes[1].grid(axis="y", color="#E8E6DC", lw=0.7)
    fig.tight_layout(); fig.savefig(plots / "expert_cache_usage.png", dpi=150); plt.close(fig)

    # 6) sink mass + normalized entropy (all queries)
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    heatmap(axes[0], d["vlm_mass_sink_all"], "VLM sink mass (all queries)", cmap="Purples", vmin=0, vmax=1)
    heatmap(axes[1], d["vlm_entropy_all"], "VLM normalized entropy (all queries)", cmap="Greens", vmin=0, vmax=1)
    fig.tight_layout(); fig.savefig(plots / "vlm_sink_entropy.png", dpi=150); plt.close(fig)

    # 7) head output norms
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    heatmap(axes[0], d["vlm_headnorm"], "VLM head output norm")
    heatmap(axes[1], d["expert_headnorm"], "Expert head output norm")
    fig.tight_layout(); fig.savefig(plots / "headnorm.png", dpi=150); plt.close(fig)

    # 8) redundancy: per-layer mean off-diagonal CKA and mean nearest-neighbor JS
    cka = d["vlm_cka"]  # (L, H, H)
    js = d["vlm_js"]
    L, H, _ = cka.shape
    off = ~np.eye(H, dtype=bool)
    mean_cka = np.array([cka[i][off].mean() for i in range(L)])
    js_nn = np.array([
        np.where(off, js[i], np.inf).min(axis=1).mean() for i in range(L)
    ])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(L), mean_cka, color=C1, lw=2, label="mean off-diag CKA (head outputs)")
    ax.plot(np.arange(L), js_nn, color=C2, lw=2, label="mean nearest-head JS (attention)")
    ax.set_xlabel("VLM layer"); ax.set_ylabel("similarity / divergence")
    ax.set_title("Within-layer head redundancy")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", color="#E8E6DC", lw=0.7)
    fig.tight_layout(); fig.savefig(plots / "redundancy.png", dpi=150); plt.close(fig)

    # 9) Pareto: cumulative share of total Taylor importance
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, scores, c in [
        ("VLM Q heads (1152)", d["taylor_vlm_q"].flatten(), C1),
        ("VLM KV heads (288, comb.)", comb.flatten(), C2),
        ("Expert Q heads (576)", d["taylor_expert_q"].flatten(), C3),
    ]:
        s = np.sort(scores)[::-1]
        cum = np.cumsum(s) / s.sum()
        x = np.arange(1, len(s) + 1) / len(s) * 100
        ax.plot(x, cum * 100, color=c, lw=2, label=name)
    ax.axhline(90, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("top heads kept (%)"); ax.set_ylabel("cumulative importance (%)")
    ax.set_title("Concentration of Taylor importance")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="y", color="#E8E6DC", lw=0.7)
    fig.tight_layout(); fig.savefig(plots / "pareto.png", dpi=150); plt.close(fig)

    # numeric summaries for the report
    np.set_printoptions(precision=3, suppress=True)
    q = d["taylor_vlm_q"]
    print("clips:", int(d["n_clips"]))
    print("Q-head importance: top-5 layers by mean:", np.argsort(q.mean(1))[::-1][:5])
    print("Q-head importance: bottom-5 layers by mean:", np.argsort(q.mean(1))[:5])
    print("KV combined: per-layer mean:", comb.mean(1))
    print("FM cache per-layer share (top 8):")
    share = per_layer_fm / per_layer_fm.sum()
    for i in np.argsort(share)[::-1][:8]:
        print(f"  layer {i}: {share[i] * 100:.1f}%")
    for frac in (0.5, 0.75, 0.9):
        s = np.sort(q.flatten())[::-1]
        cum = np.cumsum(s) / s.sum()
        n = int(np.searchsorted(cum, frac)) + 1
        print(f"VLM Q heads for {frac * 100:.0f}% importance: {n}/{len(s)} ({n / len(s) * 100:.0f}%)")
    print("plots ->", plots)


if __name__ == "__main__":
    main()
