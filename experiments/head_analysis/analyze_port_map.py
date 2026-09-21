"""Merge the port-map shards and evaluate PM1-PM3 (plans/2026-09-20_gradient-anatomy.md).

The additive identity does the work here. With S(m, l) = E_c sum_u G_m,u sign(G_full,u) the
signed share of cache layer m in the units of layer l,

    I_traj(l) = sum_u E_c |G_full,u| = sum_m S(m, l),      S(m, l) = 0 for m <= l,

so the shipped depth profile is rebuilt from ports, and the step between layers 21 and 23
splits exactly into "ports a layer-23 unit can no longer reach" and "change through the
ports it still reaches".

  PM1  for units in layers 16-21 the per-cache-layer share peaks inside 22-24, and those
       three layers carry >= 30% of the units' I_traj
  PM2  the closed-port term is >= 50% of the fall I_traj(21) - I_traj(23)  (< 25% refutes)
  PM3  ports m >= 22 are read mostly through non-vision cache entries, ports m <= 15
       mostly through vision entries

Usage:
  python experiments/head_analysis/analyze_port_map.py --shards portmap_v1_s0 portmap_v1_s1 \
      portmap_v1_s2 --out portmap_v1
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
LAST = 35  # I_traj is identically zero at layer 35
PORT_GROUPS = (("cache 1-21", 0, 22), ("cache 22-24", 22, 25), ("cache 25-29", 25, 30),
               ("cache 30-35", 30, 36))

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})


def merge(shards):
    """Clip-weighted mean of every array; the shards hold means over their own clips."""
    acc, n_tot, recs = {}, 0, []
    for s in shards:
        d = REPO / "outputs" / s
        met = json.loads((d / "metrics.json").read_text())
        n = met["n_clips"]
        with np.load(d / "port_map.npz") as z:
            for k in z.files:
                acc[k] = acc.get(k, 0) + z[k] * n
        n_tot += n
        recs += met["per_clip"]
    return {k: v / n_tot for k, v in acc.items()}, n_tot, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", default="portmap_v1")
    args = ap.parse_args()

    z, n, recs = merge(args.shards)
    d = REPO / "outputs" / args.out
    (d / "plots").mkdir(parents=True, exist_ok=True)
    np.savez(d / "port_map.npz", **z)
    out, lines = {"n_clips": n, "shards": args.shards}, [f"port map -- {n} clips from {args.shards}", ""]
    L = np.arange(36)

    err = np.array([r["ports_vs_full_relerr_by_layer"] for r in recs])  # (N, 35)
    out["integrity"] = {"ports_vs_full_relerr_median_layer": float(np.median(err)),
                        "ports_vs_full_relerr_worst_layer_median_clip": float(np.median(err.max(1))),
                        "mlp_signed_over_full_median": float(np.median([r["mlp_signed_vs_full"] for r in recs]))}
    lines += ["integrity (72 partial-seed bf16 backwards against one full-seed backward)",
              (f"  Q heads, sum of ports vs full: rel err median layer {np.median(err):.1e}, "
              f"worst layer of a clip (median) {np.median(err.max(1)):.1e}"),
              (f"  MLP, sum of signed shares / shipped layer sum: median "
              f"{out['integrity']['mlp_signed_over_full_median']:.4f} (1 = exact)"), ""]

    for a in ("mlp", "q"):
        S = z[f"{a}_signed"].sum((1, 2))  # (36 cache layers, 36 unit layers) additive shares
        Sp = z[f"{a}_signed"].sum(2)  # (36, 2, 36) by position group of the cache entries
        full = z[f"{a}_full"]  # (36,) the shipped layer sum
        # a unit in layer l cannot reach cache layer m <= l (layer m's k/v read the stream
        # BEFORE layer m writes to it), so those cells must be exactly zero
        leak = float(np.abs(S[L[:, None] <= L[None, :]]).max())
        res = {"rebuilt_over_shipped_by_layer": (S.sum(0)[:LAST] / full[:LAST]).tolist(),
               "structural_zero_max": leak}

        # PM1: where, over cache layers, do units in 16-21 get their I_traj from?
        prof = S[:, 16:22].sum(1) / full[16:22].sum()  # (36,)
        res["pm1_profile_units_16_21"] = prof.tolist()
        res["pm1_argmax_cache_layer"] = int(prof.argmax())
        res["pm1_share_cache_22_24"] = float(prof[22:25].sum())
        res["pm1_share_by_group"] = {g: float(prof[lo:hi].sum()) for g, lo, hi in PORT_GROUPS}
        res["pm1_per_layer_density"] = {g: float(prof[lo:hi].mean()) for g, lo, hi in PORT_GROUPS}
        res["pm1_pass"] = bool(22 <= prof.argmax() <= 24 and prof[22:25].sum() >= 0.30)

        # PM2: the fall between unit layers 21 and 23, split exactly
        fall = float(S[:, 21].sum() - S[:, 23].sum())
        closed = float(S[22:24, 21].sum())  # cache layers 22, 23: unreachable from layer 23
        still = float((S[24:, 21] - S[24:, 23]).sum())
        res.update({"pm2_fall": fall, "pm2_closed_ports": closed, "pm2_open_ports_change": still,
                    "pm2_closed_share_of_fall": closed / fall,
                    "pm2_shipped_fall": float(full[21] - full[23]),
                    "pm2_I21": float(full[21]), "pm2_I23": float(full[23])})
        res["pm2_verdict"] = ("PASS" if closed / fall >= 0.5 else
                              "REFUTED" if closed / fall < 0.25 else "in between")
        # the same split for every adjacent pair, to see whether 21->23 is special
        res["closed_term_by_layer"] = [float(S[l + 1, l]) for l in range(LAST - 1)]
        res["next_port_share_of_layer"] = [float(S[l + 1, l] / full[l]) for l in range(LAST - 1)]

        # PM3: through which cache entries is each port read?
        tot_m = Sp.sum(2)  # (36, 2) over all unit layers
        vis_share = tot_m[:, 0] / np.where(tot_m.sum(1) != 0, tot_m.sum(1), np.nan)
        res["pm3_vision_share_by_cache_layer"] = vis_share.tolist()
        res["pm3_vision_share_ports_le15"] = float(tot_m[1:16, 0].sum() / tot_m[1:16].sum())
        res["pm3_vision_share_ports_ge22"] = float(tot_m[22:, 0].sum() / tot_m[22:].sum())
        res["pm3_pass"] = bool(res["pm3_vision_share_ports_ge22"] < 0.5 < res["pm3_vision_share_ports_le15"])
        # the trunk (units in layers 6-21): through which door does I_traj reach it?
        trunk = Sp[:, :, 6:22].sum(2)  # (36, 2) cache layer x position group
        res["trunk_share_by_port_group"] = {g: float(trunk[lo:hi].sum() / trunk.sum()) for g, lo, hi in PORT_GROUPS}
        res["trunk_vision_share"] = float(trunk[:, 0].sum() / trunk.sum())
        res["trunk_share_late_nonvision"] = float(trunk[22:, 1].sum() / trunk.sum())
        res["trunk_share_late_vision"] = float(trunk[22:, 0].sum() / trunk.sum())
        res["trunk_share_early_vision"] = float(trunk[:22, 0].sum() / trunk.sum())
        res["trunk_share_early_nonvision"] = float(trunk[:22, 1].sum() / trunk.sum())
        # and by the token type the trunk unit itself acts on (D1), for the late non-vision door
        by_type = z[f"{a}_signed"][22:, 1][:, :, 6:22].sum((0, 2))  # (5,)
        res["trunk_late_nonvision_by_unit_type"] = (by_type / by_type.sum()).tolist()
        res["port_profile_total"] = (tot_m.sum(1) / full[:LAST].sum()).tolist()  # share of ALL I_traj
        res["port_profile_by_group"] = (tot_m / full[:LAST].sum()).T.tolist()
        out[a] = res

        lines += [f"{a.upper()} axis",
                  (f"  rebuilt I_traj / shipped, by unit layer: min {np.min(res['rebuilt_over_shipped_by_layer']):.3f} "
                  f"max {np.max(res['rebuilt_over_shipped_by_layer']):.3f}; cells that must be zero: max {leak:.1e}"),
                  (f"  PM1 units 16-21: share of I_traj by port group "
                  f"{ {g: round(v, 3) for g, v in res['pm1_share_by_group'].items()} }; "
                  f"peak at cache layer {res['pm1_argmax_cache_layer']}; cache 22-24 carries "
                  f"{res['pm1_share_cache_22_24']:.3f} (>=0.30) -> {'PASS' if res['pm1_pass'] else 'FAIL'}"),
                  (f"  PM2 fall I(21)-I(23) = {fall:.4f} ({res['pm2_I21']:.4f} -> {res['pm2_I23']:.4f}): "
                  f"ports 22-23 closing {closed:.4f} ({closed / fall:.1%}), change through ports >=24 "
                  f"{still:.4f} ({still / fall:.1%}) -> {res['pm2_verdict']}"),
                  (f"  trunk units (layers 6-21): I_traj by door -- cache >=22 non-vision entries "
                  f"{res['trunk_share_late_nonvision']:.3f}, cache >=22 vision entries "
                  f"{res['trunk_share_late_vision']:.3f}, cache <=21 vision {res['trunk_share_early_vision']:.3f}, "
                  f"cache <=21 non-vision {res['trunk_share_early_nonvision']:.3f}; of the first door, by the "
                  f"token type the unit acts on (vision, hist, prompt, coc, sink) "
                  f"{np.round(res['trunk_late_nonvision_by_unit_type'], 3).tolist()}"),
                  (f"  PM3 vision share of the signed share: ports <=15 {res['pm3_vision_share_ports_le15']:.3f}, "
                  f"ports >=22 {res['pm3_vision_share_ports_ge22']:.3f} -> {'PASS' if res['pm3_pass'] else 'FAIL'}"), ""]

    # ------------------------------------------------------------------ plots
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, a, title in zip(axes, ("mlp", "q"), ("MLP channels", "Q heads")):
        S = z[f"{a}_signed"].sum((1, 2))  # (36, 36)
        bottom = np.zeros(LAST)
        for (g, lo, hi), col in zip(PORT_GROUPS, (C2, C4, C1, C3)):
            y = S[lo:hi, :LAST].sum(0)
            ax.bar(L[:LAST], y, bottom=bottom, color=col, width=0.9, label=g)
            bottom += y
        ax.plot(L[:LAST], z[f"{a}_full"][:LAST], color=INK, lw=1.2, label="shipped I_traj (layer sum)")
        ax.set_title(f"{title}: I_traj rebuilt from the cache layer it arrives through")
        ax.set_xlabel("layer of the unit")
        ax.grid(axis="y", color="#E8E6DC", lw=0.7)
    axes[0].set_ylabel("sum over units of E|dL/dg|")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(d / "plots" / "itraj_by_port.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    S = z["mlp_signed"].sum((1, 2))
    share = S[:, :LAST] / z["mlp_full"][None, :LAST]
    im = axes[0].imshow(np.where(np.arange(36)[:, None] > L[None, :LAST], share, np.nan),
                        aspect="auto", origin="lower", cmap="viridis", vmin=0)
    axes[0].set_xlabel("layer of the unit (MLP)")
    axes[0].set_ylabel("cache layer the FM gradient enters through")
    axes[0].set_title("Share of a layer's I_traj by port")
    plt.colorbar(im, ax=axes[0], fraction=0.04, pad=0.02)
    pg = np.array(out["mlp"]["port_profile_by_group"])  # (2, 36)
    axes[1].bar(L, pg[0], color=C1, width=0.85, label="through vision cache entries")
    axes[1].bar(L, pg[1], bottom=pg[0], color=C4, width=0.85, label="through all other entries")
    axes[1].set_xlabel("cache layer")
    axes[1].set_ylabel("share of all VLM I_traj (MLP)")
    axes[1].set_title("Port profile: how much importance each cache layer carries")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", color="#E8E6DC", lw=0.7)
    fig.tight_layout()
    fig.savefig(d / "plots" / "port_profile.png", dpi=160)
    plt.close(fig)

    (d / "summary.txt").write_text("\n".join(lines) + "\n")
    (d / "metrics_analysis.json").write_text(json.dumps(out, indent=2))
    (d / "config.json").write_text(json.dumps({"merged_from": args.shards, "n_clips": n}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
