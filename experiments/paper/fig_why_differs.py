"""Paper figures for WHY the two objectives weight the VLM differently (follow-up to Figure 1).

Figure 1 shows that I_traj and I_CoC peak at different depths and stop agreeing on unit
ranks after layer ~22. These panels ask why, using only stored runs (CPU, no model):

  fig3_ratio_step
      I_traj / I_CoC per layer, raw layer sums. The two depth profiles are ONE shared
      profile times a single step: the ratio is flat over layers 0-21, flat again over
      23-34, and drops ~4.5x in between. The different peaks of Figure 1 follow from
      that step and the per-curve max normalisation, so the question reduces to what
      happens at layers 22-23.
  fig3_ratio_step_draws  (supplementary)
      The same ratio for independent calibration draws, OOD clips, a self-anchored
      trajectory target and single action dimensions, each divided by its own early
      plateau. The change point is layer 23 in every run.
  fig3_noise_ceiling_q_head / fig3_noise_ceiling_mlp
      Rank agreement across DISJOINT halves of the calibration clips: the same objective
      on the other half (the ceiling any agreement can reach) against the other objective
      on the other half. For Q heads the ceiling is flat in depth, so the late collapse is
      a real disagreement; for MLP channels the ceiling itself falls, so part of the raw
      collapse in fig1_rank_agreement is attenuation.
  fig3_partition_q_head / fig3_partition_mlp
      Clip x objective partition of single-clip rank variance. With r[o,c,u] = mu_u +
      alpha_{c,u} + beta_{o,u} + eps:  rho(other clip, other objective) = V_mu,
      rho(same clip, other objective) = V_mu + V_alpha, rho(other clip, same objective) =
      V_mu + V_beta. Aggregate agreement is then ~ V_mu / (V_mu + V_beta), which reproduces
      the noise-corrected agreement measured independently above.
  fig3_boundary
      Three independent signals on one depth axis, each min-max normalised: the importance
      ratio (both losses' gradients), the FM loss's sensitivity to the cache tensors it
      reads (expert side), and the excess kurtosis of the Jacobian-lens readout (no loss
      at all). All three change regime over layers 21-24.
  fig3_text_write_q_head / fig3_text_write_mlp
      Rank correlation between each objective's importance and the unit's write norm
      measured at TEXT positions only (J-lens q_w / mlp_w exclude vision, history and
      sink). Late-layer CoC importance is text-position activity; trajectory importance
      is not.
  fig3_pairs_q_head / fig3_pairs_mlp
      Agreement for objective pairs that share their input, their output family, both or
      neither. Late layers follow the output, not the input and not the read-out head:
      CoC on another dataset reproduces CoC's ranking, while VQA-NLL and CoC-NLL -- both
      read through the LM head -- disagree as much as trajectory and CoC do.

Sources: outputs/importance_v2 (+ importance_perclip.npz), outputs/importance_vqa,
outputs/importance_dim0 / _dim1, outputs/jlens_v2, and the independent draws listed in
DRAWS. Numbers quoted in paper/2026-09-20_why-importance-differs.md are written to
figures/fig3_why_differs_stats.json.

Usage:
  .venv/bin/python experiments/paper/fig_why_differs.py [--outputs PATH] [--out-dir PATH]
"""

import argparse
import json
from pathlib import Path

import fig_criterion as fc
import matplotlib.pyplot as plt
import numpy as np
from fig_criterion import COC, GREY, SINGLE, TRAJ, rho_per_layer, save
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
GREEN, PURPLE = "#009E73", "#CC79A7"
N_L = 36
LAST = 35  # I_traj is identically zero at layer 35, so every ratio / rho stops at 34
DRAWS = ["importance_rd100_a", "importance_rd100_b", "importance_rd100_c", "importance_se100_a",
         "importance_su100_a", "importance_val100", "importance_ood", "importance_nt500",
         "importance_st4000", "importance_selftraj_v1", "importance_dim0", "importance_dim1"]


def step_fit(y):
    """Two-level piecewise-constant least-squares fit. y: (n,) -> tau, level0, level1, R2."""
    best = None
    for tau in range(3, len(y) - 3):
        a, b = y[:tau].mean(), y[tau:].mean()
        sse = ((y[:tau] - a) ** 2).sum() + ((y[tau:] - b) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, tau, a, b)
    sse, tau, a, b = best
    return tau, a, b, 1 - sse / ((y - y.mean()) ** 2).sum()


def log_ratio(z, ax):
    t, c = z[f"traj_vlm_{ax}"].sum(1)[:LAST], z[f"coc_vlm_{ax}"].sum(1)[:LAST]  # (35,)
    return np.log(t / c)


def depth_axes(ylabel, ylim=None, shade=True):
    fig, ax = plt.subplots(figsize=SINGLE)
    if shade:
        ax.axvspan(21.5, 35.5, color=GREY, alpha=0.13, lw=0)
    ax.set_xlabel("VLM layer")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.5, 35.5)
    if ylim is not None:
        ax.set_ylim(*ylim)
    return fig, ax


def ceiling_share(bands):
    """Share of the multiplicative early->late fall in cross-objective agreement that is the
    ceiling (geometric mean of the two self-agreements) falling. bands: name -> [early, late]."""
    ceil = [np.sqrt(bands["traj_self"][i] * bands["coc_self"][i]) for i in (0, 1)]
    return float(np.log(ceil[1] / ceil[0]) / np.log(bands["cross"][1] / bands["cross"][0]))


def legend_above(ax, ncol):
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, fontsize=6,
              handlelength=1.8, columnspacing=1.0, borderaxespad=0.2)


def ratio_panels(z, pc, outputs, out):
    print("ratio step")
    L = np.arange(LAST)
    fig, ax = depth_axes(r"$I_{\mathrm{traj}}\,/\,I_{\mathrm{CoC}}$")
    rng = np.random.default_rng(0)
    for name, col, label in (("q", TRAJ, "Q head"), ("mlp", COC, "MLP channel")):
        y = log_ratio(z, name)
        tau, a, b, r2 = step_fit(y)
        ax.plot(L, np.exp(y), color=col, lw=1.4, label=label)
        ax.hlines([np.exp(a), np.exp(b)], [0, tau], [tau - 1, LAST - 1], color=col, lw=0.8, ls=":")
        T = pc[f"traj_vlm_{name}"].sum(2)[:, :LAST]  # (100, 35) per-clip layer sums
        C = pc[f"coc_vlm_{name}"].sum(2)[:, :LAST]  # (100, 35)
        boot = np.array([step_fit(np.log(T[i].mean(0) / C[i].mean(0)))[:3]
                         for i in rng.integers(0, len(T), (2000, len(T)))])  # (2000, 3)
        steps = np.exp(boot[:, 1] - boot[:, 2])
        per_clip = np.array([step_fit(np.log(T[i] / C[i]))[:3] for i in range(len(T))])
        early = np.exp(y[:22])
        t, c = z[f"traj_vlm_{name}"].sum(1)[:LAST], z[f"coc_vlm_{name}"].sum(1)[:LAST]  # (35,)
        fit = np.where(L < tau, np.exp(a), np.exp(b))  # (35,) the two-level ratio alone
        out[f"peaks_{name}"] = {
            # does the step by itself move Figure 1's peak? rebuild each profile from the other one
            "traj": int(t.argmax()), "coc": int(c.argmax()),
            "traj_from_coc_x_step": int((c * fit).argmax()), "coc_from_traj_over_step": int((t / fit).argmax()),
            "traj_fall_L21_to_L23": float(t[21] / t[23]), "coc_rise_L23_to_L27": float(c[27] / c[23]),
        }
        out[f"ratio_{name}"] = {
            "tau": int(tau), "early": float(np.exp(a)), "late": float(np.exp(b)),
            "step": float(np.exp(a - b)), "r2": float(r2),
            "boot_tau_share": float(np.mean(boot[:, 0] == tau)),
            "boot_step_ci": [float(np.percentile(steps, 2.5)), float(np.percentile(steps, 97.5))],
            "clips_with_step": float(np.mean(per_clip[:, 1] > per_clip[:, 2])),
            "early_plateau_cv": float(early.std() / early.mean()),
            "traj_range_over_early_plateau": float(
                z[f"traj_vlm_{name}"].sum(1)[:22].max() / z[f"traj_vlm_{name}"].sum(1)[:22].min()),
        }
        print(f"    {label:11s} tau {tau}  {np.exp(a):.1f}x -> {np.exp(b):.1f}x  "
              f"step {np.exp(a - b):.2f}x {out[f'ratio_{name}']['boot_step_ci']}  R2 {r2:.2f}")
    ax.set_yscale("log")
    ax.set_yticks([2, 5, 10, 20])
    ax.set_yticklabels(["2", "5", "10", "20"])
    ax.set_ylim(1.2, 30)
    ax.legend(loc="lower left")
    save(fig, "fig3_ratio_step")

    fig, ax = depth_axes("Ratio / early plateau (MLP)")
    taus = {}
    for run in DRAWS:
        p = outputs / run / "importance.npz"
        if not p.exists():
            continue
        with np.load(p) as d:
            y = log_ratio(d, "mlp")
            taus[run] = [int(step_fit(y)[0]), int(step_fit(log_ratio(d, "q"))[0])]
        ax.plot(L, np.exp(y - y[:22].mean()), color=GREY, lw=0.7, alpha=0.8)
    y = log_ratio(z, "mlp")
    ax.plot(L, np.exp(y - y[:22].mean()), color=TRAJ, lw=1.5, label="shipped calibration set")
    ax.plot([], [], color=GREY, lw=0.7, label=f"{len(taus)} other runs")
    ax.set_yscale("log")
    ax.set_yticks([0.1, 0.2, 0.5, 1])
    ax.set_yticklabels(["0.1", "0.2", "0.5", "1"])
    ax.set_ylim(0.08, 1.6)
    ax.legend(loc="lower left")
    save(fig, "fig3_ratio_step_draws")
    out["ratio_tau_other_runs_mlp_q"] = taus
    print(f"    change point (MLP, Q) in other runs: {sorted(set(map(tuple, taus.values())))}")


def ceiling_panels(pc, out, n_split=50):
    print("noise ceiling")
    rng = np.random.default_rng(0)
    L = np.arange(N_L)
    for name, fname in (("q", "fig3_noise_ceiling_q_head"), ("mlp", "fig3_noise_ceiling_mlp")):
        T, C = pc[f"traj_vlm_{name}"], pc[f"coc_vlm_{name}"]  # (100, 36, U)
        n = len(T)
        same, cross = [], []
        for _ in range(n_split):
            p = rng.permutation(n)
            A, B = p[:n // 2], p[n // 2:]
            tA, tB, cA, cB = T[A].mean(0), T[B].mean(0), C[A].mean(0), C[B].mean(0)  # (36, U)
            same.append([rho_per_layer(tA, tB), rho_per_layer(cA, cB)])
            cross.append(0.5 * (rho_per_layer(tA, cB) + rho_per_layer(tB, cA)))
        same, cross = np.mean(same, 0), np.mean(cross, 0)  # (2, 36), (36,)
        dis = cross / np.sqrt(same[0] * same[1])
        fig, ax = depth_axes(r"Rank agreement across halves ($\rho$)", (-0.2, 1.02))
        ax.plot(L, same[0], color=TRAJ, lw=1.2, label=r"$I_{\mathrm{traj}}$ vs itself")
        ax.plot(L, same[1], color=COC, lw=1.2, ls=(0, (4, 2)), label=r"$I_{\mathrm{CoC}}$ vs itself")
        ax.plot(L, cross, color="black", lw=1.5, label=r"$I_{\mathrm{traj}}$ vs $I_{\mathrm{CoC}}$")
        ax.axhline(0, color="black", lw=0.7)
        ax.legend(loc="lower left")
        save(fig, fname)
        out[f"ceiling_{name}"] = {
            k: [float(np.nanmean(v[:22])), float(np.nanmean(v[22:LAST]))]
            for k, v in (("traj_self", same[0]), ("coc_self", same[1]), ("cross", cross),
                         ("disattenuated", dis))}
        out[f"ceiling_{name}"]["ceiling_share_of_log_fall"] = ceiling_share(out[f"ceiling_{name}"])
        # layers 6-19: does the OTHER objective predict a held-out half as well as the same one?
        out[f"ceiling_{name}_L6_19"] = {"traj_self": float(same[0][6:20].mean()),
                                        "coc_self": float(same[1][6:20].mean()),
                                        "cross": float(cross[6:20].mean())}
        print(f"    {name}: " + "  ".join(f"{k} {v[0]:.2f}->{v[1]:.2f}"
                                          for k, v in out[f"ceiling_{name}"].items() if isinstance(v, list)))
        print(f"    {name} L6-19: {out[f'ceiling_{name}_L6_19']}")


def draw_reliability(outputs, out):
    """The split-half question again, on three DISJOINT 100-clip draws measured as separate runs."""
    print("independent draws (rd100_a/b/c)")
    runs = [dict(np.load(outputs / f"importance_rd100_{k}" / "importance.npz")) for k in "abc"]
    for name in ("q", "mlp"):
        t = [r[f"traj_vlm_{name}"] for r in runs]  # 3 x (36, U)
        c = [r[f"coc_vlm_{name}"] for r in runs]  # 3 x (36, U)
        ij = [(i, j) for i in range(3) for j in range(3) if i != j]
        same_t = np.mean([rho_per_layer(t[i], t[j]) for i, j in ij if i < j], 0)  # (36,)
        same_c = np.mean([rho_per_layer(c[i], c[j]) for i, j in ij if i < j], 0)  # (36,)
        cross = np.mean([rho_per_layer(t[i], c[j]) for i, j in ij], 0)  # (36,)
        dis = cross / np.sqrt(same_t * same_c)
        out[f"draws_{name}"] = {
            k: [float(np.nanmean(v[:22])), float(np.nanmean(v[22:LAST]))]
            for k, v in (("traj_self", same_t), ("coc_self", same_c), ("cross", cross),
                         ("disattenuated", dis))}
        print(f"    {name}: " + "  ".join(f"{k} {v[0]:.2f}->{v[1]:.2f}"
                                          for k, v in out[f"draws_{name}"].items()))
        out[f"draws_{name}"]["ceiling_share_of_log_fall"] = ceiling_share(out[f"draws_{name}"])


def partition_panels(pc, out):
    print("clip x objective partition")
    L = np.arange(N_L)
    for name, fname in (("q", "fig3_partition_q_head"), ("mlp", "fig3_partition_mlp")):
        T, C = pc[f"traj_vlm_{name}"], pc[f"coc_vlm_{name}"]  # (100, 36, U)
        n = len(T)
        off = ~np.eye(n, dtype=bool)
        A, Bt, Bc, Cx = (np.full(N_L, np.nan) for _ in range(4))
        for l in range(LAST):
            zt, zc = (stats.zscore(stats.rankdata(x[:, l], axis=1), axis=1) / np.sqrt(x.shape[2])
                      for x in (T, C))  # (100, U) unit-norm rank vectors -> dot product = Spearman
            tt, cc, tc = zt @ zt.T, zc @ zc.T, zt @ zc.T  # (100, 100)
            A[l], Bt[l], Bc[l], Cx[l] = np.diag(tc).mean(), tt[off].mean(), cc[off].mean(), tc[off].mean()
        B = 0.5 * (Bt + Bc)
        fig, ax = depth_axes("Share of single-clip rank variance")
        ax.plot(L, Cx, color="black", lw=1.4, label="shared")
        ax.plot(L, B - Cx, color=PURPLE, lw=1.4, ls=(0, (4, 2)), label="objective-specific")
        ax.plot(L, A - Cx, color=GREEN, lw=1.2, ls=(0, (1, 1)), label="clip-specific")
        if name == "mlp":
            ax.set_yscale("log")
            ax.set_ylim(5e-4, 1.2)
        else:
            ax.set_ylim(-0.02, 0.55)
        legend_above(ax, 3)
        save(fig, fname)
        pred = Cx / (Cx + (B - Cx))
        out[f"partition_{name}"] = {
            band: {"shared": float(np.nanmean(Cx[a:b])), "objective": float(np.nanmean((B - Cx)[a:b])),
                   "clip": float(np.nanmean((A - Cx)[a:b])),
                   "predicted_agreement": float(np.nanmean(pred[a:b]))}
            for band, a, b in (("L0-5", 0, 6), ("L6-21", 6, 22), ("L22-34", 22, LAST))}
        print(f"    {name}: {out[f'partition_{name}']}")


def boundary_panel(z, jl, out):
    print("boundary markers")
    L = np.arange(N_L)

    def mm(x):
        return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

    ratio = np.full(N_L, np.nan)
    ratio[:LAST] = log_ratio(z, "mlp")
    port = (z["traj_kv_k"] + z["traj_kv_v"]).sum(1)  # (36,) FM sensitivity to cache layer m
    kurt = jl["kurtosis"]  # (36,)
    fig, ax = depth_axes("Min-max normalized", (-0.03, 1.05), shade=False)
    ax.axvspan(20.5, 24.5, color=GREY, alpha=0.13, lw=0)
    ax.plot(L, mm(ratio), color=TRAJ, lw=1.4, label="importance ratio")
    ax.plot(L, mm(port), color=GREEN, lw=1.2, ls=(0, (1, 1)), label="FM cache sensitivity")
    kplot = np.where(L < LAST, kurt, np.nan)  # layer 35 is the read-out layer itself (31.3)
    ax.plot(L, mm(kplot), color=COC, lw=1.4, ls=(0, (4, 2)), label="lens kurtosis")
    legend_above(ax, 3)
    save(fig, "fig3_boundary")
    out["boundary"] = {
        "kurtosis_L14_20": float(kurt[14:21].mean()), "kurtosis_L24_34": float(kurt[24:35].mean()),
        "kurtosis_tau": int(step_fit(kurt[:LAST])[0]), "kurtosis_step_r2": float(step_fit(kurt[:LAST])[3]),
        "port_tau": int(step_fit(port[6:])[0]) + 6, "port_step_r2": float(step_fit(port[6:])[3]),
        "port_share_L0_22": float(port[:23].sum() / port.sum()),
        "port_mean_L8_22_over_L25_35": float(port[8:23].mean() / port[25:].mean()),
    }
    print(f"    {out['boundary']}")


def text_write_panels(z, jl, out):
    print("text-position write norm")
    L = np.arange(N_L)
    for name, fname in (("q", "fig3_text_write_q_head"), ("mlp", "fig3_text_write_mlp")):
        w = jl[f"{name}_w"]  # (36, U) write norm over text + CoC positions only
        rt, rc = rho_per_layer(w, z[f"traj_vlm_{name}"]), rho_per_layer(w, z[f"coc_vlm_{name}"])
        fig, ax = depth_axes(r"$\rho$(importance, text-position write)", (-0.5, 1.02))
        ax.plot(L, rt, color=TRAJ, lw=1.5, label=r"$I_{\mathrm{traj}}$")
        ax.plot(L, rc, color=COC, lw=1.5, label=r"$I_{\mathrm{CoC}}$")
        ax.axhline(0, color="black", lw=0.7)
        ax.legend(loc="lower left")
        save(fig, fname)
        out[f"text_write_{name}"] = {
            "traj": [float(np.nanmean(rt[6:22])), float(np.nanmean(rt[22:LAST]))],
            "coc": [float(np.nanmean(rc[6:22])), float(np.nanmean(rc[22:LAST]))]}
        print(f"    {name}: L6-21 -> L22-34  {out[f'text_write_{name}']}")


def pair_panels(z, outputs, out):
    print("objective pairs")
    L = np.arange(N_L)
    with np.load(outputs / "importance_vqa" / "importance.npz") as v, \
            np.load(outputs / "importance_dim0" / "importance.npz") as d0, \
            np.load(outputs / "importance_dim1" / "importance.npz") as d1:
        for name, fname in (("q", "fig3_pairs_q_head"), ("mlp", "fig3_pairs_mlp")):
            k = f"_vlm_{name}"
            pairs = (
                ("trajectory vs CoC", z["traj" + k], z["coc" + k], "black", "-", 1.6),
                ("CoC vs CoC, other dataset", z["coc" + k], v["coc" + k], COC, (0, (4, 2)), 1.1),
                ("accel. vs curvature", d0["traj" + k], d1["traj" + k], TRAJ, "-", 1.1),
                ("VQA vs CoC, same images", v["vqa" + k], v["coc" + k], PURPLE, (0, (1, 1)), 1.1),
            )
            fig, ax = depth_axes(r"Rank agreement ($\rho$)", (-0.2, 1.02))
            out[f"pairs_{name}"] = {}
            for label, a, b, col, ls, lw in pairs:
                r = rho_per_layer(a, b)
                ax.plot(L, r, color=col, lw=lw, ls=ls, label=label)
                out[f"pairs_{name}"][label] = [float(np.nanmean(r[6:22])), float(np.nanmean(r[22:LAST]))]
            ax.axhline(0, color="black", lw=0.7)
            legend_above(ax, 2)
            save(fig, fname)
            print(f"    {name}: L6-21 -> L22-34  {out[f'pairs_{name}']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=REPO / "outputs")
    parser.add_argument("--out-dir", type=Path, default=fc.OUT)
    args = parser.parse_args()
    fc.OUT = args.out_dir
    o = args.outputs
    out = {}
    with np.load(o / "importance_v2" / "importance.npz") as z, \
            np.load(o / "importance_v2" / "importance_perclip.npz") as pc, \
            np.load(o / "jlens_v2" / "jlens.npz") as jl:
        ratio_panels(z, pc, o, out)
        ceiling_panels(pc, out)
        draw_reliability(o, out)
        partition_panels(pc, out)
        boundary_panel(z, jl, out)
        text_write_panels(z, jl, out)
        pair_panels(z, o, out)
    (args.out_dir / "fig3_why_differs_stats.json").write_text(json.dumps(out, indent=2))
    print(f"  {args.out_dir / 'fig3_why_differs_stats.json'}")


if __name__ == "__main__":
    main()
