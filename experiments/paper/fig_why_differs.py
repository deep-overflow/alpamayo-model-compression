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

  fig4_token_handoff_mlp / fig4_token_handoff_q_head      (needs outputs/gradanat_v1)
      Additive share of each shipped layer score that comes from what the unit does at
      VISION positions (signed shares; over token types they sum to the layer's score).
      Both scores hand over from vision to text-side positions across the same layers.
  fig4_ratio_by_token_mlp / fig4_ratio_by_token_q_head    (needs outputs/gradanat_v1)
      fig3_ratio_step again, with both scores restricted to the unit's action at vision
      tokens, and at text-side tokens. The step is absent at vision tokens.
  fig4_same_token_q_head                                  (needs outputs/gradanat_v1)
      Ceiling-corrected agreement between the two objectives with both restricted to the
      same token type, next to the pooled (shipped) agreement. The pre-registered test of
      whether the late-layer split is a difference in token support.
  fig4_itraj_ports_mlp / fig4_itraj_ports_q_head          (needs outputs/portmap_v1)
      I_traj per layer rebuilt from the cache layer the FM gradient arrives through
      (additive; a unit in layer l cannot reach cache layers <= l). The bars sum to the
      blue curve of fig1_depth_*.
  fig4_port_profile                                       (needs outputs/portmap_v1)
      Share of all VLM I_traj carried by each cache layer, split by whether it enters
      through vision or non-vision cache entries.

  fig5_*   does the score anatomy predict function?   (plans/2026-09-21_importance-causal-validation.md)
      fig5_probe_ratio_mlp / _q_head   the ratio of fig3_ratio_step next to the same ratio for two
          RANDOM linear readouts through the same two doors (expert output vs LM-head input) and
          for a random readout taken straight off every cache entry: the step is the doors'.
      fig5_probe_rank_q_head           ceiling-corrected rank agreement of each probe with the real
          loss through its door, and of the two doors' probes with each other.
      fig5_matched_fm / fig5_matched_nll   the three shipped arms on the dense model's text (same
          tokens, same noise): FM loss and CoC NLL, arm minus dense, per clip.
      fig5_dissociation_q_head / _mlp  token-targeted ablation on held-out clips: damage to the FM
          loss against damage to the CoC NLL for the trajectory-side, CoC-side and random sets.
      fig5_dual_saves                  the units each single criterion dropped and the dual one kept.
      fig5_vv_nested / fig5_vv_window_profile / fig5_vision_census   vision-vision interaction by
          depth: nested knockouts, their window profile against I_traj at vision tokens, and where
          vision queries put their attention.
      fig5_failure_by_manoeuvre        open-loop damage of the three arms by driving situation.

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


def anatomy_panels(outputs, out, exp="gradanat_v1"):
    path = outputs / exp / "metrics_analysis.json"
    if not path.exists():
        print(f"anatomy panels skipped: {path} missing")
        return
    print("gradient anatomy")
    m = json.loads(path.read_text())
    L = np.arange(N_L)
    t_vis, t_pt, t_coc = 0, 2, 3  # rows of the (5, 36) share arrays
    for name, fname in (("mlp", "fig4_token_handoff_mlp"), ("q", "fig4_token_handoff_q_head")):
        fm = np.array(m["additive_by_type"][f"{name}_fm"]["share_by_type_and_layer"])  # (5, 36)
        ce = np.array(m["additive_by_type"][f"{name}_ce"]["share_by_type_and_layer"])  # (5, 36)
        fig, ax = depth_axes("Share of importance from vision tokens", (-0.05, 1.05))
        ax.plot(L[:LAST], fm[t_vis][:LAST], color=TRAJ, lw=1.5, label=r"$I_{\mathrm{traj}}$")
        ax.plot(L, ce[t_vis], color=COC, lw=1.5, label=r"$I_{\mathrm{CoC}}$")
        ax.axhline(0, color="black", lw=0.7)
        ax.legend(loc="upper right")
        save(fig, fname)
        # hand-off layer: the first layer from which the vision share stays below one half
        # through layer 32 (a single early dip does not count; Q-head shares are jagged)
        half = {k: next(l for l in range(6, 33) if (v[t_vis][l:33] < 0.5).all())
                for k, v in (("traj", fm), ("coc", ce))}
        out[f"handoff_{name}"] = {
            "traj_vision": [float(fm[t_vis][6:18].mean()), float(fm[t_vis][27:LAST].mean())],
            "coc_vision": [float(ce[t_vis][6:18].mean()), float(ce[t_vis][27:LAST].mean())],
            "traj_text": [float((fm[t_pt] + fm[t_coc])[6:18].mean()), float((fm[t_pt] + fm[t_coc])[27:LAST].mean())],
            "coc_text": [float((ce[t_pt] + ce[t_coc])[6:18].mean()), float((ce[t_pt] + ce[t_coc])[27:LAST].mean())],
            "first_layer_below_half": half}
        print(f"    {name}: vision share L6-17 -> L27-34  {out[f'handoff_{name}']}")

    with np.load(outputs / exp / "anatomy.npz") as z:
        for name, fname in (("mlp", "fig4_ratio_by_token_mlp"), ("q", "fig4_ratio_by_token_q_head")):
            parts = {"pooled": (z[f"fm_full_{name}"], z[f"ce_full_{name}"]),
                     "vision tokens": (z[f"fm_type_{name}"][t_vis], z[f"ce_{name}"][t_vis]),
                     "text tokens": (z[f"fm_text_{name}"], z[f"ce_text_{name}"])}
            fig, ax = depth_axes(r"$I_{\mathrm{traj}}\,/\,I_{\mathrm{CoC}}$")
            out[f"ratio_by_token_{name}"] = {}
            for (label, (t, c)), col, ls, lw in zip(parts.items(), ("black", TRAJ, PURPLE),
                                                    ("-", "-", (0, (4, 2))), (1.6, 1.2, 1.2)):
                tl, cl = t.sum(-1)[:LAST], c.sum(-1)[:LAST]  # (35,) layer sums
                y = np.log(tl / cl)
                tau, _, _, r2 = step_fit(y)
                ax.plot(L[:LAST], np.exp(y), color=col, lw=lw, ls=ls, label=label)
                out[f"ratio_by_token_{name}"][label] = {
                    "L0_21": float(np.exp(y[:22].mean())), "L23_34": float(np.exp(y[23:].mean())),
                    "step_at_23": float(np.exp(y[:22].mean() - y[23:].mean())),
                    "best_step_tau": int(tau), "best_step_r2": float(r2),
                    "traj_L18_21_to_L24_29": float(tl[24:30].mean() / tl[18:22].mean()),
                    "coc_L18_21_to_L24_29": float(cl[24:30].mean() / cl[18:22].mean())}
            ax.set_yscale("log")
            ax.set_yticks([1, 2, 5, 10, 20])
            ax.set_yticklabels(["1", "2", "5", "10", "20"])
            ax.set_ylim(0.6, 45)
            legend_above(ax, 3)
            save(fig, fname)
            print(f"    {name} ratio by token: {out[f'ratio_by_token_{name}']}")

    fig, ax = depth_axes(r"Corrected rank agreement ($\rho$)", (-0.3, 1.1))
    series = (("pooled (the shipped score)", "pooled", "black", "-", 1.6),
              ("both at VISION positions", "vision tokens only", TRAJ, "-", 1.1),
              ("both at TEXT-side positions", "text tokens only", PURPLE, (0, (4, 2)), 1.1))
    out["same_token_q"] = {}
    for key, label, col, ls, lw in series:
        b = m["P2"]["q"][key]
        st, sc, cr = (np.array(b["per_layer"][k]) for k in ("self_traj", "self_coc", "cross"))
        ok = (st > 0.2) & (sc > 0.2)
        y = np.where(ok, cr / np.sqrt(np.where(ok, st * sc, 1.0)), np.nan)
        ax.plot(L[:LAST], y, color=col, lw=lw, ls=ls, label=label)
        out["same_token_q"][label] = {"early": b["early"]["corrected"], "late": b["late"]["corrected"],
                                      "late_self": [b["late"]["self_traj"], b["late"]["self_coc"]],
                                      "late_layers_used": b["late"]["layers_used"]}
    for key in ("both at CoC positions only", "both at prompt-text positions only",
                "traj at vision vs CoC at text", "traj: vision vs text-side", "CoC: vision vs text-side"):
        b = m["P2"]["q"][key]
        out["same_token_q"][key] = {"early": b["early"]["corrected"], "late": b["late"]["corrected"]}
    out["same_token_mlp"] = {k: {"early": v["early"]["corrected"], "late": v["late"]["corrected"]}
                             for k, v in m["P2"]["mlp"].items()}
    out["anatomy_gates"] = {
        "G0_typed_sum_relerr_max": m["G0"].get("typed_vs_single_relerr_max"),
        "G0_seed_split_relerr_median_layer": m["G0"]["split_vs_full_relerr_median_layer"],
        "G0_seed_split_relerr_worst": m["G0"]["split_vs_full_relerr_worst_layer_max"],
        "G0_reproduces_ref": m["G0"]["reproduces_ref"],
        "P1_pass": m["P1"]["pass"], "P1_q": {k: v for k, v in m["P1"]["q"].items() if not isinstance(v, list)},
        "P1_mlp": {k: v for k, v in m["P1"]["mlp"].items() if not isinstance(v, list)},
        "P2_decisive_value": m["P2"]["decisive_value"], "P2_verdict": m["P2"]["verdict"],
        "P3_verdict": m["P3"]["verdict"],
        "P3_late_cache_share": [m["P3"]["q"]["late_cache_share_units_L8_21"],
                                m["P3"]["mlp"]["late_cache_share_units_L8_21"]],
        "P4_pass": m["P4"]["pass"],
        "P4_vision_share_mean": [m["P4"]["q"]["vision_share_mean"], m["P4"]["mlp"]["vision_share_mean"]],
        "P4_share_by_type_mlp": m["P4"]["mlp"]["share_mean_by_type"],
        "residual_cosine": {k: v for k, v in m["residual_cosine"].items() if not isinstance(v, list)}}
    ax.axhline(0, color="black", lw=0.7)
    legend_above(ax, 3)
    save(fig, "fig4_same_token_q_head")
    print(f"    same-token agreement (Q): {out['same_token_q']}")


def port_panels(outputs, out, exp="portmap_v1"):
    path = outputs / exp / "port_map.npz"
    if not path.exists():
        print(f"port panels skipped: {path} missing")
        return
    print("port map")
    L = np.arange(N_L)
    groups = (("cache 1-21", 0, 22, "#BBBBBB"), ("cache 22-24", 22, 25, "#E69F00"),
              ("cache 25-29", 25, 30, "#56B4E9"), ("cache 30-35", 30, 36, TRAJ))
    with np.load(path) as z:
        for name, fname in (("mlp", "fig4_itraj_ports_mlp"), ("q", "fig4_itraj_ports_q_head")):
            S = z[f"{name}_signed"].sum((1, 2))  # (36 cache layers, 36 unit layers)
            full = z[f"{name}_full"]  # (36,)
            fig, ax = depth_axes("Normalized importance", (0, 1.05), shade=False)
            bottom = np.zeros(LAST)
            for label, lo, hi, col in groups:
                y = S[lo:hi, :LAST].sum(0) / full.max()
                ax.bar(L[:LAST], y, bottom=bottom, color=col, width=0.9, lw=0, label=label)
                bottom += y
            ax.plot(L[:LAST], full[:LAST] / full.max(), color="black", lw=0.9, label=r"$I_{\mathrm{traj}}$")
            legend_above(ax, 3)
            save(fig, fname)
            out[f"ports_{name}"] = {
                "rebuilt_over_shipped_min_max": [float((S.sum(0)[:LAST] / full[:LAST]).min()),
                                                 float((S.sum(0)[:LAST] / full[:LAST]).max())],
                "units_16_21_share_by_group": {g[0]: float(S[g[1]:g[2], 16:22].sum() / full[16:22].sum())
                                               for g in groups},
                "fall_21_to_23": float(full[21] - full[23]),
                "closed_ports_share_of_fall": float(S[22:24, 21].sum() / (S[:, 21].sum() - S[:, 23].sum()))}
            print(f"    {name}: {out[f'ports_{name}']}")
        Sp = z["mlp_signed"].sum(2)  # (36, 2, 36) cache layer x position group x unit layer
        tot = Sp.sum(2) / z["mlp_full"][:LAST].sum()  # (36, 2)
        fig, ax = depth_axes(r"Share of all $I_{\mathrm{traj}}$", shade=False)
        ax.bar(L, tot[:, 0], color=TRAJ, width=0.85, lw=0, label="through vision cache entries")
        ax.bar(L, tot[:, 1], bottom=tot[:, 0], color="#E69F00", width=0.85, lw=0,
               label="through text-side entries")
        ax.set_xlabel("VLM cache layer")
        ax.set_xticks([0, 7, 14, 21, 28, 35])
        legend_above(ax, 1)
        save(fig, "fig4_port_profile")
        for name in ("mlp", "q"):
            prof = z[f"{name}_signed"].sum((1, 2, 3))[:N_L] / z[f"{name}_full"][:LAST].sum()  # (36,)
            out[f"port_profile_{name}"] = {
                "top_cache_layers": [int(i) for i in np.argsort(prof)[::-1][:5]],
                "cache_22": float(prof[22]), "cache_21_23": float(prof[21:24].sum()),
                "cache_le_20": float(prof[:21].sum()), "cache_ge_22": float(prof[22:].sum())}
        trunk = Sp[:, :, 6:22].sum(2)  # (36, 2)
        out["ports_trunk_doors"] = {"cache_ge_22": float(trunk[22:].sum() / trunk.sum()),
                                    "late_nonvision": float(trunk[22:, 1].sum() / trunk.sum()),
                                    "late_vision": float(trunk[22:, 0].sum() / trunk.sum()),
                                    "early_vision": float(trunk[:22, 0].sum() / trunk.sum()),
                                    "early_nonvision": float(trunk[:22, 1].sum() / trunk.sum())}
        print(f"    trunk doors: {out['ports_trunk_doors']}")
        print(f"    port profile: {out['port_profile_mlp']} {out['port_profile_q']}")
    pm = outputs / exp / "metrics_analysis.json"
    if pm.exists():
        m = json.loads(pm.read_text())
        out["port_gates"] = {a: {k: m[a][k] for k in (
            "pm1_share_by_group", "pm1_argmax_cache_layer", "pm1_pass", "pm2_closed_share_of_fall",
            "pm2_verdict", "pm2_I21", "pm2_I23", "pm3_vision_share_ports_le15",
            "pm3_vision_share_ports_ge22", "pm3_pass", "structural_zero_max")} for a in ("mlp", "q")}
        out["port_gates"]["integrity"] = m["integrity"]


def _load(path):
    return json.loads(path.read_text()) if path.exists() else None


def causal_panels(outputs, out):
    """fig5_*: panels from the analysis outputs of the causal-validation plan; each is skipped
    when its run is not there yet."""
    print("causal validation")
    L = np.arange(LAST)
    ORANGE = "#E69F00"

    probes = outputs / "gradanat_probes_v1" / "anatomy.npz"
    pm = _load(outputs / "gradanat_probes_v1" / "metrics_analysis.json")
    if probes.exists() and pm:
        with np.load(probes) as z:
            for name, fname in (("mlp", "fig5_probe_ratio_mlp"), ("q", "fig5_probe_ratio_q_head")):
                prof = {k: z[f"{k}_full_{name}"].sum(1)[:LAST] for k in ("fm", "ce", "pr_head", "pr_expert", "pr_cache")}
                fig, ax = depth_axes("Ratio / its layer 0-21 mean")
                for (a, b), col, ls, lw, label in (
                        (("fm", "ce"), "black", "-", 1.6, r"$I_{\mathrm{traj}}\,/\,I_{\mathrm{CoC}}$"),
                        (("pr_expert", "pr_head"), TRAJ, "-", 1.2, "random readouts: expert door / head door"),
                        (("pr_cache", "pr_head"), ORANGE, (0, (4, 2)), 1.2, "random readouts: bare cache / head door")):
                    y = prof[a] / prof[b]
                    ax.plot(L, y / y[:22].mean(), color=col, ls=ls, lw=lw, label=label)
                ax.set_yscale("log")
                ax.set_yticks([0.1, 0.2, 0.5, 1, 2])
                ax.set_yticklabels(["0.1", "0.2", "0.5", "1", "2"])
                legend_above(ax, 1)
                save(fig, fname)
        d3 = pm["D3"]
        rows = (("expert probe\nvs $I_{\\mathrm{traj}}$", d3["expert"]), ("head probe\nvs $I_{\\mathrm{CoC}}$", d3["head"]),
                ("expert probe\nvs head probe", d3["expert_vs_head_probe"]))
        fig, ax = plt.subplots(figsize=SINGLE)
        x = np.arange(len(rows))
        ax.bar(x - 0.2, [r[1]["trunk"]["corrected"] for r in rows], width=0.38, color=TRAJ, label="layers 6-21")
        ax.bar(x + 0.2, [r[1]["late"]["corrected"] for r in rows], width=0.38, color=ORANGE, label="layers 22-34")
        ax.set_xticks(x, [r[0] for r in rows], fontsize=6)
        ax.set_ylabel("Rank agreement (ceiling corrected)")
        ax.set_ylim(0, 1.05)
        legend_above(ax, 2)
        save(fig, "fig5_probe_rank_q_head")
        out["probes"] = {k: pm[k] for k in ("D1", "D2", "D3", "steps")}
        out["probes"]["D4"] = {k: v for k, v in pm["D4"].items() if not isinstance(v, dict) or "fm" not in v}

    pa = _load(outputs / "gradanat_pruned_v1" / "metrics.json")
    if pa:
        arms = ("dual", "traj", "coc")
        per = {a: {r["clip_id"]: r for r in _load(outputs / f"gradanat_{a}_u40" / "metrics.json")["per_clip"]} for a in arms}
        dense = {r["clip_id"]: r for r in _load(outputs / "gradanat_probes_v1" / "metrics.json")["per_clip"]}
        for key, fname, ylabel in (("fm_loss", "fig5_matched_fm", "FM loss: arm - dense"),
                                   ("nll", "fig5_matched_nll", "CoC NLL: arm - dense")):
            d = [np.array([per[a][c][key] - dense[c][key] for c in dense]) for a in arms]
            fig, ax = plt.subplots(figsize=SINGLE)
            ax.boxplot(d, showfliers=False, widths=0.55, medianprops={"color": "black", "lw": 1.0},
                       boxprops={"lw": 0.7}, whiskerprops={"lw": 0.7}, capprops={"lw": 0.7})
            ax.axhline(0, color="black", lw=0.6)
            ax.set_xticks([1, 2, 3], ["dual", "trajectory-only", "CoC-only"])
            ax.set_ylabel(ylabel)
            save(fig, fname)
        out["pruned"] = {"A1_gate": pa["A1"]["gate"], "A2": pa["A2"], "S0": {k: v for k, v in pa["S0"].items() if "gate" in k}}

    ah = _load(outputs / "armheld_v1" / "metrics.json")
    if ah and "bands" in ah:
        for key, fname, ylabel in (("fm", "fig5_damage_by_band_fm", "FM loss: mask - dense"),
                                   ("nll", "fig5_damage_by_band_nll", "CoC NLL: mask - dense")):
            fig, ax = plt.subplots(figsize=SINGLE)
            for i, (a, col, label) in enumerate((("dual", "#009E73", "dual"), ("traj", TRAJ, "trajectory-only"),
                                                 ("coc", COC, "CoC-only"))):
                cfgs = (f"{a}_trunk", f"{a}_late", a)
                vals = [ah["configs"][c][key]["delta"] for c in cfgs]
                err = [[v - ah["configs"][c][key]["ci"][0] for v, c in zip(vals, cfgs)],
                       [ah["configs"][c][key]["ci"][1] - v for v, c in zip(vals, cfgs)]]
                ax.bar(np.arange(3) + 0.27 * (i - 1), vals, width=0.25, color=col, yerr=err,
                       error_kw={"lw": 0.6}, label=label)
            ax.set_xticks(np.arange(3), ["layers 0-21\nonly", "layers 22-35\nonly", "all layers"])
            ax.axhline(0, color="black", lw=0.5)
            ax.set_ylabel(ylabel)
            legend_above(ax, 3)
            save(fig, fname)
        out["arm_heldout"] = {"tests": ah["tests"], "pass": ah["pass"], "bands": ah["bands"],
                              "nll_gap_traj_minus_coc": ah["nll_gap_traj_minus_coc"],
                              "configs": ah["configs"]}
    mx = _load(outputs / "armmix_v1" / "metrics.json")
    if mx:
        out["arm_mix"] = {k: mx[k] for k in ("configs", "a", "b", "c", "reproducibility_max_abs_diff")}

    tb = _load(outputs / "tokabl_v1" / "metrics.json")
    if tb:
        meta = {c["name"]: c for c in _load(outputs / "tokabl_v1_s0" / "config.json")["configs"]}
        for name, fname in (("q", "fig5_dissociation_q_head"), ("mlp", "fig5_dissociation_mlp")):
            fig, ax = plt.subplots(figsize=SINGLE)
            for c, r in tb["configs"].items():
                if not c.startswith(f"{name}_late_") or meta[c]["set"].startswith(("S", "RS")):
                    continue
                st = meta[c]["set"]
                col = TRAJ if st.startswith("T") else COC if st.startswith("C") else GREY
                ax.errorbar(r["dfm_mean"], r["dnll_mean"],
                            xerr=[[r["dfm_mean"] - r["dfm_ci"][0]], [r["dfm_ci"][1] - r["dfm_mean"]]],
                            yerr=[[r["dnll_mean"] - r["dnll_ci"][0]], [r["dnll_ci"][1] - r["dnll_mean"]]],
                            fmt="o" if st.endswith("tok") else "s" if st.endswith("pool") else ".",
                            color=col, ms=4, lw=0.6, capsize=0)
            ax.axhline(0, color="black", lw=0.5)
            ax.axvline(0, color="black", lw=0.5)
            ax.set_xlabel("FM loss: set removed - dense")
            ax.set_ylabel("CoC NLL: set removed - dense")
            # the CoC handle's colour as RGBA: save() dashes every legend line whose colour == COC,
            # which would draw a line through a marker-only handle
            ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=4, color=TRAJ, label="trajectory-side sets"),
                               plt.Line2D([], [], marker="o", ls="", ms=4, color=plt.matplotlib.colors.to_rgba(COC),
                                          label="CoC-side sets"),
                               plt.Line2D([], [], marker=".", ls="", ms=5, color=GREY, label="random sets")],
                      loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6, handletextpad=0.2,
                      columnspacing=0.8, borderaxespad=0.2)
            save(fig, fname)
        # what the dual criterion saves: NLL only (no late-layer set moves the FM loss by 1%)
        fig, ax = plt.subplots(figsize=SINGLE)
        groups = (("q", "Straj", "heads\nkept over\ntrajectory-only"), ("q", "Scoc", "heads\nkept over\nCoC-only"),
                  ("mlp", "Straj", "channels\nkept over\ntrajectory-only"), ("mlp", "Scoc", "channels\nkept over\nCoC-only"))
        base = tb["dense"]["nll"]
        for i, (name, st, _) in enumerate(groups):
            rs = [100 * tb["configs"][f"{name}_late_R{st}{j}"]["dnll_mean"] / base for j in range(3)]
            r = tb["configs"][f"{name}_late_{st}"]
            ax.bar(i - 0.2, 100 * r["dnll_mean"] / base, width=0.38, color=COC if st == "Straj" else TRAJ,
                   yerr=[[100 * (r["dnll_mean"] - r["dnll_ci"][0]) / base], [100 * (r["dnll_ci"][1] - r["dnll_mean"]) / base]],
                   error_kw={"lw": 0.6}, label="removed set" if i == 0 else None)
            ax.bar(i + 0.2, np.mean(rs), width=0.38, color=GREY,
                   yerr=[[np.mean(rs) - min(rs)], [max(rs) - np.mean(rs)]], error_kw={"lw": 0.6},
                   label="random sets of the same size" if i == 0 else None)
        ax.set_xticks(np.arange(len(groups)), [g[2] for g in groups], fontsize=5.5)
        ax.set_ylabel("CoC NLL increase (% of dense)")
        ax.axhline(0, color="black", lw=0.5)
        legend_above(ax, 2)
        save(fig, "fig5_dual_saves")
        out["token_ablation"] = {"gates": tb["gates"], "first_order": tb["first_order"], "n_clips": tb["n_clips"]}

    vv = _load(outputs / "vvdepth_v1" / "metrics.json")
    if vv:
        cols = {"VV_allvision": "black", "E1_crossframe": TRAJ, "E2_crosscam": "#009E73", "V3_ownimage": ORANGE}
        labels = {"VV_allvision": "any other vision token", "E1_crossframe": "same-camera earlier frames",
                  "E2_crosscam": "other cameras", "V3_ownimage": "own image"}
        fig, ax = plt.subplots(figsize=SINGLE)
        for e, col in cols.items():
            d = vv["curves"].get(f"{e}|from")
            if d:
                x = [int(c) for c in d]
                ax.plot(x, [100 * d[str(c)]["ade"]["rel"] for c in x], color=col, lw=1.3, marker="o", ms=2.5, label=labels[e])
                ax.fill_between(x, [100 * d[str(c)]["ade"]["rel_lo"] for c in x],
                                [100 * d[str(c)]["ade"]["rel_hi"] for c in x], color=col, alpha=0.10, lw=0)
        ax.axhspan(-5, 5, color=GREY, alpha=0.15, lw=0)
        ax.set_xlabel("Vision <- ... blocked from this layer to the last")
        ax.set_ylabel("minADE change (% of baseline)")
        ax.set_xticks([0, 6, 12, 18, 24, 30])
        legend_above(ax, 2)
        save(fig, "fig5_vv_nested")
        cuts = vv["cuts"] + [36]
        centers = [(a + b - 1) / 2 for a, b in zip(cuts[:-1], cuts[1:])]
        prof = np.array(vv["window_profile"]["VV_allvision"]["ade"])
        with np.load(outputs / "gradanat_v1" / "anatomy.npz") as z:
            fig, ax = plt.subplots(figsize=SINGLE)
            ax.bar(centers, prof / prof.max(), width=2.6, color=GREY, alpha=0.45, lw=0,
                   label="action damage added by the window")
            for key, col, label in (("fm_type_q", TRAJ, r"$I_{\mathrm{traj}}$ at vision tokens"),
                                    ("ce_q", COC, r"$I_{\mathrm{CoC}}$ at vision tokens")):
                lay = z[key][0].sum(1)
                w = np.array([lay[a:b].mean() for a, b in zip(cuts[:-1], cuts[1:])])
                ax.plot(centers, w / w.max(), color=col, lw=1.3, marker="o", ms=2.5, label=label)
            ax.set_xlabel("VLM layer")
            ax.set_ylabel("Window profile / its maximum")
            legend_above(ax, 1)
            save(fig, "fig5_vv_window_profile")
        out["vv_depth"] = {k: vv[k] for k in ("C1", "C2", "C3", "C4", "integrity", "window_profile") if k in vv}
        if "census_layer_mean" in vv:
            lay = np.array(vv["census_layer_mean"])  # (36, 5)
            fig, ax = plt.subplots(figsize=SINGLE)
            ax.stackplot(np.arange(N_L), lay.T, colors=("#BBBBBB", "#DDDDDD", ORANGE, TRAJ, "#009E73"), lw=0,
                         labels=("sink", "text", "own image", "earlier frames, same camera", "other cameras"))
            ax.set_xlabel("VLM layer")
            ax.set_ylabel("Attention mass of vision queries")
            ax.set_xlim(0, 35)
            ax.set_ylim(0, 1)
            legend_above(ax, 3)
            save(fig, "fig5_vision_census")

    fs = _load(outputs / "failure_by_situation_v1" / "metrics.json")
    if fs:
        b4 = ["cruise", "accel", "decel_stop", "turn"]
        fig, ax = plt.subplots(figsize=SINGLE)
        for i, (a, col, label) in enumerate((("coc", COC, "CoC-only"), ("traj", TRAJ, "trajectory-only"), ("dual", "#009E73", "dual"))):
            ax.bar(np.arange(4) + 0.27 * (i - 1), [100 * fs["ALL"][b][a]["rel"] for b in b4], width=0.25, color=col, label=label)
        ax.set_xticks(np.arange(4), ["cruise", "accelerate", "decelerate\n/ stop", "turn"])
        ax.set_ylabel("minADE increase (% of dense)")
        legend_above(ax, 3)
        save(fig, "fig5_failure_by_manoeuvre")
        out["failure_by_situation"] = {k: fs[k] for k in ("ALL", "confound", "calib_mass") if k in fs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=REPO / "outputs")
    parser.add_argument("--out-dir", type=Path, default=fc.OUT)
    parser.add_argument("--anatomy", default="gradanat_v1")
    parser.add_argument("--portmap", default="portmap_v1")
    parser.add_argument("--only-causal", action="store_true",
                        help="only the fig5_* panels and their stats file; fig3_* / fig4_* stay untouched")
    args = parser.parse_args()
    fc.OUT = args.out_dir
    o = args.outputs
    out = {}
    causal = {}
    causal_panels(o, causal)
    (args.out_dir / "fig5_causal_validation_stats.json").write_text(json.dumps(causal, indent=2))
    print(f"  {args.out_dir / 'fig5_causal_validation_stats.json'}")
    if args.only_causal:
        return
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
    anatomy_panels(o, out, args.anatomy)
    port_panels(o, out, args.portmap)
    (args.out_dir / "fig3_why_differs_stats.json").write_text(json.dumps(out, indent=2))
    print(f"  {args.out_dir / 'fig3_why_differs_stats.json'}")


if __name__ == "__main__":
    main()
