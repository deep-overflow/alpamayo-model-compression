"""Gates and maps for the cache-use runner (plans/2026-08-28_cache-use-map.md).

Merges the shards of run_cacheuse.py, judges G0-G4 and draws the layer x step maps:
attention mass on the cache, readout share, causal move, and the skip-candidate map
(cells whose blocking moves the trajectory less than the seed-noise floor).

Usage:
  .venv/bin/python experiments/head_analysis/analyze_cacheuse.py \
      --shards cacheuse_v1_s0 cacheuse_v1_s25 cacheuse_v1_s50 cacheuse_v1_s75 --out cacheuse_v1
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

BG, INK, MUTED, GRID = "#FAF9F5", "#29261B", "#6B6555", "#E8E6DC"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.size": 9, "axes.grid": True, "axes.axisbelow": True,
})
REPO = Path(__file__).resolve().parents[2]
SPANS = ("vision", "text", "hist", "sink", "coc")


def merge(shards):
    per, stat, n = {}, {}, 0
    for s in shards:
        m = json.loads((REPO / "outputs" / s / "metrics.json").read_text())
        z = dict(np.load(REPO / "outputs" / s / "cacheuse.npz"))
        k = m["n_clips"]
        for key, v in z.items():
            if key.startswith("stat_"):
                stat[key] = stat.get(key, 0) + v * k  # saved as per-clip means
            else:
                per.setdefault(key, []).append(v[:k])
        per.setdefault("clip_ids", []).append(np.array(m["clip_ids"][:k]))
        per.setdefault("buckets", []).append(np.array(m["buckets"][:k]))
        n += k
    per = {key: np.concatenate(v) for key, v in per.items()}
    stat = {key[5:]: v / n for key, v in stat.items()}
    return per, stat, n


def med_ci(x, n=5000, seed=0):
    """Median with a bootstrap CI. Blocking the cache makes some clips' trajectories
    explode (moves of 50-130 m), so the median is the primary reading everywhere and the
    mean is reported beside it."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    b = np.median(x[rng.integers(0, len(x), (n, len(x)))], 1)
    return float(np.median(x)), *[float(q) for q in np.percentile(b, [2.5, 97.5])]


EXPLODE = 10.0  # m; a same-seed move this large is a diverged trajectory, not a nudge


def ci_mean(x, n=5000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), *[float(q) for q in np.percentile(b, [2.5, 97.5])]


def heat(ax, m, title, xlabel="denoising step", ylabel="expert layer", cmap="viridis",
         vmin=None, vmax=None, fmt=None):
    im = ax.imshow(m, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, format=fmt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", default="cacheuse_v1")
    args = ap.parse_args()
    out = REPO / "outputs" / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)
    per, stat, n = merge(args.shards)
    L, S = per["move_grid"].shape[1], per["move_grid"].shape[2]

    # --- G0
    g0 = {"steps_seen": sorted({int(x) for x in per["steps_seen"]}),
          "none_block_move_max": float(per["move_none"].max()),
          "all_block_move": med_ci(per["move_all"]),
          "all_block_move_mean": ci_mean(per["move_all"]),
          "all_block_exploded_share": float((per["move_all"] > EXPLODE).mean()),
          "all_block_dade": med_ci(per["ade_all"] - per["ade_ref"]),
          "noise_floor": med_ci(per["noise_floor"])}
    # the seed floor is the spread between DIFFERENT seeds; every move here is same-seed, so
    # the positive control is judged on its own scale: a large move that also hurts minADE
    g0["pass"] = (g0["steps_seen"] == [10] and g0["none_block_move_max"] == 0.0
                  and g0["all_block_move"][1] > 0.5 and g0["all_block_dade"][1] > 0)

    # --- maps (clip means)
    move = np.nanmedian(per["move_grid"], 0)  # (L, S) median over clips
    move_mean = np.nanmean(per["move_grid"], 0)
    explode = np.nanmean(per["move_grid"] > EXPLODE, 0)
    move_end = np.nanmedian(per["move_grid_end"], 0)
    dade = np.nanmedian(per["ade_grid"] - per["ade_ref1"][:, None, None], 0)
    mass = stat["mass_cache"].mean(-1).T  # (S, L, H) -> (L, S)
    share = stat["read_share"].mean(-1).T
    ent = stat["entropy"].mean(-1).T
    span_mass = {s: stat[f"mass_{s}"].mean(-1).T for s in SPANS}
    floor = float(np.mean(per["noise_floor"]))

    # --- G1 agreement over the 360 cells
    ok = ~np.isnan(move)
    g1 = {"spearman_mass_move": [float(x) for x in spearmanr(mass[ok], move[ok])],
          "spearman_share_move": [float(x) for x in spearmanr(share[ok], move[ok])],
          "spearman_mass_share": [float(x) for x in spearmanr(mass[ok], share[ok])]}
    g1["mass_is_proxy"] = g1["spearman_mass_move"][0] >= 0.9

    # --- G2 step trend (step marginals, all layers blocked at one step)
    ms = per["move_step"]  # (n, S)
    step_mean = np.median(ms, 0)
    adj = [med_ci(ms[:, s + 1] - ms[:, s]) for s in range(S - 1)]
    g2 = {"step_marginal_move": step_mean.tolist(),
          "step_marginal_move_mean": ms.mean(0).tolist(),
          "step_marginal_exploded_share": (ms > EXPLODE).mean(0).tolist(),
          "step_marginal_move_ci": [med_ci(ms[:, s]) for s in range(S)],
          "adjacent_diff_ci": adj,
          "monotone_decreasing": all(a[2] < 0 for a in adj),
          "monotone_increasing": all(a[1] > 0 for a in adj),
          "spearman_step_vs_move": [float(x) for x in spearmanr(np.arange(S), step_mean)],
          "grid_step_profile": np.nanmean(move, 0).tolist(),
          "mass_step_profile": mass.mean(0).tolist(),
          "share_step_profile": share.mean(0).tolist()}
    ml_ = per["move_layer"]
    g2["layer_marginal_move"] = np.median(ml_, 0).tolist()
    g2["layer_marginal_move_mean"] = ml_.mean(0).tolist()
    g2["layer_marginal_exploded_share"] = (ml_ > EXPLODE).mean(0).tolist()
    g2["layer_marginal_move_ci"] = [med_ci(ml_[:, l]) for l in range(L)]

    # --- G3 concentration + skip candidates
    flat = np.sort(move[ok].ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    cells80 = int(np.searchsorted(cum, 0.8) + 1)
    # skip candidates: cells whose blocking moves the trajectory by < SKIP_MOVE m on average
    # AND whose paired dminADE@1 across clips has no positive lower CI (no measurable harm)
    # harm-based: the cell is skippable if the paired dminADE@1 over clips has a median CI
    # whose upper bound is below +0.01 m (no measurable harm), whatever the move
    SKIP_HARM = 0.01
    dade_cells = per["ade_grid"] - per["ade_ref1"][:, None, None]  # (n, L, S)
    lo_ci = np.full((L, S), np.nan)
    hi_ci = np.full((L, S), np.nan)
    for l in range(L):
        for st in range(S):
            d = dade_cells[:, l, st]
            d = d[~np.isnan(d)]
            if len(d) >= 5:
                _, lo_ci[l, st], hi_ci[l, st] = med_ci(d)
    skippable = ok & (hi_ci < SKIP_HARM)
    harmful = ok & (lo_ci > 0)
    # per-clip: a cell is under that clip's SEED spread (context only, not the criterion)
    under_clip = per["move_grid"] < per["noise_floor"][:, None, None]
    g3 = {"cells_for_80pct_move": cells80, "n_cells": int(ok.sum()),
          "skip_harm_threshold": SKIP_HARM,
          "harmful_cells": int(harmful.sum()),
          "harmful_by_step": harmful.sum(0).tolist(),
          "skippable_cells": int(skippable.sum()),
          "skippable_share": float(skippable.sum() / max(ok.sum(), 1)),
          "skippable_by_step": skippable.sum(0).tolist(),
          "skippable_by_layer": skippable.sum(1).tolist(),
          "share_cells_under_seed_spread_per_clip": float(np.nanmean(under_clip)),
          "floor": floor,
          "top10_cells": [[int(l), int(s), float(move[l, s])] for l, s in
                          np.argwhere(ok)[np.argsort(-move[ok])[:10]]]}

    # --- G4 redundancy: sum of single cells vs the layer marginal
    cell_sum = np.nansum(move, 1)  # (L,)
    lay_marg = ml_.mean(0)
    g4 = {"ratio_cellsum_over_marginal": (cell_sum / np.maximum(lay_marg, 1e-9)).tolist(),
          "median_ratio": float(np.median(cell_sum / np.maximum(lay_marg, 1e-9)))}
    step_sum = np.nansum(move, 0)
    g4["ratio_cellsum_over_step_marginal"] = (step_sum / np.maximum(step_mean, 1e-9)).tolist()

    # --- head maps: expert-side reliance on each cache group (GQA: head h <- group h//2)
    mh = np.nanmedian(per["move_head"], 0)  # (L, H) median over clips
    mh_explode = np.nanmean(per["move_head"] > EXPLODE, 0)
    G = mh.shape[1] // 2
    R = mh.reshape(L, G, 2).sum(-1)  # (L, G) reliance per (layer, KV group)
    mass_h = stat["mass_cache"].mean(0)  # (L, H) step-mean
    share_h = stat["read_share"].mean(0)
    okh = ~np.isnan(mh)
    heads = {"spearman_mass_move_heads": [float(x) for x in spearmanr(mass_h[okh], mh[okh])],
             "spearman_share_move_heads": [float(x) for x in spearmanr(share_h[okh], mh[okh])],
             "top10_heads": [[int(l), int(h), float(mh[l, h])] for l, h in
                             np.argwhere(okh)[np.argsort(-mh[okh])[:10]]],
             "head_move_gini": float(1 - 2 * np.trapz(np.cumsum(np.sort(mh[okh])) /
                                                     mh[okh].sum(), dx=1 / okh.sum()))}
    # against what the VLM criterion protects, and what dual's pruning actually moved
    ext = {}
    shift = Rr = sr = None
    try:
        imp = np.load(REPO / "outputs" / "importance_v2" / "importance.npz")
        for key in ("traj_kv_k", "traj_kv_v", "coc_kv_k", "coc_kv_v"):
            ext[f"spearman_R_vs_{key}"] = [float(x) for x in spearmanr(R.ravel(), imp[key].ravel())]
    except FileNotFoundError:
        pass
    try:
        shift, nshift = 0, 0
        for sh in ("cachediff_v1_s0", "cachediff_v1_s100"):
            z = np.load(REPO / "outputs" / sh / "cachediff.npz")
            k = json.loads((REPO / "outputs" / sh / "metrics.json").read_text())["n_clips"]
            shift = shift + z["div_all_rel_v"] * k
            nshift += k
        shift = shift / nshift  # (L, G) relative V move under dual_u40_v2
        ext["spearman_R_vs_dual_cache_shift_v"] = [float(x) for x in
                                                   spearmanr(R.ravel(), shift.ravel())]
        # depth is a confound for both (shift grows with depth): partial out the layer mean
        Rr = R - R.mean(1, keepdims=True)
        sr = shift - shift.mean(1, keepdims=True)
        ext["spearman_R_vs_dual_cache_shift_v_within_layer"] = [float(x) for x in
                                                                spearmanr(Rr.ravel(), sr.ravel())]
    except FileNotFoundError:
        pass
    heads["external"] = ext
    heads["R_layer_group"] = R.tolist()

    # --- 3D sample: does move(l, h, s) factorise as head map x step profile?
    cells = json.loads((REPO / "outputs" / args.shards[0] / "config.json").read_text())["cells3d"]
    m3 = np.nanmedian(per["move_3d"], 0)  # (n_cells,)
    step_prof = step_mean / step_mean.sum()
    pred_fact = np.array([mh[l, h] * step_prof[st] * S for l, h, st in cells])
    pred_stat = np.array([stat["read_share"][st, l, h] for l, h, st in cells])
    pred_mass = np.array([stat["mass_cache"][st, l, h] for l, h, st in cells])
    ok3 = ~np.isnan(m3)
    fact = {"n_cells": int(ok3.sum()),
            "spearman_vs_headmap_x_stepprofile": [float(x) for x in
                                                  spearmanr(pred_fact[ok3], m3[ok3])],
            "spearman_vs_readout_share": [float(x) for x in spearmanr(pred_stat[ok3], m3[ok3])],
            "spearman_vs_mass": [float(x) for x in spearmanr(pred_mass[ok3], m3[ok3])]}

    # --- Stage C: which groups' ACTUAL dual shift hurts, vs which groups the expert relies on
    swapc = {}
    if "move_swap" in per and not np.all(np.isnan(per["move_swap"])):
        sw = np.nanmedian(per["move_swap"], 0)  # (L, G)
        sw_layer = np.nanmedian(per["move_swap_layer"], 0)  # (L,)
        oks = ~np.isnan(sw)
        swapc = {"swap_all_move": med_ci(per["move_swap_all"]),
                 "swap_all_move_mean": ci_mean(per["move_swap_all"]),
                 "swap_all_dade": med_ci(per["ade_swap_all"] - per["ade_ref1"]),
                 "swap_layer_dade": np.nanmedian(per["ade_swap_layer"] - per["ade_ref1"][:, None],
                                                 0).tolist(),
                 "swap_layer_move": sw_layer.tolist(),
                 "sum_group_swaps_over_layer_swap_median": float(np.nanmedian(
                     np.nansum(sw, 1) / np.maximum(sw_layer, 1e-9))),
                 "spearman_swap_vs_reliance_R": [float(x) for x in spearmanr(sw[oks], R[oks])],
                 "top10_swap_cells": [[int(l), int(g), float(sw[l, g])] for l, g in
                                      np.argwhere(oks)[np.argsort(-sw[oks])[:10]]]}
        if "spearman_R_vs_dual_cache_shift_v" in ext:
            swapc["spearman_swap_vs_shift_v"] = [float(x) for x in
                                                 spearmanr(sw[oks], shift[oks])]
            # within-layer versions (depth drives both shift and reliance)
            swr = sw - np.nanmean(sw, 1, keepdims=True)
            swapc["spearman_swap_vs_reliance_within_layer"] = [
                float(x) for x in spearmanr(swr[oks], Rr[oks])]
            swapc["spearman_swap_vs_shift_within_layer"] = [
                float(x) for x in spearmanr(swr[oks], sr[oks])]
            # product model: damage ~ reliance x shift
            prod = (R * shift)
            swapc["spearman_swap_vs_R_times_shift"] = [float(x) for x in
                                                       spearmanr(sw[oks], prod[oks])]
        np.savez(out / "maps_swap.npz", swap=sw, swap_layer=sw_layer)

    # --- dADE reading (is the move harmful?)
    d_all = per["ade_all"] - per["ade_ref"]
    try:
        p_all = float(wilcoxon(d_all).pvalue)
    except ValueError:
        p_all = float("nan")
    harm = {"all_block_dade_mean_ci": ci_mean(d_all), "all_block_dade_med_ci": med_ci(d_all),
            "all_block_wilcoxon_p": p_all,
            "grid_dade_map_median_of_medians": float(np.nanmedian(dade)),
            "corr_move_dade_cells": [float(x) for x in spearmanr(move[ok], dade[ok])]}

    res = {"n_clips": n, "g0": g0, "g1": g1, "g2": g2, "g3": g3, "g4": g4, "harm": harm,
           "heads": heads, "factorisation": fact, "swap": swapc,
           "span_mass_step_profile": {s: v.mean(0).tolist() for s, v in span_mass.items()},
           "span_mass_layer_profile": {s: v.mean(1).tolist() for s, v in span_mass.items()}}
    (out / "metrics_analysis.json").write_text(json.dumps(res, indent=1))
    np.savez(out / "maps.npz", move=move, move_mean=move_mean, explode=explode,
             move_end=move_end, dade=dade, mass=mass, share=share,
             entropy=ent, skippable=skippable, harmful=harmful, dade_lo_ci=lo_ci, dade_hi_ci=hi_ci,
             move_head=mh, move_head_explode=mh_explode, mass_head=mass_h,
             share_head=share_h, R_group=R,
             **{f"mass_{s}": v for s, v in span_mass.items()})

    swap_txt = "not run"
    if swapc:
        swap_txt = (f"all-swap move {swapc['swap_all_move'][0]:.3f} m (dADE "
                    f"{swapc['swap_all_dade'][0]:+.3f}); Spearman swap~reliance R "
                    f"{swapc['spearman_swap_vs_reliance_R'][0]:+.3f}")
        if "spearman_swap_vs_shift_v" in swapc:
            swap_txt += (f", swap~shift {swapc['spearman_swap_vs_shift_v'][0]:+.3f}, "
                         f"swap~R*shift {swapc['spearman_swap_vs_R_times_shift'][0]:+.3f}; "
                         f"within-layer: swap~R "
                         f"{swapc['spearman_swap_vs_reliance_within_layer'][0]:+.3f}, "
                         f"swap~shift {swapc['spearman_swap_vs_shift_within_layer'][0]:+.3f}")
    lines = [
        f"cache-use map -- {n} clips, {L} layers x {S} steps", "",
        (f"G0 {'PASS' if g0['pass'] else 'FAIL'}: steps seen {g0['steps_seen']}, none-block "
         f"max move {g0['none_block_move_max']:.1e}, all-block move median "
         f"{g0['all_block_move'][0]:.2f} [{g0['all_block_move'][1]:.2f},"
         f"{g0['all_block_move'][2]:.2f}] m (mean {g0['all_block_move_mean'][0]:.1f}, "
         f"{g0['all_block_exploded_share']:.0%} of clips diverge >{EXPLODE:.0f} m; dADE median "
         f"{g0['all_block_dade'][0]:+.2f}), seed spread {g0['noise_floor'][0]:.2f} m"),
        (f"G1 Spearman over {g3['n_cells']} cells: mass~move "
         f"{g1['spearman_mass_move'][0]:+.3f}, readout-share~move "
         f"{g1['spearman_share_move'][0]:+.3f}, mass~share {g1['spearman_mass_share'][0]:+.3f}"
         f"  -> mass is {'' if g1['mass_is_proxy'] else 'NOT '}a proxy (0.9 rule)"),
        "G2 step marginal move, median over clips (all layers blocked at one step):",
        "    " + " ".join(f"s{s}={v:.3f}" for s, v in enumerate(step_mean)),
        "    diverged share by step: " + " ".join(
            f"{v:.2f}" for v in g2["step_marginal_exploded_share"]),
        (f"    monotone decreasing {g2['monotone_decreasing']}, increasing "
         f"{g2['monotone_increasing']}, Spearman(step, move) "
         f"{g2['spearman_step_vs_move'][0]:+.3f}"),
        "    cache mass by step: " + " ".join(f"{v:.3f}" for v in g2["mass_step_profile"]),
        "    readout share by step: " + " ".join(f"{v:.3f}" for v in g2["share_step_profile"]),
        "    layer marginal move: " + " ".join(f"{v:.2f}" for v in g2["layer_marginal_move"]),
        (f"G3 {cells80}/{g3['n_cells']} cells carry 80% of the (median) grid move; harmful "
         f"(median paired dminADE CI > 0): {g3['harmful_cells']}/{g3['n_cells']}, by step "
         f"{g3['harmful_by_step']}; skippable (CI upper < +{SKIP_HARM}): "
         f"{g3['skippable_cells']}/{g3['n_cells']} = {g3['skippable_share']:.1%}, by step "
         f"{g3['skippable_by_step']}"),
        "    top cells (layer, step, move): "
        + ", ".join(f"({l},{s}) {v:.2f}" for l, s, v in g3["top10_cells"]),
        (f"G4 sum of single-cell moves / layer-marginal move: median "
         f"{g4['median_ratio']:.2f} (>1 = steps re-read the same information)"),
        (f"heads: Spearman mass~move {heads['spearman_mass_move_heads'][0]:+.3f}, "
         f"share~move {heads['spearman_share_move_heads'][0]:+.3f}; top heads (layer, head, "
         f"move): " + ", ".join(f"({l},{h}) {v:.2f}" for l, h, v in heads["top10_heads"][:6])),
        "external (R = per (layer, KV group) reliance): "
        + ", ".join(f"{k} {v[0]:+.3f} (p={v[1]:.2g})" for k, v in ext.items()),
        (f"3D sample ({fact['n_cells']} cells): Spearman vs head-map x step-profile "
         f"{fact['spearman_vs_headmap_x_stepprofile'][0]:+.3f}, vs readout share "
         f"{fact['spearman_vs_readout_share'][0]:+.3f}, vs mass "
         f"{fact['spearman_vs_mass'][0]:+.3f}"),
        "stage C (swap a group's cache for its dual-pruned version): " + swap_txt,
        (f"harm: all-block dADE {harm['all_block_dade_mean_ci'][0]:+.3f} "
         f"[{harm['all_block_dade_mean_ci'][1]:+.3f},{harm['all_block_dade_mean_ci'][2]:+.3f}] "
         f"p={harm['all_block_wilcoxon_p']:.2g}; Spearman(move, dADE) over cells "
         f"{harm['corr_move_dade_cells'][0]:+.3f}"),
    ]
    text = "\n".join(lines)
    print(text)
    (out / "cacheuse_summary.txt").write_text(text + "\n")

    # --- plots
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    heat(axes[0], mass, "attention mass on the cache")
    heat(axes[1], share, "readout share of the cache  ||sum_cache a v|| / total")
    heat(axes[2], move, "causal: median move when the cell is blocked (m)", cmap="magma")
    heat(axes[3], dade, "harm: median paired dminADE@1 when the cell is blocked (m)",
         cmap="magma")
    fig.tight_layout()
    fig.savefig(out / "plots" / "cacheuse_maps.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
    ax = axes[0]
    ci = np.array(g2["step_marginal_move_ci"])
    ax.errorbar(range(S), ci[:, 0], yerr=[ci[:, 0] - ci[:, 1], ci[:, 2] - ci[:, 0]], fmt="o-",
                color=C1, capsize=3, label="block all layers at one step")
    ax.plot(range(S), np.nanmean(move, 0) * L, "s--", color=C3, ms=4,
            label="sum of single cells at that step")
    ax.axhline(floor, color=MUTED, ls=":", label=f"seed-noise floor {floor:.2f} m")
    ax.set_xlabel("denoising step")
    ax.set_ylabel("move, median over clips (m)")
    ax.set_title("G2: cache reliance by step")
    ax.legend(fontsize=8)
    ax = axes[1]
    ci = np.array(g2["layer_marginal_move_ci"])
    ax.errorbar(range(L), ci[:, 0], yerr=[ci[:, 0] - ci[:, 1], ci[:, 2] - ci[:, 0]], fmt="o-",
                color=C1, ms=3, capsize=2, label="block one layer at all steps")
    ax.plot(range(L), cell_sum, "s--", color=C3, ms=3, label="sum of that layer's cells")
    ax.axhline(floor, color=MUTED, ls=":")
    ax.set_xlabel("expert layer")
    ax.set_title("cache reliance by layer (G4: sum vs marginal)")
    ax.legend(fontsize=8)
    ax = axes[2]
    for s, v in span_mass.items():
        ax.plot(range(S), v.mean(0), "o-", ms=3, label=s)
    ax.plot(range(S), stat["mass_own"].mean(-1).mean(-1), "k--", label="own tokens")
    ax.set_xlabel("denoising step")
    ax.set_ylabel("attention mass (layer/head mean)")
    ax.set_title("what the expert reads at each step")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "plots" / "cacheuse_profiles.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    heat(axes[0], mh.T, "causal: block one head's cache reads, all steps (m)",
         xlabel="expert layer", ylabel="expert head", cmap="magma")
    heat(axes[1], share_h.T, "readout share of the cache, per head (step mean)",
         xlabel="expert layer", ylabel="expert head")
    heat(axes[2], R.T, "reliance per (layer, KV group) = sum of its two heads",
         xlabel="expert layer", ylabel="KV group", cmap="magma")
    fig.tight_layout()
    fig.savefig(out / "plots" / "cacheuse_heads.png", dpi=150)
    plt.close(fig)

    if swapc:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
        heat(axes[0], sw.T, "Stage C: move when the group's cache is swapped for dual's (m)",
             xlabel="VLM layer", ylabel="KV group", cmap="magma")
        if "spearman_swap_vs_shift_v" in swapc:
            heat(axes[1], shift.T, "dual_u40_v2 cache shift  ||dV||/||V|| (cachediff_v1)",
                 xlabel="VLM layer", ylabel="KV group")
        axes[2].scatter(R[oks], sw[oks], s=12, color=C1, alpha=0.7)
        axes[2].set_xlabel("expert reliance R (block-move of the group's two heads, m)")
        axes[2].set_ylabel("swap move (m)")
        axes[2].set_title(f"swap damage vs reliance, Spearman "
                          f"{swapc['spearman_swap_vs_reliance_R'][0]:+.2f}")
        fig.tight_layout()
        fig.savefig(out / "plots" / "cacheuse_swap.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(mass[ok], move[ok], s=10, color=C1, alpha=0.7)
    axes[0].set_xlabel("attention mass on the cache")
    axes[0].set_ylabel("move when blocked (m)")
    axes[0].set_title(f"G1: mass vs move, Spearman {g1['spearman_mass_move'][0]:+.2f}")
    axes[1].scatter(share[ok], move[ok], s=10, color=C2, alpha=0.7)
    axes[1].set_xlabel("readout share of the cache")
    axes[1].set_title(f"readout share vs move, Spearman {g1['spearman_share_move'][0]:+.2f}")
    fig.tight_layout()
    fig.savefig(out / "plots" / "cacheuse_agreement.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
