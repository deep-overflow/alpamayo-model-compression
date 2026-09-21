"""Stage 0 and gates A1-A2 of plans/2026-09-21_importance-causal-validation.md.

  S0  what each shipped criterion kept, in the dense model's token-resolved scores
      (gradanat_v1 x the three slim_meta.json): CPU bookkeeping, no run needed.
  A1  the anatomy re-measured ON the dual-pruned model keeps the dense structure: ratio
      change point at layer 23 on both axes, layer-profile correlation with dense >= 0.98 for
      both scores, MLP hand-over layers within +-1 of dense, and ceiling-corrected agreement
      of the two losses at VISION tokens among kept Q heads, layers 22-34, >= 0.70.
  A2  on the dense model's text (same tokens, same noise for every arm) the FM loss rises most
      under the CoC-only criterion and the NLL most under the trajectory-only one
      (paired one-sided Wilcoxon over clips, p < 0.01 each).

The arms were run with run_gradient_anatomy.py --mask; the dense reference for A2 is the
--probes run, which records both losses on the same clips and seeds.

Usage:
  python experiments/head_analysis/analyze_pruned_anatomy.py --out gradanat_pruned_v1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))

from analyze_gradient_anatomy import (  # noqa: E402
    C1, C2, C3, C4, COC, HIST, LAST, LATE, MUTED, PT, VIS, corrected, plt, split_half,
)

REPO = Path(__file__).resolve().parents[2]
ARMS = ("dual", "traj", "coc")
ARM_COL = {"dense": "black", "dual": C2, "traj": C1, "coc": C3}


def step_fit(y):
    """Two-level piecewise-constant least-squares fit (fig_why_differs.step_fit)."""
    best = None
    for tau in range(3, len(y) - 3):
        a, b = y[:tau].mean(), y[tau:].mean()
        sse = ((y[:tau] - a) ** 2).sum() + ((y[tau:] - b) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, tau, a, b)
    sse, tau, a, b = best
    return int(tau), float(np.exp(a - b)), float(1 - sse / ((y - y.mean()) ** 2).sum())


def kept_masks(arm):
    meta = json.loads((REPO / "outputs" / f"slim_{arm}_u40_v2" / "slim_meta.json").read_text())
    q, mlp = np.zeros((36, 32), bool), np.zeros((36, 12288), bool)
    for li, e in enumerate(meta["vlm"]):
        q[li, e["q"]] = True
        mlp[li, e["mlp"]] = True
    return {"q": q, "mlp": mlp}


def additive_vision_share(arr):
    """Per-clip signed typed grads (N, 5, 36, U) -> (36,) share of the layer's shipped score
    that comes from what its units do at vision positions (signed shares add to one)."""
    part = np.zeros((5, 36))
    for i in range(len(arr)):  # clip by clip: the MLP array is 885 MB
        g = arr[i].astype(np.float64)  # (5, 36, U)
        part += (g * np.sign(g.sum(0, keepdims=True))).sum(-1)
    tot = part.sum(0)
    return part[VIS] / np.where(tot > 0, tot, np.nan)


def handover(share):
    """First layer from which the vision share stays below one half through layer 32."""
    return next((li for li in range(6, 33) if (share[li:33] < 0.5).all()), None)


def gather_kept(x, keep):
    """x (N, L, U), keep (L, U) bool with the same count per layer -> (N, L, n_keep)."""
    idx = np.stack([np.flatnonzero(keep[li]) for li in range(keep.shape[0])])  # (L, n_keep)
    return np.take_along_axis(x, idx[None], axis=2)


def stage0(z, lines, out):
    lines += ["S0  share of the dense score mass that sits on kept units",
              f"    {'':4s} {'':36s} {'dual':>7s} {'traj':>7s} {'coc':>7s}"]
    keep = {a: kept_masks(a) for a in ARMS}
    res = {}
    for ax in ("q", "mlp"):
        scores = {"I_traj at vision+history tokens": z[f"fm_type_{ax}"][VIS] + z[f"fm_type_{ax}"][HIST],
                  "I_traj at text-side tokens": z[f"fm_type_{ax}"][PT] + z[f"fm_type_{ax}"][COC],
                  "I_CoC at CoC tokens": z[f"ce_{ax}"][COC],
                  "I_CoC at prompt tokens": z[f"ce_{ax}"][PT]}
        for band, lo, hi in (("L22-34", 22, 35), ("L6-21", 6, 22)):
            for name, s in scores.items():
                v = {a: float((s[lo:hi] * keep[a][ax][lo:hi]).sum() / s[lo:hi].sum()) for a in ARMS}
                res[f"{ax}|{band}|{name}"] = v
                lines.append(f"    {ax:4s} {band:7s} {name:28s} {v['dual']:7.3f} {v['traj']:7.3f} {v['coc']:7.3f}")
        t, c = res[f"{ax}|L22-34|I_traj at vision+history tokens"], res[f"{ax}|L22-34|I_CoC at CoC tokens"]
        res[f"{ax}|gate"] = {
            "coc_keeps_least_traj_at_vision": bool(t["coc"] < min(t["dual"], t["traj"])),
            "traj_keeps_least_coc_at_coc": bool(c["traj"] < min(c["dual"], c["coc"])),
            "dual_gap_traj_side_pp": 100 * (max(t["traj"], t["coc"]) - t["dual"]),
            "dual_gap_coc_side_pp": 100 * (max(c["traj"], c["coc"]) - c["dual"])}
        g = res[f"{ax}|gate"]
        lines.append(f"    {ax}: ordering {'PASS' if g['coc_keeps_least_traj_at_vision'] and g['traj_keeps_least_coc_at_coc'] else 'FAIL'}; "
                     f"dual within 5 pp of the better single criterion: trajectory side {g['dual_gap_traj_side_pp']:.1f} pp "
                     f"({'PASS' if g['dual_gap_traj_side_pp'] <= 5 else 'FAIL'}), CoC side {g['dual_gap_coc_side_pp']:.1f} pp "
                     f"({'PASS' if g['dual_gap_coc_side_pp'] <= 5 else 'FAIL'})")
    lines.append("")
    out["S0"] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", default="gradanat_v1")
    ap.add_argument("--dense-losses", default="gradanat_probes_v1",
                    help="dense run that recorded both the NLL and the FM loss per clip")
    ap.add_argument("--arm-exp", default="gradanat_{arm}_u40")
    ap.add_argument("--out", default="gradanat_pruned_v1")
    ap.add_argument("--n-split", type=int, default=50)
    args = ap.parse_args()

    outs = REPO / "outputs"
    out_dir = outs / args.out
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    runs = {"dense": outs / args.dense, **{a: outs / args.arm_exp.format(arm=a) for a in ARMS}}
    z = {k: dict(np.load(d / "anatomy.npz")) for k, d in runs.items()}
    out, lines = {}, [f"pruned-model anatomy -- dense {args.dense}, arms {args.arm_exp}", ""]
    stage0(z["dense"], lines, out)

    # ------------------------------------------------------------------ A1
    lines.append("A1  structure of the two scores, measured on each model (kept units only)")
    a1 = {}
    L = np.arange(LAST)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax_i, ax in enumerate(("q", "mlp")):
        for k in runs:
            t, c = z[k][f"fm_full_{ax}"].sum(1)[:LAST], z[k][f"ce_full_{ax}"].sum(1)[:LAST]  # (35,)
            td, cd = z["dense"][f"fm_full_{ax}"].sum(1)[:LAST], z["dense"][f"ce_full_{ax}"].sum(1)[:LAST]
            tau, step, r2 = step_fit(np.log(t / c))
            a1[f"{k}|{ax}"] = {"change_point": tau, "step": step, "r2": r2,
                               "profile_corr_traj": float(np.corrcoef(t, td)[0, 1]),
                               "profile_corr_coc": float(np.corrcoef(c, cd)[0, 1]),
                               # not gated: the same without the first two layers, where the surviving
                               # units of a pruned model take on several times the dense importance
                               "profile_corr_traj_L2_34": float(np.corrcoef(t[2:], td[2:])[0, 1]),
                               "profile_corr_coc_L2_34": float(np.corrcoef(c[2:], cd[2:])[0, 1]),
                               "layer_sum_over_dense_L0_1": [float((t[:2] / td[:2]).mean()), float((c[:2] / cd[:2]).mean())],
                               "layer_sum_over_dense_L6_34": [float((t[6:] / td[6:]).mean()), float((c[6:] / cd[6:]).mean())],
                               "peak_traj": int(t.argmax()), "peak_coc": int(c.argmax())}
            r = a1[f"{k}|{ax}"]
            lines.append(f"    {k:6s} {ax:3s} ratio change point {tau} (step {step:.2f}x, R2 {r2:.2f}) | profile corr with "
                         f"dense: I_traj {r['profile_corr_traj']:.3f} I_CoC {r['profile_corr_coc']:.3f} (layers 2-34: "
                         f"{r['profile_corr_traj_L2_34']:.3f} / {r['profile_corr_coc_L2_34']:.3f}) | layer sum / dense, "
                         f"L0-1: {r['layer_sum_over_dense_L0_1'][0]:.1f}x / {r['layer_sum_over_dense_L0_1'][1]:.1f}x, L6-34: "
                         f"{r['layer_sum_over_dense_L6_34'][0]:.1f}x / {r['layer_sum_over_dense_L6_34'][1]:.1f}x")
            axs[ax_i].plot(L, t / c, color=ARM_COL[k], lw=1.6 if k == "dense" else 1.1, label=k)
        axs[ax_i].set_yscale("log")
        axs[ax_i].set_title(f"I_traj / I_CoC by depth, {'Q heads' if ax == 'q' else 'MLP channels'}")
        axs[ax_i].set_xlabel("VLM layer")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_dir / "plots" / "ratio_by_depth.png", dpi=150)
    plt.close(fig)

    # hand-over layers from the additive token split (per-clip signed arrays)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax_i, (ax, fname, keys) in enumerate((("q", "anatomy_perclip_q.npz", ("ce", "fm_full")),
                                              ("mlp", "anatomy_perclip_mlp.npz", ("ce_type", "fm_type")))):
        for k, d in runs.items():
            with np.load(d / fname) as p:
                sh_c, sh_t = additive_vision_share(p[keys[0]]), additive_vision_share(p[keys[1]])
            a1[f"{k}|{ax}"].update({"handover_coc": handover(sh_c), "handover_traj": handover(sh_t),
                                    "vision_share_traj": np.nan_to_num(sh_t).tolist(),
                                    "vision_share_coc": np.nan_to_num(sh_c).tolist()})
            lines.append(f"    {k:6s} {ax:3s} hand-over layer (vision share stays < 0.5): I_CoC "
                         f"{a1[f'{k}|{ax}']['handover_coc']}  I_traj {a1[f'{k}|{ax}']['handover_traj']}")
            axs[ax_i].plot(L, sh_t[:LAST], color=ARM_COL[k], lw=1.4, label=f"{k} I_traj")
            axs[ax_i].plot(np.arange(36), sh_c, color=ARM_COL[k], lw=1.0, ls=(0, (4, 2)), label=f"{k} I_CoC")
        axs[ax_i].axhline(0.5, color=MUTED, lw=0.7)
        axs[ax_i].set_title(f"share of importance from vision tokens, {'Q heads' if ax == 'q' else 'MLP channels'}")
        axs[ax_i].set_xlabel("VLM layer")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out_dir / "plots" / "token_handover.png", dpi=150)
    plt.close(fig)

    # same-token agreement among kept Q heads
    lines.append("    agreement of the two losses among kept Q heads, split-half ceiling corrected (late = 22-34)")
    for k, d in runs.items():
        with np.load(d / "anatomy_perclip_q.npz") as p:
            ce, fm = p["ce"], p["fm_full"]  # (N, 5, 36, 32) signed
        keep = np.ones((36, 32), bool) if k == "dense" else kept_masks(k)["q"]
        for name, t, c in (("pooled", np.abs(fm.sum(1)), np.abs(ce.sum(1))),
                           ("vision", np.abs(fm[:, VIS]), np.abs(ce[:, VIS]))):
            t, c = gather_kept(t, keep)[:, :LAST], gather_kept(c, keep)[:, :LAST]
            st, sc_, cr = split_half(t, c, args.n_split, rng)
            r = corrected(st, sc_, cr, LATE)
            a1[f"{k}|q"][f"late_agreement_{name}"] = r
            lines.append(f"      {k:6s} {name:7s} self {r['self_traj']:.2f}/{r['self_coc']:.2f} cross {r['cross']:.2f} "
                         f"corrected {r['corrected']:.3f} ({r['layers_used']}/13 layers)")
    dq, dm = a1["dual|q"], a1["dual|mlp"]
    ref = a1["dense|mlp"]
    a1["gate"] = {
        "change_point_23": bool(dq["change_point"] == 23 and dm["change_point"] == 23),
        "profile_corr_ge_0.98": bool(min(dq["profile_corr_traj"], dq["profile_corr_coc"],
                                         dm["profile_corr_traj"], dm["profile_corr_coc"]) >= 0.98),
        "mlp_handover_within_1": bool(None not in (dm["handover_coc"], dm["handover_traj"])
                                      and abs(dm["handover_coc"] - ref["handover_coc"]) <= 1
                                      and abs(dm["handover_traj"] - ref["handover_traj"]) <= 1),
        "vision_agreement_ge_0.70": bool(dq["late_agreement_vision"]["corrected"] >= 0.70)}
    a1["gate"]["pass"] = bool(all(a1["gate"].values()))
    g = a1["gate"]
    lines += [f"    A1 (dual): change point 23 on both axes {'PASS' if g['change_point_23'] else 'FAIL'} "
              f"({dq['change_point']}/{dm['change_point']}) | profile corr >= 0.98 "
              f"{'PASS' if g['profile_corr_ge_0.98'] else 'FAIL'} | MLP hand-over within +-1 of dense "
              f"({ref['handover_coc']}/{ref['handover_traj']} -> {dm['handover_coc']}/{dm['handover_traj']}) "
              f"{'PASS' if g['mlp_handover_within_1'] else 'FAIL'} | vision-token agreement "
              f"{dq['late_agreement_vision']['corrected']:.3f} >= 0.70 "
              f"{'PASS' if g['vision_agreement_ge_0.70'] else 'FAIL'}  ->  {'PASS' if g['pass'] else 'FAIL'}", ""]
    out["A1"] = a1

    # cache-band split of I_traj (descriptive): is cache 22-24 still the main port?
    lines.append("    share of E|G| entering through each cache band (0-7, 8-15, 16-21, 22-24, 25-29, 30-35)")
    out["bands"] = {}
    for k in runs:
        for ax in ("q", "mlp"):
            b = z[k][f"fm_bandtot_{ax}"].sum((1, 2))
            out["bands"][f"{k}|{ax}"] = (b / b.sum()).tolist()
            lines.append(f"      {k:6s} {ax:3s} " + "  ".join(f"{v:.3f}" for v in b / b.sum()))
    lines.append("")

    # ------------------------------------------------------------------ A2
    met = {k: json.loads((d / "metrics.json").read_text())["per_clip"] for k, d in runs.items()}
    met["dense"] = json.loads((outs / args.dense_losses / "metrics.json").read_text())["per_clip"]
    ids = [r["clip_id"] for r in met["dense"]]
    by = {k: {r["clip_id"]: r for r in v} for k, v in met.items()}
    same = [c for c in ids if all(c in by[a] and by[a][c]["coc_len"] == by["dense"][c]["coc_len"]
                                  and by[a][c]["prompt_len"] == by["dense"][c]["prompt_len"] for a in ARMS)]
    ref_fm = {r["clip_id"]: r["fm_loss"] for r in json.loads((runs["dense"] / "metrics.json").read_text())["per_clip"]}
    drift = max(abs(by["dense"][c]["fm_loss"] - ref_fm[c]) for c in same if c in ref_fm)
    fm = {k: np.array([by[k][c]["fm_loss"] for c in same]) for k in by}
    nll = {k: np.array([by[k][c]["nll"] for c in same]) for k in by}
    a2 = {"n_clips": len(same), "n_text_mismatch": len(ids) - len(same),
          "dense_fm_loss_drift_vs_" + args.dense: float(drift),
          "fm_mean": {k: float(v.mean()) for k, v in fm.items()},
          "nll_mean": {k: float(v.mean()) for k, v in nll.items()},
          "fm_median_delta": {a: float(np.median(fm[a] - fm["dense"])) for a in ARMS},
          "nll_median_delta": {a: float(np.median(nll[a] - nll["dense"])) for a in ARMS}, "tests": {}}
    lines += [f"A2  losses on the dense model's text, {len(same)} clips with identical text in every run "
              f"({len(ids) - len(same)} dropped); dense FM loss vs {args.dense}: max |diff| {drift:.1e}",
              f"    {'':6s} {'FM loss':>9s} {'dFM med':>9s} {'NLL':>9s} {'dNLL med':>9s}"]
    for k in ("dense",) + ARMS:
        lines.append(f"    {k:6s} {fm[k].mean():9.4f} {a2['fm_median_delta'].get(k, 0):+9.4f} "
                     f"{nll[k].mean():9.4f} {a2['nll_median_delta'].get(k, 0):+9.4f}")
    for label, x, hi, lo in (("FM", fm, "coc", "dual"), ("FM", fm, "coc", "traj"),
                             ("NLL", nll, "traj", "dual"), ("NLL", nll, "traj", "coc")):
        p = float(wilcoxon(x[hi], x[lo], alternative="greater").pvalue)
        a2["tests"][f"{label}:{hi}>{lo}"] = {"p": p, "median_diff": float(np.median(x[hi] - x[lo])),
                                             "pass": bool(p < 0.01)}
        lines.append(f"    {label:3s} {hi} > {lo}: median paired diff {np.median(x[hi] - x[lo]):+.4f}, "
                     f"one-sided Wilcoxon p = {p:.2e} -> {'PASS' if p < 0.01 else 'FAIL'}")
    a2["pass"] = bool(all(t["pass"] for t in a2["tests"].values()))
    lines += [f"    A2 -> {'PASS' if a2['pass'] else 'FAIL'}"]
    # not pre-registered: does the dual criterion beat each specialist on the specialist's OWN channel?
    a2["own_channel"] = {}
    for label, x, spec in (("FM", fm, "traj"), ("NLL", nll, "coc")):
        p = float(wilcoxon(x[spec], x["dual"], alternative="greater").pvalue)
        a2["own_channel"][f"{label}:{spec}>dual"] = {"p": p, "median_diff": float(np.median(x[spec] - x["dual"]))}
        lines.append(f"    descriptive, {label:3s} {spec} > dual (the specialist on its own channel): median paired diff "
                     f"{np.median(x[spec] - x['dual']):+.4f}, one-sided Wilcoxon p = {p:.2e}")
    lines.append("")
    out["A2"] = a2

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax_, x, name in ((axs[0], fm, "FM loss"), (axs[1], nll, "CoC NLL")):
        d = [x[a] - x["dense"] for a in ARMS]
        ax_.boxplot(d, showfliers=False, medianprops={"color": C4})
        ax_.set_xticks([1, 2, 3], ARMS)
        ax_.axhline(0, color=MUTED, lw=0.7)
        ax_.set_title(f"{name}: arm - dense, per clip (dense text)")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "matched_losses.png", dpi=150)
    plt.close(fig)

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    (out_dir / "config.json").write_text(json.dumps({
        "plan": "plans/2026-09-21_importance-causal-validation.md (Stage 0, part A)",
        "dense": args.dense, "dense_losses": args.dense_losses,
        "arms": {a: args.arm_exp.format(arm=a) for a in ARMS}, "n_split": args.n_split}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
