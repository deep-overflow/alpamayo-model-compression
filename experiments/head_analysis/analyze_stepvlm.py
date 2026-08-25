"""Gates V0, V1, V4, V5 for the VLM step-axis decomposition.

  V0  integrity   sum-over-steps reproduces run_importance on the same clips. The threshold
                  is looser than the expert's (2e-2 median, not 1e-3): the VLM backward runs
                  through a far deeper bf16 graph, and the shipped path's single summed-seed
                  backward differs from ten per-step backwards by ~5e-3 median for that
                  reason alone. Measured before the plan was written, not tuned after.
  V1  go/no-go    does the fixed score actually move the u40_v2 selection, on `traj` alone
                  and inside max(rank traj, rank coc)? This decides whether any GPU is spent
                  on evaluation.
  V4  cheap trick does the one-backward seed reweighting approximate the exact znorm?
  V5  side        step-mass profile, depth profile, KV axis, binding share.

Usage:
  python experiments/head_analysis/analyze_stepvlm.py --stepvlm importance_stepvlm_v1 \
      --ref importance_v2_ada --out stepvlm_analysis
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

import mask_lib as ml  # noqa: E402
from make_stepvlm_importance import AGGS, aggregate, zscore_layers  # noqa: E402
from run_cocsafe import rank_norm  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RATIO = 0.3985632694  # the u40_v2 cell; not 0.40 (run_grid.allocations matched budget)
EPS = 1e-30

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})


def layer_spearman(a, b):
    """Mean per-layer Spearman, skipping layers with no variation.

    Layer 35 carries exactly zero trajectory importance (P1 measured it, and every shipped
    config drops it wholesale), so its rows are constant and spearmanr returns NaN there.
    Averaging that in would poison the gate.
    """
    rs = []
    for i in range(a.shape[0]):
        if np.ptp(a[i]) == 0 or np.ptp(b[i]) == 0:
            continue
        r = spearmanr(a[i], b[i])[0]
        if np.isfinite(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else float("nan"), len(rs)


def keep_overlap(a, b, ratio=RATIO):
    layers = list(range(a.shape[0]))
    ka = ml.select_mask(a, ratio, layers) == 1
    kb = ml.select_mask(b, ratio, layers) == 1
    return float((ka & kb).sum() / max(ka.sum(), 1))


def gate_v0(scores, ref, lines):
    lines.append("V0 integrity -- does sum-over-steps reproduce run_importance?")
    ok, res = True, {}
    for unit in ("q", "mlp"):
        got, want = scores["sum"][unit], ref[f"traj_vlm_{unit}"]
        rel = np.abs(got - want) / (np.abs(want) + EPS)
        rho, n_l = layer_spearman(got, want)
        ov = keep_overlap(got, want)
        good = np.median(rel) < 2e-2 and rho > 0.99 and ov > 0.98
        ok = ok and good
        res[unit] = {"median_rel": float(np.median(rel)),
                     "p99_rel": float(np.percentile(rel, 99)),
                     "layer_rho": rho, "n_layers_scored": n_l,
                     "kept_overlap": ov, "pass": bool(good)}
        lines.append(f"  vlm_{unit:3s} median rel {np.median(rel):.3e}  "
                     f"p99 {np.percentile(rel, 99):.3e}  rho {rho:.6f} ({n_l} layers)  "
                     f"kept-overlap {ov:.4f}   {'PASS' if good else 'FAIL'}")
    lines.append(f"  -> V0 {'PASS' if ok else 'FAIL'}\n")
    res["pass"] = bool(ok)
    return res


def gate_v1(scores, ref, lines):
    """Does the fix move the selection -- alone, and through the max-rank guardrail?"""
    lines.append("V1 go/no-go -- does the fixed score change what gets pruned?")
    res = {}
    for unit in ("q", "mlp"):
        coc = ref[f"coc_vlm_{unit}"]
        entry = {}
        base = scores["sum"][unit]
        # how often the traj half is the binding one bounds how much fixing it can matter
        binds = float((rank_norm(base) >= rank_norm(coc)).mean())
        entry["traj_binds_in_dual"] = binds
        for agg, sc_ in scores.items():
            if agg == "sum":
                continue
            ov_traj = keep_overlap(base, sc_[unit])
            d_base = np.maximum(rank_norm(base), rank_norm(coc))
            d_new = np.maximum(rank_norm(sc_[unit]), rank_norm(coc))
            ov_dual = keep_overlap(d_base, d_new)
            entry[agg] = {"overlap_traj": ov_traj, "overlap_dual": ov_dual}
            lines.append(f"  vlm_{unit:3s} {agg:6s}: kept-overlap vs sum -- "
                         f"traj alone {ov_traj:.4f}, inside dual {ov_dual:.4f}")
        lines.append(f"  vlm_{unit:3s} traj half binds in max() for {binds:.3f} of units")
        res[unit] = entry
    # the gate reads the primary arm on the sensitive (guardrail-free) criterion
    prim = min(res[u]["znorm"]["overlap_traj"] for u in ("q", "mlp"))
    go = prim < 0.97
    lines.append(f"  -> V1 {'GO' if go else 'STOP'} (znorm vs sum on traj alone: "
                 f"{prim:.4f}, gate wants < 0.97)")
    if not go:
        lines.append("     the fix does not move the selection; evaluating it would spend "
                     "GPU to measure nothing")
    lines.append("")
    res["primary_overlap_traj"] = prim
    res["go"] = bool(go)
    return res


def cancellation(z, scores, lines):
    """How much of the gradient does the shipped |sum_s| erase, and does removing it matter?

    The VLM analogue of the expert's G1. `sumabs` is the only aggregation that removes the
    sign cancellation without touching the step-mass ordering -- which matters here because
    flattening that ordering is exactly what made znorm regress on this tower.
    """
    lines.append("sign cancellation -- |sum_s g| vs sum_s |g| (predicts what sumabs can do)")
    res = {}
    for unit in ("q", "mlp"):
        shipped = z[f"{unit}_shipped"].astype(np.float64)
        sabs = z[f"{unit}_abs_step"].astype(np.float64).sum(0)
        c = shipped / (sabs + EPS)
        ov = keep_overlap(shipped, sabs)
        res[unit] = {"median_C": float(np.median(c)),
                     "p10_C": float(np.percentile(c, 10)),
                     "p90_C": float(np.percentile(c, 90)),
                     "overlap_sum_sumabs": ov}
        lines.append(f"  vlm_{unit:3s} C median {np.median(c):.3f} "
                     f"[p10 {np.percentile(c, 10):.3f}, p90 {np.percentile(c, 90):.3f}]"
                     f"   kept-overlap(sum, sumabs) {ov:.4f}")
    worst = min(res[u]["overlap_sum_sumabs"] for u in ("q", "mlp"))
    lines.append(f"  -> sumabs moves at most {(1 - worst) * 100:.1f}% of the picks; "
                 f"{'worth evaluating' if worst < 0.97 else 'likely a no-op'}")
    lines.append("")
    return res


def gate_v4(scores, lines):
    lines.append("V4 cheap approximation -- one reweighted backward vs ten")
    res = {}
    for unit in ("q", "mlp"):
        ov = keep_overlap(scores["znorm"][unit], scores["seedz"][unit])
        rho, _ = layer_spearman(scores["znorm"][unit], scores["seedz"][unit])
        ov_sum = keep_overlap(scores["sum"][unit], scores["seedz"][unit])
        res[unit] = {"overlap_vs_znorm": ov, "rho_vs_znorm": rho,
                     "overlap_vs_sum": ov_sum}
        lines.append(f"  vlm_{unit:3s} seedz vs znorm: kept-overlap {ov:.4f}, rho {rho:.4f}"
                     f"   | seedz vs sum: {ov_sum:.4f}")
    worst = min(res[u]["overlap_vs_znorm"] for u in ("q", "mlp"))
    verdict = "ACCEPT" if worst > 0.98 else ("REJECT" if worst < 0.95 else "MARGINAL")
    lines.append(f"  -> V4 {verdict} (worst overlap {worst:.4f})")
    if verdict != "ACCEPT":
        lines.append("     expected in hindsight: linearity buys a weighted SIGNED sum in "
                     "one backward, but znorm takes |g_s| per step first, and that "
                     "nonlinearity cannot be folded into a seed. seedz is a reweighted "
                     "`sum`, not a cheap `znorm`.")
    lines.append("")
    res["verdict"] = verdict
    return res


def gate_v5(z, scores, ref, lines):
    lines.append("V5 side records")
    res = {}
    for unit in ("q", "mlp"):
        a = z[f"{unit}_abs_step"].astype(np.float64)
        mass = a.sum((1, 2))
        prof = a.sum(2)
        com = (prof * np.arange(prof.shape[1])[None]).sum(1) / prof.sum(1)
        res[unit] = {"mass_by_step": [float(x) for x in mass / mass.max()],
                     "mass_max_over_min": float(mass.max() / mass.min()),
                     "depth_com_by_step": [float(x) for x in com]}
        lines.append(f"  vlm_{unit:3s} mass by step: " +
                     " ".join(f"{x:.3f}" for x in mass / mass.max()))
        lines.append(f"           max/min {mass.max() / mass.min():.1f}, depth c.o.m. "
                     f"{com[0]:.1f} -> {com[-1]:.1f}")
    kv_sum = z["kv_k_shipped"] + z["kv_v_shipped"]
    kv_z = sum(np.mean([zscore_layers(a) for a in z[f"kv_{c}_abs_step"].astype(np.float64)],
                       axis=0) for c in ("k", "v"))
    # KV group choice is 1-of-8 per layer, so report the drop pick rather than an overlap
    drop_sum = np.argmin(kv_sum, axis=1)
    drop_z = np.argmin(kv_z, axis=1)
    agree = float((drop_sum == drop_z).mean())
    res["kv"] = {"kv1_drop_agreement": agree,
                 "drop_sum": drop_sum.tolist(), "drop_znorm": drop_z.tolist()}
    lines.append(f"  KV-1 drop pick agrees on {agree:.3f} of layers (sum vs znorm); "
                 f"u40_v2 does not use KV, this is for cocsafe / j_traj / integrated_mag")
    lines.append("")
    return res


def plot_all(out_dir, z, scores, v1):
    pd_ = out_dir / "plots"
    pd_.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, unit in zip(axes, ("q", "mlp")):
        m = z[f"{unit}_abs_step"].astype(np.float64).sum((1, 2))
        ax.plot(m / m.max(), "o-", color=C1)
        ax.set_title(f"VLM {unit}: importance mass by denoising step")
        ax.set_xlabel("denoising step")
        ax.set_ylabel("normalised mass")
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(pd_ / "vlm_step_mass.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, unit in zip(axes, ("q", "mlp")):
        prof = z[f"{unit}_abs_step"].astype(np.float64).sum(2)
        prof = prof / (prof.max(axis=1, keepdims=True) + EPS)
        im = ax.imshow(prof.T, aspect="auto", cmap="Oranges", interpolation="nearest")
        ax.set_title(f"VLM {unit}: depth profile per step (row-normalised)")
        ax.set_xlabel("denoising step")
        ax.set_ylabel("layer")
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.tight_layout()
    fig.savefig(pd_ / "vlm_depth_by_step.png", dpi=150)
    plt.close(fig)

    aggs = [a for a in scores if a != "sum"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(aggs))
    w = 0.36
    for off, unit, c in ((-w / 2, "q", C1), (w / 2, "mlp", C2)):
        ax.bar(x + off, [v1[unit][a]["overlap_traj"] for a in aggs], w,
               color=c, label=f"{unit}, traj alone")
        ax.plot(x + off, [v1[unit][a]["overlap_dual"] for a in aggs], "k_",
                markersize=14, markeredgewidth=2)
    ax.axhline(0.97, color=MUTED, ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(aggs)
    ax.set_ylabel("kept-overlap vs shipped `sum`")
    ax.set_title("how much each fix moves the selection (dash = V1 gate; tick = inside dual)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(pd_ / "vlm_selection_shift.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stepvlm", default="importance_stepvlm_v1")
    ap.add_argument("--ref", default="importance_v2_ada")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = REPO / "outputs" / args.stepvlm
    z = dict(np.load(src / "step_importance_vlm.npz"))
    pc = dict(np.load(src / "step_importance_vlm_perclip.npz"))
    ref = dict(np.load(REPO / "outputs" / args.ref / "importance.npz"))
    cfg = json.loads((src / "config.json").read_text())

    scores = {agg: {u: aggregate(z, pc, agg, u) for u in ("q", "mlp")} for agg in AGGS}

    out_dir = REPO / "outputs" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"VLM step-axis decomposition -- {args.out}",
        (f"  step run {args.stepvlm} ({cfg['num_clips']} clips, {cfg['gpu']}), "
         f"reference {args.ref}"),
        f"  u40_v2 ratio {RATIO}",
        "",
    ]
    res = {"args": vars(args), "n_clips": cfg["num_clips"]}
    res["V0"] = gate_v0(scores, ref, lines)
    res["V1"] = gate_v1(scores, ref, lines)
    res["cancellation"] = cancellation(z, scores, lines)
    res["V4"] = gate_v4(scores, lines)
    res["V5"] = gate_v5(z, scores, ref, lines)

    plot_all(out_dir, z, scores, res["V1"])
    (out_dir / "summary.txt").write_text("\n".join(lines))
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    print("\n".join(lines))
    print("saved ->", out_dir)


if __name__ == "__main__":
    main()
