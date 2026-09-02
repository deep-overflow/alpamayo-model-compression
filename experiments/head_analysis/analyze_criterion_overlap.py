"""How much of the kept set do two VLM pruning criteria share, at a matched budget?

plans/2026-08-31_znorm11-criterion.md. Runs on the stored importance files only -- no
model, no GPU -- so a criterion can be judged (or a config's masks verified) before
spending a build.

Criteria (all evaluated per layer, then a uniform u40 budget is applied with
mask_lib.select_mask_ratios, i.e. the same within-layer argsort make_slim uses):

  dual        max(rank_norm I_traj, rank_norm I_CoC)   -- the shipped criterion
  dualfix     same, but a layer whose half is CONSTANT contributes nothing to the max
  znorm11     mean of eleven within-layer z-scores: CoC NLL + the ten flow-matching
              step losses (per-step VLM gradients from run_step_importance_vlm)
  znorm10     the ten step losses only -- isolates what adding CoC does
  znorm5050   0.5 z(CoC) + 0.5 mean_s z(step_s) -- CoC weighted as one half, like dual
  traj / coc  the single-criterion halves

Reports, per pair: kept-set agreement |A and B| / |A| per layer (both keep the same
count, so this is symmetric), Jaccard, the within-layer Spearman of the scores, and the
parameter churn the disagreement represents. Layers whose trajectory importance is
structurally zero (the last one) are called out separately, because rank_norm of a
constant row is the INDEX ORDER and that is what dual ranks there.

Usage:
  .venv/bin/python experiments/head_analysis/analyze_criterion_overlap.py \
      --ref dual --arms znorm11 znorm10 znorm5050 dualfix traj coc --out criterion_overlap
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
from run_cocsafe import rank_norm  # noqa: E402

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": "#E8E6DC",
    "font.size": 9, "axes.grid": True, "axes.axisbelow": True,
})
REPO = Path(__file__).resolve().parents[2]
U40 = 0.3985632694          # the matched uniform ratio the u40 family uses
P_Q, P_M = 2 * 4096 * 128, 3 * 4096      # params per kept Q head / MLP channel


def zscore(x):
    """Within-layer z-score of an (L, U) array."""
    return (x - x.mean(1, keepdims=True)) / np.maximum(x.std(1, keepdims=True), 1e-12)


def rank_skip_constant(x):
    """rank_norm, with constant layers excluded from a max() combination.

    rank_norm(zeros) is the index order, so a structurally-zero layer (the VLM's last,
    whose o_proj / down_proj never reach the KV cache) would otherwise make
    max(rank traj, rank coc) prefer high-numbered units there.
    """
    r = rank_norm(x)
    r[np.ptp(x, axis=1) == 0] = -np.inf
    return r


def build_scores(imp, step, arms):
    """{name: (q_scores, mlp_scores)} for every requested criterion."""
    out = {}
    need_step = any(a.startswith("znorm") for a in arms)
    if need_step and step is None:
        raise SystemExit("znorm* criteria need --stepvlm (step_importance_vlm.npz)")
    if need_step:
        zq = np.stack([zscore(step["q_abs_step"][i]) for i in range(step["q_abs_step"].shape[0])])
        zm = np.stack([zscore(step["mlp_abs_step"][i]) for i in range(step["mlp_abs_step"].shape[0])])
        zqc, zmc = zscore(imp["coc_vlm_q"]), zscore(imp["coc_vlm_mlp"])
        n = zq.shape[0] + 1
        out["znorm11"] = ((zq.sum(0) + zqc) / n, (zm.sum(0) + zmc) / n)
        out["znorm10"] = (zq.mean(0), zm.mean(0))
        out["znorm5050"] = (0.5 * zq.mean(0) + 0.5 * zqc, 0.5 * zm.mean(0) + 0.5 * zmc)
    out["traj"] = (imp["traj_vlm_q"], imp["traj_vlm_mlp"])
    out["coc"] = (imp["coc_vlm_q"], imp["coc_vlm_mlp"])
    out["dual"] = (np.maximum(rank_norm(imp["traj_vlm_q"]), rank_norm(imp["coc_vlm_q"])),
                   np.maximum(rank_norm(imp["traj_vlm_mlp"]), rank_norm(imp["coc_vlm_mlp"])))
    out["dualfix"] = (
        np.maximum(rank_skip_constant(imp["traj_vlm_q"]), rank_skip_constant(imp["coc_vlm_q"])),
        np.maximum(rank_skip_constant(imp["traj_vlm_mlp"]),
                   rank_skip_constant(imp["coc_vlm_mlp"])))
    return {k: v for k, v in out.items() if k in set(arms) | {"dual", "dualfix"}}


def agreement(a, b):
    """Per-layer |kept_a and kept_b| / |kept_a| (equal counts, so symmetric)."""
    return (a * b).sum(1) / np.maximum(a.sum(1), 1)


def jaccard(a, b):
    return float(np.mean((a * b).sum(1) / np.maximum(((a + b) > 0).sum(1), 1)))


def layer_spearman(a, b):
    """Mean within-layer Spearman; constant layers are undefined and skipped."""
    vals = [spearmanr(a[l], b[l])[0] for l in range(a.shape[0])
            if np.ptp(a[l]) > 0 and np.ptp(b[l]) > 0]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--importance", default="importance_v2_ada",
                    help="must be the same architecture as --stepvlm")
    ap.add_argument("--stepvlm", default="importance_stepvlm_v1")
    ap.add_argument("--ref", default="dual")
    ap.add_argument("--arms", nargs="+",
                    default=["znorm11", "znorm10", "znorm5050", "dualfix", "traj", "coc"])
    ap.add_argument("--ratio", type=float, default=U40)
    ap.add_argument("--out", default="criterion_overlap")
    args = ap.parse_args()
    out = REPO / "outputs" / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)

    imp = dict(np.load(REPO / "outputs" / args.importance / "importance.npz"))
    sp = REPO / "outputs" / args.stepvlm / "step_importance_vlm.npz"
    step = np.load(sp) if sp.exists() else None
    scores = build_scores(imp, step, args.arms + [args.ref])
    L, NQ = imp["traj_vlm_q"].shape
    NM = imp["traj_vlm_mlp"].shape[1]
    ratios = np.full(L, args.ratio)
    masks = {k: (ml.select_mask_ratios(q, ratios), ml.select_mask_ratios(m, ratios))
             for k, (q, m) in scores.items()}
    keep_q, keep_m = int(masks[args.ref][0][0].sum()), int(masks[args.ref][1][0].sum())

    flat_q = [l for l in range(L) if np.ptp(imp["traj_vlm_q"][l]) == 0]
    flat_m = [l for l in range(L) if np.ptp(imp["traj_vlm_mlp"][l]) == 0]
    res = {"ref": args.ref, "importance": args.importance, "stepvlm": args.stepvlm,
           "ratio": args.ratio, "keep_per_layer": {"q": keep_q, "mlp": keep_m},
           "constant_traj_layers": {"q": flat_q, "mlp": flat_m}, "pairs": {}}
    lines = [f"criterion kept-set overlap vs {args.ref} "
             f"(u{args.ratio:.4f}, keep {keep_q}/{NQ} heads, {keep_m}/{NM} channels per layer)",
             f"importance {args.importance}, per-step {args.stepvlm}",
             f"layers with structurally constant trajectory importance: Q {flat_q}, MLP {flat_m}",
             "",
             f"{'criterion':11s} {'Q agree':>8s} {'MLP agree':>10s} {'Q jacc':>7s} {'MLP jacc':>9s} "
             f"{'Q rho':>7s} {'MLP rho':>8s} {'param churn':>12s}"]
    for arm in args.arms:
        if arm == args.ref or arm not in masks:
            continue
        aq = agreement(masks[arm][0], masks[args.ref][0])
        am = agreement(masks[arm][1], masks[args.ref][1])
        fq = int(((masks[arm][0] == 1) & (masks[args.ref][0] == 0)).sum())
        fm = int(((masks[arm][1] == 1) & (masks[args.ref][1] == 0)).sum())
        churn = fq * P_Q + fm * P_M
        total = L * (keep_q * P_Q + keep_m * P_M)
        res["pairs"][arm] = {
            "q_agreement_mean": float(aq.mean()), "mlp_agreement_mean": float(am.mean()),
            "q_agreement_by_layer": aq.tolist(), "mlp_agreement_by_layer": am.tolist(),
            "q_agreement_ex_constant": float(np.delete(aq, flat_q).mean()) if flat_q else float(aq.mean()),
            "mlp_agreement_ex_constant": float(np.delete(am, flat_m).mean()) if flat_m else float(am.mean()),
            "q_jaccard": jaccard(masks[arm][0], masks[args.ref][0]),
            "mlp_jaccard": jaccard(masks[arm][1], masks[args.ref][1]),
            "q_spearman": layer_spearman(scores[arm][0], scores[args.ref][0]),
            "mlp_spearman": layer_spearman(scores[arm][1], scores[args.ref][1]),
            "flipped_heads": fq, "flipped_channels": fm,
            "param_churn": churn, "param_churn_frac": churn / total}
        r = res["pairs"][arm]
        lines.append(f"{arm:11s} {r['q_agreement_mean']:8.4f} {r['mlp_agreement_mean']:10.4f} "
                     f"{r['q_jaccard']:7.3f} {r['mlp_jaccard']:9.3f} {r['q_spearman']:+7.3f} "
                     f"{r['mlp_spearman']:+8.3f} {churn / total:11.1%}")

    # what the degenerate layer does to the reference criterion
    if flat_q:
        l = flat_q[0]
        kept = np.nonzero(masks[args.ref][0][l])[0]
        top_idx = set(range(NQ - keep_q, NQ))
        rt, rc = rank_norm(imp["traj_vlm_q"])[l], rank_norm(imp["coc_vlm_q"])[l]
        res["degenerate_layer"] = {
            "layer": l,
            "ref_kept_overlap_with_top_indices": len(set(kept.tolist()) & top_idx) / keep_q,
            "kept_decided_by_index_half": int(((rt > rc) & (masks[args.ref][0][l] > 0)).sum()),
            "rank_norm_is_index_order": bool(np.allclose(rt, np.arange(NQ) / (NQ - 1)))}
        d = res["degenerate_layer"]
        lines += ["", f"degenerate layer {l} under `{args.ref}`: rank_norm(traj) is the index "
                      f"order ({d['rank_norm_is_index_order']}); {d['kept_decided_by_index_half']}"
                      f"/{keep_q} kept heads were decided by that half; kept set overlaps the "
                      f"top-{keep_q} indices by {d['ref_kept_overlap_with_top_indices']:.0%}"]

    text = "\n".join(lines)
    print(text)
    (out / "criterion_overlap_summary.txt").write_text(text + "\n")
    (out / "metrics.json").write_text(json.dumps(res, indent=1))

    arms = [a for a in args.arms if a in res["pairs"]]
    if arms:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, key, title in zip(axes, ("q_agreement_by_layer", "mlp_agreement_by_layer"),
                                  (f"Q head kept-set agreement with {args.ref}",
                                   f"MLP channel kept-set agreement with {args.ref}")):
            for arm, c in zip(arms, (C1, C2, C3, C4, MUTED, INK)):
                ax.plot(range(L), res["pairs"][arm][key], "o-", ms=3, color=c, label=arm)
            for l in (flat_q if "q_" in key else flat_m):
                ax.axvline(l, color=MUTED, ls=":", lw=1)
            ax.set_xlabel("VLM layer")
            ax.set_ylabel("|kept both| / |kept|")
            ax.set_title(title)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "plots" / "criterion_overlap.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
