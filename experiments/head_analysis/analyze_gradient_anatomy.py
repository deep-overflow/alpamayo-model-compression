"""Gates G0, P1-P4 for the gradient-anatomy pass (plans/2026-09-20_gradient-anatomy.md).

  G0  integrity   (a) the typed grads sum to the shipped single-gate grad (--verify run);
                  (b) each seed split's total against the single full-seed backward, on every
                  clip -- the VLM backward is a deep bf16 graph, so this is a rounding floor,
                  not an identity; (c) |sum of parts| reproduces importance_v2_ada's ranking
                  in every layer. P1 and P2 use only the full-seed backward, so (b) does not
                  enter them.
  P1  support     the CE gradient sits on text-side positions in late layers and on vision
                  positions in early ones; the FM gradient sits on vision positions in both.
  P2  decisive    Q heads, layers 22-34: do the two objectives still agree once both are
                  restricted to the unit's action at VISION positions? >= 0.70 supports the
                  token-support account, <= 0.45 refutes it. Every agreement is corrected
                  for its own split-half ceiling, because a restricted score is noisier.
  P3  ports       units in layers 8-21 receive < 20% of their FM mass through cache layers
                  25-35 (> 40% refutes the port account of the I_traj fall).
  P4  positions   >= 65% of FM mass enters through vision-position cache entries at every
                  unit depth.

Shares are E|G_part| / sum_parts E|G_part|. The abs is taken after the sum in the shipped
score, so parts are not additive in magnitude; the cancellation index
sum_parts E|G_part| / E|sum_parts G_part| is reported next to every share.

Usage:
  python experiments/head_analysis/analyze_gradient_anatomy.py --exp gradanat_v1 \
      --verify-exp gradanat_verify --ref importance_v2_ada
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TYPES = ("vision", "hist", "prompt_text", "coc", "sink")
BANDS = ((0, 7), (8, 15), (16, 21), (22, 24), (25, 29), (30, 35))
VIS, HIST, PT, COC, SINK = range(5)
LAST = 35  # I_traj is identically zero at layer 35
EARLY, LATE = slice(0, 22), slice(22, LAST)
P1_LO, P1_HI = slice(6, 18), slice(27, LAST)

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})


def rho_layers(a, b):
    """a, b (L, U) -> (L,) within-layer Spearman, NaN where either row is constant."""
    out = np.full(a.shape[0], np.nan)
    for l in range(a.shape[0]):
        if np.ptp(a[l]) > 0 and np.ptp(b[l]) > 0:
            out[l] = spearmanr(a[l], b[l])[0]
    return out


def split_half(t, c, n_split, rng):
    """Per-clip scores t, c (N, L, U) -> self_t, self_c, cross (L,) across DISJOINT halves."""
    n = len(t)
    st, sc_, cr = [], [], []
    for _ in range(n_split):
        p = rng.permutation(n)
        a, b = p[:n // 2], p[n // 2:]
        ta, tb, ca, cb = t[a].mean(0), t[b].mean(0), c[a].mean(0), c[b].mean(0)  # (L, U)
        st.append(rho_layers(ta, tb))
        sc_.append(rho_layers(ca, cb))
        cr.append(0.5 * (rho_layers(ta, cb) + rho_layers(tb, ca)))
    return np.nanmean(st, 0), np.nanmean(sc_, 0), np.nanmean(cr, 0)


def corrected(st, sc_, cr, band):
    """Band summary: mean of per-layer cross / sqrt(self*self) where both ceilings are usable,
    and the same ratio taken on band means (robust when single layers have low ceilings)."""
    st, sc_, cr = st[band], sc_[band], cr[band]
    ok = (st > 0.2) & (sc_ > 0.2)
    per_layer = float(np.nanmean(cr[ok] / np.sqrt(st[ok] * sc_[ok]))) if ok.any() else float("nan")
    of_means = float(np.nanmean(cr) / np.sqrt(max(np.nanmean(st), 1e-9) * max(np.nanmean(sc_), 1e-9)))
    return {"self_traj": float(np.nanmean(st)), "self_coc": float(np.nanmean(sc_)),
            "cross": float(np.nanmean(cr)), "corrected": per_layer, "corrected_of_means": of_means,
            "layers_used": int(ok.sum())}


def agreement_block(name, t, c, n_split, rng, out, lines):
    st, sc_, cr = split_half(t, c, n_split, rng)
    out[name] = {"early": corrected(st, sc_, cr, EARLY), "late": corrected(st, sc_, cr, LATE),
                 "per_layer": {"self_traj": st.tolist(), "self_coc": sc_.tolist(), "cross": cr.tolist()}}
    e, l = out[name]["early"], out[name]["late"]
    lines.append(f"  {name:34s} early: self {e['self_traj']:.2f}/{e['self_coc']:.2f} cross {e['cross']:.2f} "
                 f"corrected {e['corrected']:.2f} | late: self {l['self_traj']:.2f}/{l['self_coc']:.2f} "
                 f"cross {l['cross']:.2f} corrected {l['corrected']:.2f} (of means {l['corrected_of_means']:.2f}, "
                 f"{l['layers_used']}/13 layers)")
    return st, sc_, cr


def shares(x):
    """x (n_parts, L, U) mean |G| -> (n_parts, L) share of each part per layer."""
    m = x.sum(-1)  # (n_parts, L)
    return m / np.maximum(m.sum(0, keepdims=True), 1e-300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="gradanat_v1")
    ap.add_argument("--verify-exp", default="gradanat_verify")
    ap.add_argument("--ref", default="importance_v2_ada")
    ap.add_argument("--n-split", type=int, default=50)
    ap.add_argument("--n-split-mlp", type=int, default=10)
    args = ap.parse_args()

    d = REPO / "outputs" / args.exp
    (d / "plots").mkdir(exist_ok=True)
    z = dict(np.load(d / "anatomy.npz"))
    pq = dict(np.load(d / "anatomy_perclip_q.npz"))
    met = json.loads((d / "metrics.json").read_text())
    ref = np.load(REPO / "outputs" / args.ref / "importance.npz")
    n = met["n_clips"]
    rng = np.random.default_rng(0)
    out, lines = {"n_clips": n}, [f"gradient anatomy -- {args.exp}, {n} clips", ""]
    L = np.arange(36)

    # ------------------------------------------------------------------ G0
    be = np.array([r["bands_vs_full_relerr_by_layer"] for r in met["per_clip"]])  # (N, 35)
    pe = np.array([r["pos_vs_full_relerr_by_layer"] for r in met["per_clip"]])  # (N, 35)
    rb = np.array([r["bands_vs_full_rank_rho_min"] for r in met["per_clip"]])
    rp = np.array([r["pos_vs_full_rank_rho_min"] for r in met["per_clip"]])
    g0 = {"split_vs_full_relerr_median_layer": float(np.median(np.r_[be, pe])),
          "split_vs_full_relerr_worst_layer_max": float(max(be.max(), pe.max())),
          "split_vs_full_relerr_worst_layer_median_clip": float(np.median(np.r_[be.max(1), pe.max(1)])),
          "split_vs_full_rank_rho_min": float(min(rb.min(), rp.min())),
          "split_vs_full_relerr_by_layer": np.median(np.r_[be, pe], 0).tolist()}
    vpath = REPO / "outputs" / args.verify_exp / "metrics.json"
    if vpath.exists():
        v = json.loads(vpath.read_text())
        errs = np.array([r["typed_vs_single_relerr"] for r in v["per_clip"]])  # (n_verify, 13)
        g0.update({"verify_clips": len(errs), "typed_vs_single_relerr_max": float(errs.max()),
                   "typed_vs_single_relerr_ce_max": float(errs[:, 0].max()),
                   "typed_vs_single_relerr_fm_max": float(errs[:, 1:].max())})
    repro = {}
    for a in ("q", "mlp"):
        rt = rho_layers(z[f"fm_full_{a}"], ref[f"traj_vlm_{a}"])[:LAST]
        rc = rho_layers(z[f"ce_full_{a}"], ref[f"coc_vlm_{a}"])
        repro[a] = {"traj_min": float(np.nanmin(rt)), "traj_mean": float(np.nanmean(rt)),
                    "coc_min": float(np.nanmin(rc)), "coc_mean": float(np.nanmean(rc)),
                    "traj_layers_below_0.99": [int(i) for i in np.where(rt < 0.99)[0]],
                    "coc_layers_below_0.99": [int(i) for i in np.where(rc < 0.99)[0]]}
    g0["reproduces_ref"] = repro
    g0["layer35_fm_is_zero"] = bool(z["fm_full_q"][35].max() == 0 and z["fm_full_mlp"][35].max() == 0)
    g0["a_typed_sum_pass"] = bool(g0.get("typed_vs_single_relerr_max", 1.0) < 1e-3)
    g0["b_seed_split_within_1e-3"] = bool(g0["split_vs_full_relerr_worst_layer_max"] < 1e-3)
    g0["c_reproduces_ref_pass"] = bool(repro["q"]["traj_min"] >= 0.99 and repro["q"]["coc_min"] >= 0.99)
    out["G0"] = g0
    lines += [
        "G0 integrity",
        (f"  (a) typed grads summed over types vs the shipped single gate (--verify, "
        f"{g0.get('verify_clips', 0)} clips x 13 backwards): max rel err "
        f"{g0.get('typed_vs_single_relerr_max', float('nan')):.1e} (<1e-3) -> "
        f"{'PASS' if g0['a_typed_sum_pass'] else 'FAIL'}"),
        (f"  (b) band / position seed splits vs the single full-seed backward, {len(be)} clips: rel err "
        f"median layer {g0['split_vs_full_relerr_median_layer']:.1e}, worst layer of a clip median "
        f"{g0['split_vs_full_relerr_worst_layer_median_clip']:.1e} max "
        f"{g0['split_vs_full_relerr_worst_layer_max']:.1e}; within-layer rank rho min "
        f"{g0['split_vs_full_rank_rho_min']:.4f}"),
        (f"      -> the pre-registered 1e-3 is {'met' if g0['b_seed_split_within_1e-3'] else 'NOT met'}: "
        "a bf16 backward is linear in its seed only up to rounding (analyze_stepvlm V0 measured "
        "the same floor, ~5e-3 median, and set 2e-2). D2/D3 (P3, P4) carry this error; D1 (P1, P2) "
        "is read off the full-seed backward and does not"),
        (f"  (c) |sum of parts| vs {args.ref}, per-layer Spearman: Q traj min {repro['q']['traj_min']:.3f} "
        f"coc min {repro['q']['coc_min']:.3f} (>=0.99) | MLP traj min {repro['mlp']['traj_min']:.3f} "
        f"coc min {repro['mlp']['coc_min']:.3f} -> {'PASS' if g0['c_reproduces_ref_pass'] else 'FAIL'}"),
        f"  layer 35 FM exactly zero: {g0['layer35_fm_is_zero']}", ""]

    # ------------------------------------------------------------------ P1
    p1 = {}
    lines.append("P1 token support (share of E|G| by the token type the unit acts on)")
    for a in ("q", "mlp"):
        ce, fm = shares(z[f"ce_{a}"]), shares(z[f"fm_type_{a}"])  # (5, 36)
        ce_text, fm_vis = ce[PT] + ce[COC], fm[VIS]
        canc_ce = z[f"ce_{a}"].sum((0, 2)) / np.maximum(z[f"ce_full_{a}"].sum(-1), 1e-300)
        canc_fm = z[f"fm_type_{a}"].sum((0, 2))[:LAST] / z[f"fm_full_{a}"].sum(-1)[:LAST]
        p1[a] = {"ce_text_L6_17": float(ce_text[P1_LO].mean()), "ce_text_L27_34": float(ce_text[P1_HI].mean()),
                 "fm_vision_L6_17": float(fm_vis[P1_LO].mean()), "fm_vision_L27_34": float(fm_vis[P1_HI].mean()),
                 "ce_vision_L6_17": float(ce[VIS][P1_LO].mean()), "ce_vision_L27_34": float(ce[VIS][P1_HI].mean()),
                 "fm_text_L27_34": float((fm[PT] + fm[COC])[P1_HI].mean()),
                 "ce_text_first_layer_above_half": int(np.argmax(ce_text > 0.5)) if (ce_text > 0.5).any() else None,
                 "cancellation_ce_L27_34": float(canc_ce[P1_HI].mean()),
                 "cancellation_fm_L27_34": float(canc_fm[P1_HI].mean()),
                 "ce_share_by_layer": ce.tolist(), "fm_share_by_layer": fm.tolist()}
        p1[a]["pass"] = bool(p1[a]["ce_text_L6_17"] < 0.35 and p1[a]["ce_text_L27_34"] > 0.65
                             and p1[a]["fm_vision_L6_17"] > 0.65 and p1[a]["fm_vision_L27_34"] > 0.65)
        lines.append(f"  {a:3s}: CE text-side share L6-17 {p1[a]['ce_text_L6_17']:.3f} (<0.35) "
                     f"L27-34 {p1[a]['ce_text_L27_34']:.3f} (>0.65) | FM vision share L6-17 "
                     f"{p1[a]['fm_vision_L6_17']:.3f} L27-34 {p1[a]['fm_vision_L27_34']:.3f} (>0.65) "
                     f"-> {'PASS' if p1[a]['pass'] else 'FAIL'}")
        lines.append(f"       CE text-side share first exceeds 0.5 at layer {p1[a]['ce_text_first_layer_above_half']}; "
                     f"cancellation index L27-34: CE {p1[a]['cancellation_ce_L27_34']:.2f} FM {p1[a]['cancellation_fm_L27_34']:.2f}")
    res_ce = pq["res_ce"].mean(0)  # (36, 5) residual-gradient energy by token type
    res_fm = pq["res_fm"].mean(0)
    rs_ce = res_ce / res_ce.sum(1, keepdims=True)
    rs_fm = res_fm / np.maximum(res_fm.sum(1, keepdims=True), 1e-300)
    p1["residual"] = {"ce_text_L6_17": float((rs_ce[:, PT] + rs_ce[:, COC])[P1_LO].mean()),
                      "ce_text_L27_34": float((rs_ce[:, PT] + rs_ce[:, COC])[P1_HI].mean()),
                      "fm_vision_L6_17": float(rs_fm[:, VIS][P1_LO].mean()),
                      "fm_vision_L27_34": float(rs_fm[:, VIS][P1_HI].mean()),
                      "ce_share_by_layer": rs_ce.T.tolist(), "fm_share_by_layer": rs_fm.T.tolist()}
    lines.append(f"  residual-stream gradient energy (no units involved): CE text-side L6-17 "
                 f"{p1['residual']['ce_text_L6_17']:.3f} L27-34 {p1['residual']['ce_text_L27_34']:.3f} | "
                 f"FM vision L6-17 {p1['residual']['fm_vision_L6_17']:.3f} L27-34 {p1['residual']['fm_vision_L27_34']:.3f}")
    p1["pass"] = bool(p1["q"]["pass"] and p1["mlp"]["pass"])
    lines += [f"  -> {'PASS' if p1['pass'] else 'FAIL'} (both axes required)", ""]
    out["P1"] = p1

    # ------------------------------------------------------------------ P2
    lines.append("P2 same-token agreement, split-half ceiling corrected "
                 "(early = layers 0-21, late = 22-34)")
    ce, fm = pq["ce"], pq["fm_full"]  # (N, 5, 36, 32) signed; FM from the full-seed backward
    p2 = {"q": {}, "mlp": {}}
    lines.append(" Q heads")
    curves = {}
    for name, t, c in (
        ("pooled (the shipped score)", np.abs(fm.sum(1)), np.abs(ce.sum(1))),
        ("both at VISION positions", np.abs(fm[:, VIS]), np.abs(ce[:, VIS])),
        ("both at TEXT-side positions", np.abs(fm[:, PT] + fm[:, COC]), np.abs(ce[:, PT] + ce[:, COC])),
        ("traj at vision vs CoC at text", np.abs(fm[:, VIS]), np.abs(ce[:, PT] + ce[:, COC])),
    ):
        curves[name] = agreement_block(name, t[:, :LAST], c[:, :LAST], args.n_split, rng, p2["q"], lines)
    # "text-side" pools two spans; the generated-CoC span alone is literally the same ~10-16
    # positions for both losses, and the prompt-text span holds the position that predicts
    # the first CoC token
    for name, t, c in (("both at CoC positions only", np.abs(fm[:, COC]), np.abs(ce[:, COC])),
                       ("both at prompt-text positions only", np.abs(fm[:, PT]), np.abs(ce[:, PT]))):
        agreement_block(name, t[:, :LAST], c[:, :LAST], args.n_split, rng, p2["q"], lines)
    # within one objective: is a unit's importance at vision positions the same ranking as at text?
    for name, x, y in (("traj: vision vs text-side", np.abs(fm[:, VIS]), np.abs(fm[:, PT] + fm[:, COC])),
                       ("CoC: vision vs text-side", np.abs(ce[:, VIS]), np.abs(ce[:, PT] + ce[:, COC]))):
        agreement_block(name, x[:, :LAST], y[:, :LAST], args.n_split, rng, p2["q"], lines)
    val = p2["q"]["both at VISION positions"]["late"]["corrected"]
    p2["verdict"] = ("H-token supported" if val >= 0.70 else
                     "H-token refuted; H-dir stands" if val <= 0.45 else "both contribute")
    p2["decisive_value"] = val
    mpath = d / "anatomy_perclip_mlp.npz"
    if mpath.exists():
        lines.append(" MLP channels (descriptive)")
        pm = np.load(mpath)
        cem, fmm = pm["ce_type"], pm["fm_type"]  # (N, 5, 36, 12288) signed
        for name, t, c in (
            ("pooled (the shipped score)", np.abs(fmm.sum(1)), np.abs(cem.sum(1))),
            ("both at VISION positions", np.abs(fmm[:, VIS]), np.abs(cem[:, VIS])),
            ("both at TEXT-side positions", np.abs(fmm[:, PT] + fmm[:, COC]), np.abs(cem[:, PT] + cem[:, COC])),
            ("both at CoC positions only", np.abs(fmm[:, COC]), np.abs(cem[:, COC])),
        ):
            agreement_block(name, t[:, :LAST], c[:, :LAST], args.n_split_mlp, rng, p2["mlp"], lines)
    lines += [f"  -> decisive value (Q heads, vision positions, late, corrected) = {val:.3f}: {p2['verdict']}", ""]
    out["P2"] = p2

    # ------------------------------------------------------------------ additive split
    # E|G_type| shares do not add up (the abs comes after the sum over types). The signed share
    # sum_u G_type,u sign(G_full,u) does: over types it returns sum_u |G_full,u|, the layer's
    # shipped score. So this is the exact answer to "how much of I_CoC / I_traj at this depth
    # comes from what the unit does at vision positions". Not pre-registered; descriptive.
    lines.append("additive split of the shipped layer score by the token type the unit acts on "
                 "(signed shares, sum to 1)")
    add = {}
    per_axis = {"q": (pq["ce"], pq["fm_full"])}
    if mpath.exists():
        per_axis["mlp"] = (cem, fmm)
    for a, (c_arr, f_arr) in per_axis.items():
        for name, arr in (("ce", c_arr), ("fm", f_arr)):
            part = np.zeros((len(TYPES), 36))
            for i in range(len(arr)):  # clip by clip: the MLP array is 885 MB
                g = arr[i].astype(np.float64)  # (5, 36, U)
                part += (g * np.sign(g.sum(0, keepdims=True))).sum(-1)
            part /= len(arr)
            tot = part.sum(0)  # (36,) == the shipped layer sum
            sh = part / np.where(tot > 0, tot, np.nan)
            add[f"{a}_{name}"] = {"share_by_type_and_layer": np.nan_to_num(sh).tolist(),
                                  "vision_L6_17": float(np.nanmean(sh[VIS, P1_LO])),
                                  "vision_L27_34": float(np.nanmean(sh[VIS, P1_HI])),
                                  "text_L6_17": float(np.nanmean((sh[PT] + sh[COC])[P1_LO])),
                                  "text_L27_34": float(np.nanmean((sh[PT] + sh[COC])[P1_HI])),
                                  "hist_L27_34": float(np.nanmean(sh[HIST, P1_HI]))}
            r = add[f"{a}_{name}"]
            lines.append(f"  {a:3s} {name}: vision L6-17 {r['vision_L6_17']:.3f} -> L27-34 {r['vision_L27_34']:.3f} | "
                         f"text-side L6-17 {r['text_L6_17']:.3f} -> L27-34 {r['text_L27_34']:.3f}")
    lines.append("")
    out["additive_by_type"] = add

    # ------------------------------------------------------------------ P3
    p3 = {}
    lines.append("P3 ports: share of FM mass by the cache-layer band it enters through")
    for a in ("q", "mlp"):
        mass = z[f"fm_bandtot_{a}"].sum(-1)  # (6, 36)
        leak = max(float(mass[b, hi:].max()) for b, (lo, hi) in enumerate(BANDS))  # structurally zero
        sh = mass / np.maximum(mass.sum(0, keepdims=True), 1e-300)
        late = float(mass[4:, 8:22].sum() / mass[:, 8:22].sum())
        canc = float((mass.sum(0)[:LAST] / z[f"fm_full_{a}"].sum(-1)[:LAST])[8:22].mean())
        p3[a] = {"late_cache_share_units_L8_21": late, "structural_zero_max": leak,
                 "cancellation_L8_21": canc, "share_by_band_and_layer": sh.tolist(),
                 "band_share_units_L8_21": (mass[:, 8:22].sum(1) / mass[:, 8:22].sum()).tolist()}
        lines.append(f"  {a:3s}: units in L8-21 receive {late:.3f} of FM mass through cache layers 25-35 "
                     f"(<0.20 pass, >0.40 refutes); by band {np.round(p3[a]['band_share_units_L8_21'], 3).tolist()}; "
                     f"cells that must be zero: max {leak:.1e}; cancellation {canc:.2f}")
    worst = max(p3["q"]["late_cache_share_units_L8_21"], p3["mlp"]["late_cache_share_units_L8_21"])
    p3["verdict"] = "PASS" if worst < 0.20 else "REFUTED" if worst > 0.40 else "in between"
    lines += [f"  -> {p3['verdict']}", ""]
    out["P3"] = p3

    # ------------------------------------------------------------------ P4
    p4 = {}
    lines.append("P4 read positions: share of FM mass by the cache position type it enters through")
    ntok = np.array([[r["n_tok"][t] for t in TYPES] for r in met["per_clip"]]).mean(0)
    for a in ("q", "mlp"):
        mass = z[f"fm_postot_{a}"].sum(-1)[:, :LAST]  # (5, 35)
        sh = mass / mass.sum(0, keepdims=True)
        p4[a] = {"vision_share_min": float(sh[VIS].min()), "vision_share_mean": float(sh[VIS].mean()),
                 "vision_share_argmin_layer": int(sh[VIS].argmin()),
                 "share_mean_by_type": dict(zip(TYPES, sh.mean(1).tolist())),
                 "share_by_type_and_layer": sh.tolist()}
        dens = sh.mean(1) / ntok
        p4[a]["per_token_density_coc_over_prompt_text"] = float(dens[COC] / dens[PT])
        p4[a]["per_token_density_coc_over_vision"] = float(dens[COC] / dens[VIS])
        lines.append(f"  {a:3s}: vision share min over unit layers {sh[VIS].min():.3f} (layer {sh[VIS].argmin()}), "
                     f"mean {sh[VIS].mean():.3f}; by type {np.round(sh.mean(1), 3).tolist()}; per-token density "
                     f"CoC/prompt_text {p4[a]['per_token_density_coc_over_prompt_text']:.1f}x")
    direct = z["direct"] / z["direct"].sum(1, keepdims=True)  # (36, 5) the expert's direct read
    p4["direct_read_share_by_cache_layer"] = direct.T.tolist()
    p4["direct_read_vision_share_mean"] = float(direct[:, VIS].mean())
    p4["mean_tokens"] = dict(zip(TYPES, ntok.tolist()))
    p4["pass"] = bool(min(p4["q"]["vision_share_min"], p4["mlp"]["vision_share_min"]) >= 0.65)
    lines += [(f"  expert's direct read |k dk|+|v dv|: vision share mean over cache layers "
              f"{p4['direct_read_vision_share_mean']:.3f}; mean tokens {np.round(ntok, 1).tolist()}"),
              f"  -> {'PASS' if p4['pass'] else 'FAIL'}", ""]
    out["P4"] = p4

    # ------------------------------------------------------------------ fallback: directions
    e_ce, e_fm, dot = pq["res_ce"], pq["res_fm"], pq["res_dot"]  # (N, 36, 5)
    cos_cat = np.nanmean(dot / np.sqrt(np.maximum(e_ce * e_fm, 1e-300)), 0)  # (36, 5)
    out["residual_cosine"] = {"concat_by_layer": cos_cat.T.tolist(),
                              "per_position_mean_by_layer": pq["res_cos"].mean(0).T.tolist(),
                              "vision_early": float(cos_cat[EARLY, VIS].mean()),
                              "vision_late": float(cos_cat[LATE, VIS].mean()),
                              "text_early": float(cos_cat[EARLY][:, [PT, COC]].mean()),
                              "text_late": float(cos_cat[LATE][:, [PT, COC]].mean())}
    rc = out["residual_cosine"]
    lines += ["residual-gradient cosine between the two losses at the same positions (descriptive)",
              (f"  vision positions: early {rc['vision_early']:+.3f} late {rc['vision_late']:+.3f} | "
              f"text-side: early {rc['text_early']:+.3f} late {rc['text_late']:+.3f}"), ""]

    # ------------------------------------------------------------------ ratio by token type
    lines.append("I_traj / I_CoC by the token type the unit acts on (descriptive; raw layer sums)")
    out["ratio_by_type"] = {}
    for a in ("q", "mlp"):
        for name, t, c in (("vision", z[f"fm_type_{a}"][VIS], z[f"ce_{a}"][VIS]),
                           ("text-side", z[f"fm_text_{a}"], z[f"ce_text_{a}"]),
                           ("pooled", z[f"fm_full_{a}"], z[f"ce_full_{a}"])):
            r = t.sum(-1)[:LAST] / c.sum(-1)[:LAST]
            out["ratio_by_type"][f"{a}_{name}"] = r.tolist()
            lines.append(f"  {a:3s} {name:9s}: L0-21 {np.exp(np.log(r[:22]).mean()):8.2f}x  L23-34 "
                         f"{np.exp(np.log(r[23:]).mean()):8.2f}x  step {np.exp(np.log(r[:22]).mean() - np.log(r[23:]).mean()):.2f}x")

    # ------------------------------------------------------------------ plots
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, a, title in zip(axes, ("q", "mlp"), ("Q heads", "MLP channels")):
        ce_s, fm_s = np.array(p1[a]["ce_share_by_layer"]), np.array(p1[a]["fm_share_by_layer"])
        ax.plot(L, ce_s[PT] + ce_s[COC], color=C4, lw=2, label="CE: text-side positions")
        ax.plot(L, ce_s[VIS], color=C4, lw=1.2, ls="--", label="CE: vision positions")
        ax.plot(L[:LAST], fm_s[VIS][:LAST], color=C1, lw=2, label="FM: vision positions")
        ax.plot(L[:LAST], (fm_s[PT] + fm_s[COC])[:LAST], color=C1, lw=1.2, ls="--", label="FM: text-side positions")
        ax.axvspan(21.5, 35.5, color=MUTED, alpha=0.08, lw=0)
        ax.set_title(f"{title}: where each loss sees a unit")
        ax.set_xlabel("VLM layer")
        ax.grid(axis="y", color="#E8E6DC", lw=0.7)
    axes[0].set_ylabel("share of E|dL/dg| by token type")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, ncol=4, loc="lower center")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(d / "plots" / "token_support.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, a, title in zip(axes, ("q", "mlp"), ("Q heads", "MLP channels")):
        if f"{a}_ce" not in add:
            continue
        c_s = np.array(add[f"{a}_ce"]["share_by_type_and_layer"])
        f_s = np.array(add[f"{a}_fm"]["share_by_type_and_layer"])
        ax.plot(L, c_s[VIS], color=C4, lw=2, label="I_CoC: from vision positions")
        ax.plot(L[:LAST], f_s[VIS][:LAST], color=C1, lw=2, label="I_traj: from vision positions")
        ax.plot(L, c_s[PT] + c_s[COC], color=C4, lw=1.2, ls="--", label="I_CoC: from text-side positions")
        ax.plot(L[:LAST], (f_s[PT] + f_s[COC])[:LAST], color=C1, lw=1.2, ls="--",
                label="I_traj: from text-side positions")
        ax.axvspan(21.5, 35.5, color=MUTED, alpha=0.08, lw=0)
        ax.set_title(f"{title}: additive share of the shipped layer score")
        ax.set_xlabel("VLM layer")
        ax.grid(axis="y", color="#E8E6DC", lw=0.7)
    axes[0].set_ylabel("share of the layer's importance")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, ncol=4, loc="lower center")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(d / "plots" / "additive_token_split.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for (name, (st, sc_, cr)), col in zip(curves.items(), (INK, C1, C4, C3)):
        ok = (st > 0.2) & (sc_ > 0.2)
        ax.plot(L[:LAST][ok], (cr / np.sqrt(np.where(ok, st * sc_, 1)))[ok], color=col, lw=1.8, label=name)
    ax.axvspan(21.5, 35.5, color=MUTED, alpha=0.08, lw=0)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylim(-0.3, 1.1)
    ax.set_xlabel("VLM layer")
    ax.set_ylabel("ceiling-corrected rank agreement, traj vs CoC")
    ax.set_title("Q heads: does restricting both losses to the same tokens restore agreement?")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color="#E8E6DC", lw=0.7)
    fig.tight_layout()
    fig.savefig(d / "plots" / "same_token_agreement.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    sh = np.array(p3["mlp"]["share_by_band_and_layer"])[:, :LAST]  # (6, 35)
    im = axes[0].imshow(sh, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="lower")
    axes[0].set_yticks(range(len(BANDS)), [f"{lo}-{hi}" for lo, hi in BANDS])
    axes[0].set_xlabel("layer of the unit (MLP)")
    axes[0].set_ylabel("cache layers the FM gradient enters through")
    axes[0].set_title("Ports: share of a layer's FM mass by cache band")
    plt.colorbar(im, ax=axes[0], fraction=0.04, pad=0.02)
    psh = np.array(p4["mlp"]["share_by_type_and_layer"])  # (5, 35)
    bottom = np.zeros(LAST)
    for t, col in zip(range(5), (C1, C2, C4, C3, MUTED)):
        axes[1].bar(L[:LAST], psh[t], bottom=bottom, color=col, width=0.85, label=TYPES[t])
        bottom += psh[t]
    axes[1].set_xlabel("layer of the unit (MLP)")
    axes[1].set_ylabel("share of FM mass")
    axes[1].set_title("Read positions: cache entries the FM gradient enters through")
    axes[1].legend(frameon=False, fontsize=8, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.38))
    fig.tight_layout()
    fig.savefig(d / "plots" / "ports_and_positions.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(L, rs_ce[:, PT] + rs_ce[:, COC], color=C4, lw=2, label="CE: text-side")
    axes[0].plot(L[:LAST], rs_fm[:LAST, VIS], color=C1, lw=2, label="FM: vision")
    axes[0].plot(L[:LAST], (rs_fm[:, PT] + rs_fm[:, COC])[:LAST], color=C1, lw=1.2, ls="--", label="FM: text-side")
    axes[0].set_title("Residual-gradient energy by token type (no units)")
    axes[0].set_xlabel("after VLM layer")
    axes[0].set_ylabel("share of ||dL/dh||^2")
    axes[0].legend(frameon=False, fontsize=8)
    for t, col in ((VIS, C1), (PT, C4), (COC, C3), (HIST, C2)):
        axes[1].plot(L[:LAST], cos_cat[:LAST, t], color=col, lw=1.8, label=TYPES[t])
    axes[1].axhline(0, color=MUTED, lw=0.8)
    axes[1].set_title("Cosine between the two losses' residual gradients")
    axes[1].set_xlabel("after VLM layer")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.axvspan(21.5, 35.5, color=MUTED, alpha=0.08, lw=0)
        ax.grid(axis="y", color="#E8E6DC", lw=0.7)
    fig.tight_layout()
    fig.savefig(d / "plots" / "residual_gradients.png", dpi=160)
    plt.close(fig)

    (d / "summary.txt").write_text("\n".join(lines) + "\n")
    (d / "metrics_analysis.json").write_text(json.dumps(out, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
