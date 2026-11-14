"""Appendix C.4 analyses (paper Fig. 3 token types, Fig. 5 expert) on Alpamayo 2 Edge.

Plan: iclr27-rebuttal/plans/2026-11-14_a2-edge-analysis-replication.md, step 5. CPU only, repo .venv.

Every number uses the function the A1.5 / A2 Super numbers came from, imported, not copied:
  Fig. 3 (a)     a2_fig3_token_type.agreement   (paper_same_position split-half, corrected rho; Q 50 / MLP 10 splits)
  Fig. 3 (b, c)  a2_fig3_token_type.shares      (additive token-type share)
  step tau       analyze_importance_a2.ratio_block / spec   (two-level fit of log sum I_FM / sum I_CE over depth)
  Fig. 5         a2_fig5_expert.step_matrix / cum_curve      (step x step Spearman, C(r) of Eq. 5)
Gates (all must pass before the Edge numbers are used):
  GA  the imported agreement / shares reproduce the stored A2 Q-head series (a2_fig3_token_type metrics) exactly
  GB  the imported step_matrix / cum_curve reproduce the stored A2 expert numbers (a2_fig5_expert metrics) exactly
  GC  ratio_block on a2_importance_v1 gives the tau the A2 panels use (q 51, mlp 50)
  GD  the Edge run passed its own gates (finalize: every tail ok, rope deltas equal, losses finite) and has 100 clips
Edge-only extra (exploratory): Edge writes "<NOMINAL>" instead of reasoning on uneventful clips, so the CoC NLL of
those clips covers a handful of tokens; (a) and (c) are recomputed on the reasoning clips alone.

Usage:
  .venv/bin/python iclr27-rebuttal/code/edge_paper_analysis.py [--exp-id iclr27-rebuttal/edge_paper_analysis]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/home/cvlab21/project/chan/alpamayo2-distill/experiments/importance")
import a2_fig3_token_type as f3
import a2_fig5_expert as f5
import analyze_importance_a2 as aia

warnings.filterwarnings("ignore", "Mean of empty slice")
EDGE = REPO / "outputs" / "iclr27-rebuttal" / "edge_importance_v1"
A2RUN = Path("/home/cvlab21/project/chan/alpamayo2-distill/outputs/a2_importance_v1")
STORED3 = REPO / "outputs" / "iclr27-rebuttal" / "a2_fig3_token_type" / "metrics.json"
STORED5 = REPO / "outputs" / "iclr27-rebuttal" / "a2_fig5_expert" / "metrics.json"
N_LAYERS = 28
SP = aia.spec(N_LAYERS, 16, 9216)
LATE = range(SP["late"].start, SP["late"].stop)  # 17..26, the A1.5 late band moved by relative depth
SHARE_BANDS = {"lo": range(SP["tok_lo"].start, SP["tok_lo"].stop), "hi": range(SP["tok_hi"].start, SP["tok_hi"].stop)}
EDGE_MODEL = {"name": "Alpamayo 2 Edge", "n": N_LAYERS, "band": (LATE.start - 0.5, N_LAYERS - 0.5),
              "ticks": [0, 9, 18, 27], "kv_group": 2}
LABELS = ("All", "Vision", "Prompt text", "CoC", "Ego history")


def same(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return bool(a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b))
                and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=0, atol=1e-12))


def typed(run, axis):
    z = np.load(run / f"anatomy_perclip_{axis}.npz")
    return (z["fm_full"], z["ce"]) if axis == "q" else (z["fm_type"], z["ce_type"])  # (N, 5, L, U)


def expert_block(a):
    mat, per_layer = f5.step_matrix(a)
    S = mat.shape[0]
    off = mat[~np.eye(S, dtype=bool)]
    _grid, curve, c_layer, kk = f5.cum_curve(a)
    return {"shape_S_L_U": list(a.shape), "mean_pair_rho": float(off.mean()), "min_pair_rho": float(off.min()),
            "rho_by_step_distance": [float(np.mean([mat[i, i + d] for i in range(S - d)])) for d in range(1, S)],
            "layer_mean_pair_rho": per_layer[~np.eye(S, dtype=bool)].mean(0).tolist(), "k_mark": kk,
            "C_mark": float(curve[kk]), "C_mark_per_layer": c_layer.tolist(), "matrix": mat.tolist(),
            "curve": curve.tolist()}


def band_means(series, bands):
    return {b: {lab: {"corrected": float(np.nanmean(np.asarray(s["corrected"])[list(ls)])),
                      "raw": float(np.nanmean(np.asarray(s["raw"])[list(ls)])),
                      "nan_layers": int(np.isnan(np.asarray(s["corrected"])[list(ls)]).sum())}
                for lab, s in series.items()} | {"layers": [ls[0], ls[-1]]} for b, ls in bands.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="iclr27-rebuttal/edge_paper_analysis")
    args = ap.parse_args()
    out = REPO / "outputs" / args.exp_id
    out.mkdir(parents=True, exist_ok=True)
    s3, s5 = json.loads(STORED3.read_text()), json.loads(STORED5.read_text())
    gates = {}

    # GA: A2 Q-head agreement and shares through the imported functions
    fm, ce = typed(A2RUN, "q")
    ag = f3.agreement(fm, ce, 63, f3.N_SPLIT["q"])
    st = s3["per_layer"]["a2/q"]
    gates["GA_a2_q_agreement_and_share_reproduced"] = (
        all(same(ag[k][f], st["agreement"][k][f]) for k in LABELS for f in ("corrected", "raw"))
        and same(f3.shares(fm), st["share"]["traj"]) and same(f3.shares(ce), st["share"]["coc"]))
    # GB: A2 expert numbers through the imported functions
    z2 = np.load(A2RUN / "step_importance.npz")
    gates["GB_a2_expert_reproduced"] = all(
        same(expert_block(z2[k].astype(np.float64))[f], s5["results"][f"a2/{ax}"][f])
        for ax, k in f5.KEYS.items() for f in ("matrix", "curve", "C_mark", "mean_pair_rho"))
    # GC: tau of the A2 panels from ratio_block
    rng = np.random.default_rng(0)
    r2 = aia.ratio_block(np.load(A2RUN / "importance.npz"), np.load(A2RUN / "importance_perclip.npz"),
                         aia.spec(64, 64, 25600), rng)
    gates["GC_a2_tau_q51_mlp50"] = bool(r2["q"]["tau"] == 51 and r2["mlp"]["tau"] == 50)
    # GD: the Edge run's own gates
    em = json.loads((EDGE / "metrics.json").read_text())
    gates["GD_edge_run_gates"] = bool(em["n_clips"] == 100 and em["all_tail_ok"] and em["all_rope_deltas_tf_eq_gen"]
                                      and em["all_finite"])
    g0 = [max(r["typed_vs_plain_relerr_q"]) for r in em["per_clip"] if "typed_vs_plain_relerr_q" in r]
    g0b = [max(r["g0b_steps_vs_accumulated_relerr"].values()) for r in em["per_clip"]
           if r.get("g0b_steps_vs_accumulated_relerr")]
    gates["GD_edge_G0_typed_vs_plain_max_relerr"] = float(max(g0)) if g0 else None
    gates["GD_edge_G0b_steps_vs_accumulated_max_relerr"] = float(max(g0b)) if g0b else None
    gates_ok = all(v for k, v in gates.items() if k.startswith(("GA", "GB", "GC")) or k == "GD_edge_run_gates") \
        and gates["GD_edge_G0_typed_vs_plain_max_relerr"] < 1e-3 and gates["GD_edge_G0b_steps_vs_accumulated_max_relerr"] < 1e-3
    print(json.dumps(gates, indent=1), flush=True)
    if not gates_ok:
        (out / "metrics.json").write_text(json.dumps({"gates": gates, "gates_ok": False}, indent=2))
        raise SystemExit("a gate failed; Edge numbers not computed")

    # ---- Edge ----
    ratio = aia.ratio_block(np.load(EDGE / "importance.npz"), np.load(EDGE / "importance_perclip.npz"), SP,
                            np.random.default_rng(0))
    tau = {ax: ratio[ax]["tau"] for ax in ("q", "mlp")}
    EDGE_MODEL["tau"] = tau
    cots = [r["cot"].strip() for r in em["per_clip"]]
    nominal = np.array([c == "<NOMINAL>" for c in cots])
    coc_len = np.array([r["coc_len"] for r in em["per_clip"]])
    nll = np.array([r["nll"] for r in em["per_clip"]])
    res = {}
    for axis in ("q", "mlp"):
        fm, ce = typed(EDGE, axis)
        last = N_LAYERS - 1
        agree = f3.agreement(fm, ce, last, f3.N_SPLIT[axis])
        r = {"agreement": {k: {f: np.asarray(v).tolist() for f, v in s.items()} for k, s in agree.items()},
             "share": {"traj": f3.shares(fm).tolist(), "coc": f3.shares(ce).tolist()}}
        bands = {"trunk": range(LATE.start), "late": LATE, "pre-step": range(tau[axis]),
                 "post-step": range(tau[axis], last)}
        r["band_agreement"] = band_means(agree, bands)
        r["band_share"] = {f"{obj}/{b}": {n: float(np.asarray(r["share"][obj])[i][list(ls)].mean())
                                          for n, i in (("vision", f3.VIS), ("prompt", f3.PT), ("coc", f3.COC),
                                                       ("hist", f3.HIST))}
                           for obj in ("traj", "coc") for b, ls in SHARE_BANDS.items()}
        # exploratory: reasoning clips only (drops the "<NOMINAL>" clips, whose CoC NLL covers a few tokens)
        if (~nominal).sum() >= 20:
            ag_r = f3.agreement(fm[~nominal], ce[~nominal], last, f3.N_SPLIT[axis])
            r["reasoning_only"] = {
                "n": int((~nominal).sum()),
                "agreement": {k: {f: np.asarray(s[f]).tolist() for f in ("corrected", "raw")} for k, s in ag_r.items()},
                "band_agreement": band_means(ag_r, bands),
                "share_coc": f3.shares(ce[~nominal]).tolist(), "share_traj": f3.shares(fm[~nominal]).tolist()}
        res[axis] = r
        del fm, ce
        print(f"  edge {axis} done", flush=True)
    ze = np.load(EDGE / "step_importance.npz")
    expert = {ax: expert_block(ze[k].astype(np.float64)) for ax, k in f5.KEYS.items()}

    metrics = {
        "gates": gates, "gates_ok": True, "model": {**EDGE_MODEL, "band": list(EDGE_MODEL["band"])},
        "bands": {"late": [LATE.start, LATE.stop - 1], "share_lo": [SHARE_BANDS["lo"].start, SHARE_BANDS["lo"].stop - 1],
                  "share_hi": [SHARE_BANDS["hi"].start, SHARE_BANDS["hi"].stop - 1], "tau_window": list(SP["tau_window"])},
        "step_fit": {ax: {k: v for k, v in ratio[ax].items()} for ax in ("q", "mlp")},
        "coc": {"n_nominal": int(nominal.sum()), "n_reasoning": int((~nominal).sum()),
                "coc_len_mean": float(coc_len.mean()), "coc_len_reasoning_mean": float(coc_len[~nominal].mean()),
                "nll_mean_nominal": float(nll[nominal].mean()) if nominal.any() else None,
                "nll_mean_reasoning": float(nll[~nominal].mean())},
        "per_layer": {f"edge/{ax}": {"agreement": r["agreement"], "share": r["share"]} for ax, r in res.items()},
        "band_agreement": {f"edge/{ax}": r["band_agreement"] for ax, r in res.items()},
        "band_share": {f"edge/{ax}": r["band_share"] for ax, r in res.items()},
        "reasoning_only": {f"edge/{ax}": r.get("reasoning_only") for ax, r in res.items()},
        "expert": {f"edge/{ax}": v for ax, v in expert.items()},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (out / "config.json").write_text(json.dumps({
        "source": str(EDGE), "a2_reference": str(A2RUN), "stored_a2_fig3": str(STORED3), "stored_a2_fig5": str(STORED5),
        "n_split": f3.N_SPLIT, "seed": 0, "spec": {k: (str(v) if isinstance(v, slice) else v) for k, v in SP.items()},
        "model": {**EDGE_MODEL, "band": list(EDGE_MODEL["band"])}}, indent=2))

    # summary with the stored A1.5 / A2 rows next to Edge
    L = ["Paper Fig. 3 / Fig. 5 analyses on Alpamayo 2 Edge (A1.5 / A2 Super rows from the stored 2026-11-08 runs)", ""]
    L += [f"{k}: {v}" for k, v in gates.items()]
    L += ["", f"CoC: {metrics['coc']}", "",
          "step fit of log(sum I_FM / sum I_CE) over depth (tau = first layer of the low level):"]
    for ax in ("q", "mlp"):
        rr = ratio[ax]
        L.append(f"  edge {ax}: tau {rr['tau']} (rel. depth {rr['tau'] / (N_LAYERS - 1):.2f}), step {rr['step']:.2f}x, "
                 f"R2 {rr['r2']:.2f}, boot tau mode {rr['boot_tau_mode']} ({rr['boot_tau_mode_share']:.2f}), "
                 f"step CI {rr['boot_step_ci'][0]:.2f}-{rr['boot_step_ci'][1]:.2f}, window {SP['tau_window']}")
    L.append("  reference: a15 tau 23/23 (rel 0.66), a2 tau 51/50 (rel 0.81/0.79)")
    for ax in ("q", "mlp"):
        L += ["", f"(a) corrected rho (raw), {'Q heads' if ax == 'q' else 'MLP channels'}",
              f"{'model':<6}{'band':<10}{'layers':>8}" + "".join(f"{lab:>18}" for lab in LABELS)]
        for key in ("a15", "a2"):
            for band in ("pre-step", "post-step"):
                v = s3["step_aligned_agreement"][f"{key}/{ax}/{band}"]
                L.append(f"{key:<6}{band:<10}{v['layers'][0]:>4}-{v['layers'][1]:<3}" + "".join(
                    f"{v[lab]['corrected']:>11.2f} ({v[lab]['raw']:.2f})" for lab in LABELS))
        for band in ("trunk", "late", "pre-step", "post-step"):
            v = res[ax]["band_agreement"][band]
            L.append(f"{'edge':<6}{band:<10}{v['layers'][0]:>4}-{v['layers'][1]:<3}" + "".join(
                f"{v[lab]['corrected']:>11.2f} ({v[lab]['raw']:.2f})" for lab in LABELS))
        ro = res[ax].get("reasoning_only")
        if ro:
            for band in ("pre-step", "post-step"):
                v = ro["band_agreement"][band]
                L.append(f"{'edge-R':<6}{band:<10}{v['layers'][0]:>4}-{v['layers'][1]:<3}" + "".join(
                    f"{v[lab]['corrected']:>11.2f} ({v[lab]['raw']:.2f})" for lab in LABELS)
                    + f"   (reasoning clips only, n={ro['n']})")
        L += ["", (f"(b, c) token shares, {'Q heads' if ax == 'q' else 'MLP channels'} "
                   f"(edge lo {SHARE_BANDS['lo'].start}-{SHARE_BANDS['lo'].stop - 1}, hi {SHARE_BANDS['hi'].start}-"
                   f"{SHARE_BANDS['hi'].stop - 1}; a15 6-17 / 27-34; a2 11-30 / 48-60)"),
              f"{'model':<6}{'obj':<6}{'band':<5}{'vision':>8}{'prompt':>8}{'CoC':>8}{'ego':>8}"]
        for key in ("a15", "a2"):
            for obj in ("traj", "coc"):
                for band in ("lo", "hi"):
                    v = s3["band_share"][f"{key}/{ax}/{obj}/{band}"]
                    L.append(f"{key:<6}{obj:<6}{band:<5}{v['vision']:>8.2f}{v['prompt']:>8.2f}{v['coc']:>8.2f}"
                             f"{v['hist']:>8.2f}")
        for obj in ("traj", "coc"):
            for band in ("lo", "hi"):
                v = res[ax]["band_share"][f"{obj}/{band}"]
                L.append(f"{'edge':<6}{obj:<6}{band:<5}{v['vision']:>8.2f}{v['prompt']:>8.2f}{v['coc']:>8.2f}"
                         f"{v['hist']:>8.2f}")
    L += ["", "Fig. 5 expert: mean step-pair rho, min pair, C(0.9); rho by step distance 1..9",
          f"{'model':<6}{'axis':<5}{'(S,L,U)':>16}{'mean rho':>10}{'min rho':>9}{'C(0.9)':>9}"]
    for key in ("a15", "a2", "edge"):
        for ax in ("q", "mlp"):
            r = expert[ax] if key == "edge" else s5["results"][f"{key}/{ax}"]
            L.append(f"{key:<6}{ax:<5}{tuple(r['shape_S_L_U'])!s:>16}{r['mean_pair_rho']:>10.3f}{r['min_pair_rho']:>9.3f}"
                     f"{r['C_mark'] * 100:>8.1f}%   " + " ".join(f"{v:.2f}" for v in r["rho_by_step_distance"]))
    (out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
