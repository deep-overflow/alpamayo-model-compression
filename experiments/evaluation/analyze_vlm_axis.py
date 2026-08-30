"""VLM axis decomposition: Q heads vs MLP channels at dual_u40_v2's budget and score.

plans/2026-08-30_axis-taylor-comparability.md, Stage 2. The expert twin
(analyze_expert_axis.py) found the cost sits entirely on the Q-head axis. First-order
Taylor predicts the OPPOSITE for the VLM -- per parameter, importance_v2 puts 2.9x
(traj) / 2.6x (CoC) more mass on an MLP channel than on a Q head -- so this is a
pre-registered directional test, not an exploration.

dual_u40_v2's own masks are axis-separable (two independent select_mask_ratios calls),
so dualq | dualm == dual_u40_v2 unit for unit and the shipped `dual` rows are the
additivity arm at no extra compute.

  V0  recipe integrity: removed params exact, expert/KV/other axis untouched,
      dualq | dualm == dual_u40_v2 kept set for kept set
  V1  ratio-matched: vq - vm, paired per clip. Prediction: BELOW 0 (the MLP axis costs
      more), the opposite sign of the expert's G1
  V4  parameter-matched: vq - vm_pm (490.7M vs 490.6M, only the axis differs)
  V2  additivity: dual - (vq + vm)
  V5  ordering: do the three arms' measured deltas rank as sum_S I_traj predicts?

Unlike the expert arms, VLM cuts change the CoC, so CoC degeneracy and gen_coc identity
are reported as observations rather than gates.

Writes outputs/<out>/{metrics.json, summary.txt, plots/*.png}. Arms without rows are
skipped, so this can run while the queue drains.

Usage:
  .venv/bin/python experiments/evaluation/analyze_vlm_axis.py [--out vlm_axis_analysis]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "head_analysis"))
import eval_lib as el  # noqa: E402
import paper_numbers as pn  # noqa: E402
from run_cocsafe import rank_norm  # noqa: E402

BG, INK, MUTED, GRID = "#FAF9F5", "#29261B", "#6B6555", "#E8E6DC"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.size": 9, "axes.grid": True, "axes.axisbelow": True,
})
REPO = Path(__file__).resolve().parents[2]
SETS = ("indist", "test", "oodval")
P_HEAD, P_MLPC = 2 * 4096 * 128, 3 * 4096
# arm -> (recipe dir, axis, label, expected removed params, same-architecture baseline)
#
# The three axis arms were evaluated on Blackwell: a parallel session took every Ada card
# on 2026-08-30, and `baseline_bw_ps_*` already exists with per-sample arrays for all
# three sets (it was measured for the expert-axis 50% arms). dual_u40_v2 is therefore
# re-measured on Blackwell too (`dual_bw`), so V1/V2/V4 are all single-architecture; the
# shipped Ada `dual` rows stay in as the architecture cross-check.
ARMS = {
    "vlm_q": ("slim_dualq_u40_v2", "q", "Q heads 13/32 (dualq)", 490_733_568, "baseline_bw"),
    "vlm_m_pm": ("slim_dualm_c1109", "m", "MLP 1109 ch (param-matched)", 490_586_112,
                 "baseline_bw"),
    "vlm_m": ("slim_dualm_u40_v2", "m", "MLP 4898/12288 (dualm)", 2_166_718_464,
              "baseline_bw"),
    "dual_bw": ("slim_dual_u40_v2", "both", "both axes (dual_u40_v2) [BW]", 2_657_452_032,
                "baseline_bw"),
    "dual": ("slim_dual_u40_v2", "both", "both axes (dual_u40_v2) [Ada]", 2_657_452_032,
             "baseline"),
}
CONTRASTS = (("V1 ratio-matched", "vlm_q", "vlm_m"),
             ("V4 param-matched", "vlm_q", "vlm_m_pm"))


def median_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    boots = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(n)]
    return float(np.median(d)), *[float(x) for x in np.percentile(boots, [2.5, 97.5])]


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    mean, mlo, mhi = el.paired_bootstrap_ci(d)
    med, lo, hi = median_ci(d)
    try:
        p = float(wilcoxon(d).pvalue)
    except ValueError:
        p = float("nan")
    return {"n": len(d), "med": med, "lo": lo, "hi": hi, "mean": float(mean),
            "mlo": float(mlo), "mhi": float(mhi), "p": p, "sig": bool(lo > 0 or hi < 0)}


def fmt(s):
    return (f"{s['med']:+.4f} [{s['lo']:+.4f},{s['hi']:+.4f}]{'*' if s['sig'] else ' '} "
            f"mean {s['mean']:+.4f} [{s['mlo']:+.4f},{s['mhi']:+.4f}] p={s['p']:.3g}")


def recipe_gates():
    """V0: the three recipes must decompose dual_u40_v2 exactly."""
    metas, g = {}, {}
    for a, (d, *_) in ARMS.items():
        f = REPO / "outputs" / d / "slim_meta.json"
        if f.exists():
            metas[a] = json.loads(f.read_text())
    for a, m in metas.items():
        exp_ok = all(len(x["q"]) == 16 and len(x["mlp"]) == 8256 for x in m["expert"])
        g[a] = {"removed": m["params"]["removed"], "expected": ARMS[a][3],
                "removed_ok": m["params"]["removed"] == ARMS[a][3],
                "expert_kv_intact": exp_ok and not m["kvonly_layers"],
                "kept_q": sorted({len(x["q"]) for x in m["vlm"]}),
                "kept_mlp": sorted({len(x["mlp"]) for x in m["vlm"]})}
    if {"vlm_q", "vlm_m", "dual"} <= set(metas):
        v = {a: metas[a]["vlm"] for a in ("vlm_q", "vlm_m", "vlm_m_pm", "dual")
             if a in metas}
        g["union_is_dual"] = all(
            v["vlm_q"][i]["q"] == v["dual"][i]["q"]
            and v["vlm_m"][i]["mlp"] == v["dual"][i]["mlp"] for i in range(36))
        # the other axis must be whole in each single-axis arm
        g["other_axis_whole"] = all(
            len(v["vlm_q"][i]["mlp"]) == 12288 and len(v["vlm_m"][i]["q"]) == 32
            for i in range(36))
        if "vlm_m_pm" in v:
            g["pm_nested"] = all(set(v["vlm_m"][i]["mlp"]) <= set(v["vlm_m_pm"][i]["mlp"])
                                 for i in range(36))
    g["pass"] = (all(g[a].get("removed_ok") and g[a].get("expert_kv_intact")
                     for a in metas)
                 and g.get("union_is_dual", False) and g.get("other_axis_whole", False))
    return g


def predicted_mass():
    """Stage 2a: the raw first-order mass each arm removes, per objective.

    dual itself is a per-layer RANK, so it has no cross-axis scale; the comparison is read
    on its two raw ingredients. Selection still uses dual, so this reads the cost of the
    units dual actually picked.
    """
    imp = dict(np.load(REPO / "outputs" / "importance_v2" / "importance.npz"))
    sq = np.maximum(rank_norm(imp["traj_vlm_q"]), rank_norm(imp["coc_vlm_q"]))
    sm = np.maximum(rank_norm(imp["traj_vlm_mlp"]), rank_norm(imp["coc_vlm_mlp"]))
    cuts = {"vlm_q": ("q", 13), "vlm_m": ("m", 4898), "vlm_m_pm": ("m", 1109)}
    out = {"per_unit_median": {}, "per_param_median": {}, "layer_mass_ratio": {},
           "arm_mass": {}}
    for obj in ("traj", "coc"):
        q, m = imp[f"{obj}_vlm_q"], imp[f"{obj}_vlm_mlp"]
        out["per_unit_median"][obj] = float(np.median(q) / np.median(m))
        out["per_param_median"][obj] = float((np.median(q) / P_HEAD)
                                             / (np.median(m) / P_MLPC))
        out["layer_mass_ratio"][obj] = float(q.sum() / m.sum())
        out["arm_mass"][obj] = {}
        for arm, (axis, k) in cuts.items():
            s_, raw = (sq, q) if axis == "q" else (sm, m)
            sel = np.argsort(s_, axis=1)[:, :k]
            mass = float(np.take_along_axis(raw, sel, axis=1).sum())
            params = k * 36 * (P_HEAD if axis == "q" else P_MLPC)
            out["arm_mass"][obj][arm] = {"sum_I": mass, "params": params,
                                         "I_per_param": mass / params}
    out["param_ratio_head_chan"] = P_HEAD / P_MLPC
    return out


def load_set(s):
    rows = {}
    for arm in ("baseline", "baseline_bw", *ARMS):
        spec = pn.ARMS.get(arm, {}).get(s)
        if spec is None:
            continue
        r = pn.load(*spec)
        if r:
            rows[arm] = r
    return rows


def analyse_set(rows):
    out = {"arms": {}, "contrasts": {}, "coc": {}, "baselines": {}}
    for b in ("baseline", "baseline_bw"):
        if b in rows:
            out["baselines"][b] = {
                "n": len(rows[b]),
                "minADE6_mean": float(np.mean([pn.at6(r, "ade_rollout_k")
                                               for r in rows[b].values()])),
                "minFDE6_mean": float(np.mean([pn.at6(r, "fde_rollout_k")
                                               for r in rows[b].values()])),
                "degen": float(np.mean([r["coc_degenerate"] for r in rows[b].values()]))}
    out["baseline"] = out["baselines"].get("baseline_bw") or out["baselines"]["baseline"]
    ade, dade = {}, {}
    for arm in ARMS:
        if arm not in rows or ARMS[arm][4] not in rows:
            continue
        base = rows[ARMS[arm][4]]
        ids = sorted(set(base) & set(rows[arm]))
        ade[arm] = {i: pn.at6(rows[arm][i], "ade_rollout_k") for i in ids}
        dade[arm] = {i: ade[arm][i] - pn.at6(base[i], "ade_rollout_k") for i in ids}
        out["arms"][arm] = {
            "n": len(ids), "removed": ARMS[arm][3], "baseline": ARMS[arm][4],
            "minADE6_mean": float(np.mean(list(ade[arm].values()))),
            "minFDE6_mean": float(np.mean([pn.at6(rows[arm][i], "fde_rollout_k")
                                           for i in ids])),
            "degen": float(np.mean([rows[arm][i]["coc_degenerate"] for i in ids])),
            "d_ade": paired(list(ade[arm].values()),
                            [pn.at6(base[i], "ade_rollout_k") for i in ids]),
            "d_fde": paired([pn.at6(rows[arm][i], "fde_rollout_k") for i in ids],
                            [pn.at6(base[i], "fde_rollout_k") for i in ids]),
            "coc_same_as_baseline": float(np.mean([rows[arm][i]["gen_coc"]
                                                   == base[i]["gen_coc"] for i in ids]))}
    for tag, a, b in CONTRASTS:
        if a in ade and b in ade:
            ids = sorted(set(ade[a]) & set(ade[b]))
            out["contrasts"][f"{tag} {a}-{b}"] = paired([ade[a][i] for i in ids],
                                                        [ade[b][i] for i in ids])
            out["coc"][f"{a}=={b}"] = float(np.mean([rows[a][i]["gen_coc"]
                                                     == rows[b][i]["gen_coc"] for i in ids]))
    # V2 is read on deltas, so it works whichever baseline each arm carries; dual_bw keeps
    # every term on one architecture, the Ada `dual` row is the cross-check
    for which in ("dual_bw", "dual"):
        if {"vlm_q", "vlm_m", which} <= set(dade):
            ids = sorted(set(dade["vlm_q"]) & set(dade["vlm_m"]) & set(dade[which]))
            dq = np.array([dade["vlm_q"][i] for i in ids])
            dm = np.array([dade["vlm_m"][i] for i in ids])
            db = np.array([dade[which][i] for i in ids])
            out["contrasts"][f"V2 {which}-(vq+vm)"] = paired(db, dq + dm)
    if {"dual", "dual_bw"} <= set(dade):
        ids = sorted(set(dade["dual"]) & set(dade["dual_bw"]))
        out["contrasts"]["arch check d(dual_bw)-d(dual_ada)"] = paired(
            [dade["dual_bw"][i] for i in ids], [dade["dual"][i] for i in ids])
    out["_ade"] = ade
    return out


def plot_deltas(res, out):
    sets = [s for s in SETS if s in res]
    fig, axes = plt.subplots(1, len(sets), figsize=(4.6 * len(sets), 3.4), squeeze=False)
    order = ["vlm_q", "vlm_m_pm", "vlm_m", "dual_bw", "dual"]
    for ax, s in zip(axes[0], sets):
        arms = [a for a in order if a in res[s]["arms"]]
        y = np.arange(len(arms))
        for k, a in enumerate(arms):
            d = res[s]["arms"][a]["d_ade"]
            col = C1 if ARMS[a][1] == "q" else (C2 if ARMS[a][1] == "m" else C4)
            ax.plot([d["lo"], d["hi"]], [k, k], color=col, lw=3, solid_capstyle="round")
            ax.plot([d["med"]], [k], "o", color=col, ms=6)
            ax.plot([d["mean"]], [k], "x", color=col, ms=7)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels([ARMS[a][2] for a in arms], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("paired dminADE@6 vs unpruned")
        ax.set_title(s)
    fig.tight_layout()
    fig.savefig(out / "vlm_axis_deltas.png", dpi=150)
    plt.close(fig)


def plot_params(res, mass, out):
    """Removed parameters against measured cost, with the first-order prediction beside."""
    if "indist" not in res:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    arms = [a for a in ("vlm_q", "vlm_m_pm", "vlm_m") if a in res["indist"]["arms"]]
    for a in arms:
        d = res["indist"]["arms"][a]["d_ade"]
        col = C1 if ARMS[a][1] == "q" else C2
        axes[0].errorbar(ARMS[a][3] / 1e6, d["med"],
                         yerr=[[d["med"] - d["lo"]], [d["hi"] - d["med"]]],
                         fmt="o", color=col, capsize=3)
        axes[0].annotate(ARMS[a][2].split(" (")[0], (ARMS[a][3] / 1e6, d["med"]),
                         textcoords="offset points", xytext=(6, 4), fontsize=8)
        m = mass["arm_mass"]["traj"].get(a)
        if m:
            axes[1].plot(m["sum_I"], d["med"], "o", color=col)
            axes[1].annotate(ARMS[a][2].split(" (")[0], (m["sum_I"], d["med"]),
                             textcoords="offset points", xytext=(6, 4), fontsize=8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("removed parameters (M, log)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"first-order mass removed, $\sum_S I_{traj}$ (log)")
    for ax in axes:
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_ylabel("paired dminADE@6 (median)")
    axes[0].set_title("cost vs parameters")
    axes[1].set_title("cost vs first-order prediction")
    fig.tight_layout()
    fig.savefig(out / "vlm_axis_params.png", dpi=150)
    plt.close(fig)


def render_template(metrics, probe, template, out_path):
    """Fill {{...}} tokens from the two metrics.json files, so no number is typed by hand.

    Grammar (S = indist|test|oodval, A = arm key, T = contrast tag V1|V4|V2|arch):
      {{S.A.med|mean|p|ade|fde|n|coc|degen}}      one arm on one set
      {{S.base.ade|fde|n}} / {{S.basebw....}}     the two baselines
      {{S.C.T.med|mean|p}}                        a contrast
      {{mass.OBJ.A.sum_I|params|I_per_param}}     first-order mass an arm removes
      {{massx.OBJ.per_param|per_unit|layer}}      the cross-axis ratios
      {{raw.TOWER.OBJ.FIELD}}                     probe-side raw table (both towers)
      {{s0.FIELD}} {{s1.TAG.beta|lo|hi|r|sign}} {{s1r.ratio|lo|hi}} {{s1b....}}
      {{s2.NAME.FIELD}} {{s4.TAG.beta}}
    A missing token raises rather than leaving a blank cell.
    """
    import re

    def ci(d, key):
        if key == "med":
            return (f"{d['med']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"
                    + ("*" if d["sig"] else ""))
        if key == "mean":
            return f"{d['mean']:+.4f} [{d['mlo']:+.4f}, {d['mhi']:+.4f}]"
        if key == "p":
            return f"{d['p']:.2g}"
        raise KeyError(key)

    def lookup(tok):
        p = tok.split(".")
        if p[0] == "mass":
            v = metrics["mass"]["arm_mass"][p[1]][p[2]]
            return ({"sum_I": f"{v['sum_I']:.4g}", "params": f"{v['params']:,}",
                     "I_per_param": f"{v['I_per_param']:.3g}"})[p[3]]
        if p[0] == "massx":
            return {"per_param": f"{metrics['mass']['per_param_median'][p[1]]:.2f}",
                    "per_unit": f"{metrics['mass']['per_unit_median'][p[1]]:.1f}",
                    "layer": f"{metrics['mass']['layer_mass_ratio'][p[1]]:.3f}"}[p[2]]
        if p[0] == "raw":
            v = probe["raw_cross_axis"][p[1]]["obj"][p[2]]
            return (f"{v[p[3]]:.2f}" if "ratio" in p[3] and "unit" not in p[3]
                    else f"{v[p[3]]:.4g}" if p[3] not in ("bottom50_share_q",
                                                          "bottom50_share_mlp")
                    else f"{100 * v[p[3]]:.2f}%")
        if p[0] == "s0":
            return f"{probe['s0'][p[1]]:.2e}" if isinstance(probe["s0"][p[1]], float) \
                else str(probe["s0"][p[1]])
        if p[0] in ("s1", "s4"):
            f = probe["s1_fits" if p[0] == "s1" else "s4_fits_sumabs"][p[1]]
            return {"beta": f"{f['beta']:+.3f}", "lo": f"{f['lo']:+.3f}",
                    "hi": f"{f['hi']:+.3f}", "r": f"{f['pearson']:+.3f}",
                    "sign": f"{100 * f['sign_agree']:.0f}%", "n": f"{f['n']:,}",
                    "ci": (f"{f['beta']:+.3f} [{f['lo']:+.3f}, {f['hi']:+.3f}]")}[p[2]]
        if p[0] in ("s1r", "s1b"):
            f = probe["s1_ratio_unit" if p[0] == "s1r" else "s1b_ratio_parammatched"]
            return {"ratio": f"{f['ratio']:.3f}", "lo": f"{f['lo']:.3f}",
                    "hi": f"{f['hi']:.3f}",
                    "ci": f"{f['ratio']:.3f} [{f['lo']:.3f}, {f['hi']:.3f}]"}[p[1]]
        if p[0] == "s2":
            v = probe["s2"][p[1]]
            return {"pred": f"{v['predicted']:+.3e}", "meas": f"{v['all_layers']:+.3e}",
                    "ratio": f"{v['measured_over_predicted']:+.2f}",
                    "sum": f"{v['sum_of_layers']:+.3e}",
                    "add": f"{v['layer_additivity']:+.2f}",
                    "ade": f"{v['dminADE']:+.4f}"}[p[2]]
        S = metrics["sets"][p[0]]
        if p[1] in ("base", "basebw"):
            b = S["baselines"]["baseline" if p[1] == "base" else "baseline_bw"]
            return (f"{b['minADE6_mean']:.4f}" if p[2] == "ade" else
                    f"{b['minFDE6_mean']:.4f}" if p[2] == "fde" else str(b["n"]))
        if p[1] == "C":
            key = next(k for k in S["contrasts"] if k.split()[0] == p[2])
            return ci(S["contrasts"][key], p[3])
        a = S["arms"][p[1]]
        if p[2] in ("med", "mean", "p"):
            return ci(a["d_ade"], p[2])
        return {"ade": f"{a['minADE6_mean']:.4f}", "fde": f"{a['minFDE6_mean']:.4f}",
                "n": str(a["n"]), "coc": f"{a['coc_same_as_baseline']:.3f}",
                "degen": f"{a['degen']:.3f}",
                "removed": f"{a['removed']:,}"}[p[2]]

    text = Path(template).read_text()
    text = re.sub(r"\{\{([^}]+)\}\}", lambda m: lookup(m.group(1).strip()), text)
    Path(out_path).write_text(text)
    print(f"rendered {template} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="vlm_axis_analysis")
    ap.add_argument("--probe-out", default="axis_taylor_analysis",
                    help="analyze_taylor_probe.py's output, for the Stage 1 tokens")
    ap.add_argument("--template", default=None,
                    help="report template; writes <stem>_filled.html beside it")
    args = ap.parse_args()
    out = REPO / "outputs" / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)

    g0 = recipe_gates()
    mass = predicted_mass()
    res = {}
    for s in SETS:
        rows = load_set(s)
        if "baseline" in rows and len(rows) > 1:
            res[s] = analyse_set(rows)

    lines = ["VLM axis decomposition -- Q heads vs MLP channels, dual score, u40 budget",
             "", "V0 recipe integrity:"]
    for a in ARMS:
        if a in g0:
            lines.append(f"  {a:10s} removed {g0[a]['removed']:>13,} "
                         f"(expect {g0[a]['expected']:>13,}) "
                         f"{'OK' if g0[a]['removed_ok'] else 'MISMATCH'}  "
                         f"expert/KV intact {g0[a]['expert_kv_intact']}  "
                         f"kept q {g0[a]['kept_q']} mlp {g0[a]['kept_mlp']}")
    lines += [f"  union(vq,vm) == dual: {g0.get('union_is_dual')}",
              f"  other axis whole in each arm: {g0.get('other_axis_whole')}",
              f"  dualm nested in dualm_c1109: {g0.get('pm_nested')}",
              f"  V0 {'PASS' if g0['pass'] else 'FAIL'}", ""]

    lines += ["Stage 2a -- raw first-order mass (importance_v2), the prediction:",
              f"  head/channel parameter ratio {mass['param_ratio_head_chan']:.1f}x"]
    for obj in ("traj", "coc"):
        lines.append(f"  [{obj}] per-unit median Q/MLP {mass['per_unit_median'][obj]:.1f}x "
                     f"| PER PARAMETER {mass['per_param_median'][obj]:.2f}x "
                     f"| layer mass Q/MLP {mass['layer_mass_ratio'][obj]:.3f}")
        for a, v in mass["arm_mass"][obj].items():
            lines.append(f"      {a:9s} params {v['params']:>13,}  sum_I {v['sum_I']:.4e}"
                         f"  I/param {v['I_per_param']:.3e}")
    lines.append("")

    for s in SETS:
        if s not in res:
            continue
        r = res[s]
        lines.append(f"== {s}")
        for b, v in r["baselines"].items():
            lines.append(f"  {b:12s} n={v['n']:3d} minADE@6 {v['minADE6_mean']:.4f} "
                         f"minFDE@6 {v['minFDE6_mean']:.4f} degen {v['degen']:.3f}")
        for a in ("vlm_q", "vlm_m_pm", "vlm_m", "dual_bw", "dual"):
            if a in r["arms"]:
                d = r["arms"][a]
                lines.append(f"  {a:9s} n={d['n']:3d} removed {d['removed']:>13,} "
                             f"minADE@6 {d['minADE6_mean']:.4f}  vs {d['baseline']}  "
                             f"d {fmt(d['d_ade'])}")
                lines.append(f"            degen {d['degen']:.3f}  "
                             f"gen_coc == baseline {d['coc_same_as_baseline']:.3f}")
        for k, v in r["contrasts"].items():
            lines.append(f"  {k:28s} {fmt(v)}")
        lines.append("")

    if "indist" in res:
        ranked = sorted(((a, res["indist"]["arms"][a]["d_ade"]["med"])
                         for a in ("vlm_q", "vlm_m_pm", "vlm_m")
                         if a in res["indist"]["arms"]), key=lambda x: -x[1])
        pred = sorted(((a, mass["arm_mass"]["traj"][a]["sum_I"])
                       for a, _ in ranked), key=lambda x: -x[1])
        agree = [a for a, _ in ranked] == [a for a, _ in pred]
        lines += [(f"V5 ordering  measured {[a for a, _ in ranked]} "
                   f"vs predicted {[a for a, _ in pred]} "
                   f"{'AGREE' if agree else 'DISAGREE'}"), ""]

    plot_deltas(res, out / "plots")
    plot_params(res, mass, out / "plots")
    for r in res.values():
        r.pop("_ade", None)
    metrics = {"v0": g0, "mass": mass, "sets": res}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("saved ->", out)

    if args.template:
        probe = json.loads(
            (REPO / "outputs" / args.probe_out / "metrics.json").read_text())
        t = Path(args.template)
        render_template(metrics, probe, t, t.with_name(t.stem + "_filled.html"))


if __name__ == "__main__":
    main()
