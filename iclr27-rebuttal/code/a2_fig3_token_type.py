"""Paper Figure 3 (importance decomposed by token type), Alpamayo 1.5 vs Alpamayo 2 Super.

  (a)    ceiling-corrected within-layer rank agreement between I_FM (= I_traj) and I_CE (= I_CoC):
         all positions pooled (black) and both scores restricted to one token type (colours)
  (b, c) additive share of each layer's I_FM / I_CE earned at vision / prompt-text / generated-CoC /
         ego-history positions

Data: typed per-clip gate gradients (N=100 calib_100 clips, 5 position types: vision, ego history,
prompt text, generated CoC, sink).
  A1.5  outputs/gradanat_v1/anatomy_perclip_{q,mlp}.npz (the paper's source)
  A2    alpamayo2-distill/outputs/a2_importance_v1/anatomy_perclip_{q,mlp}.npz (2026-09-27)

(a) uses split_half()/corrected() of paper-analysis/scripts/paper_same_position.py (copied: that file
    runs on import): 50/50 clip splits (Q heads 50, MLP 10, as the paper), corrected = cross /
    sqrt(self_FM * self_CE), NaN where a self-agreement is <= 0.2. One rng (seed 0) per (model, axis),
    series order All, Vision, Prompt text, CoC, Ego history.
(b, c) use the additive decomposition of paper_figs_a2.token_share (per clip, each type's signed
    gradient times the sign of the unit's total, summed over units, averaged over clips).
Plotting helpers and style come from alpamayo2-distill/experiments/importance/paper_figs_a2.py.

Plan: iclr27-rebuttal/plans/2026-11-07_a2-analysis-replication.md (section added 2026-11-08).

Usage:
  .venv/bin/python iclr27-rebuttal/code/a2_fig3_token_type.py [--exp-id iclr27-rebuttal/a2_fig3_token_type]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, "/home/cvlab21/project/chan/alpamayo2-distill/experiments/importance")
import paper_figs_a2 as pfa
from paper_figs_a2 import (
    COC,
    HIST,
    MODELS,
    PT,
    PURPLE,
    VIS,
    depth_marks,
    export_pair,
    load_typed,
    pair,
    ps,
    rho_layers,
)

warnings.filterwarnings("ignore", "Mean of empty slice")
N_SPLIT = {"q": 50, "mlp": 10}
TYPES = (("Vision", VIS), ("Prompt text", PT), ("CoC", COC), ("Ego history", HIST))
COLORS = {"All": "black", "Vision": ps.BLUE, "Prompt text": ps.YELLOW, "CoC": PURPLE, "Ego history": ps.BRICK}
LATE = {"a15": range(22, 35), "a2": range(39, 63)}
# token-share bands: the paper's 6-17 / 27-34 and the 2026-09-27 report's relative-depth twins
SHARE_BANDS = {"a15": {"lo": range(6, 18), "hi": range(27, 35)}, "a2": {"lo": range(11, 31), "hi": range(48, 61)}}
# G4/G5: late-band mean of corrected Q-head agreement. A1.5: paper text (analysis_v2) /
# 2026-09-27 report; A2: 2026-09-27 report. Different rng from the stored series -> tolerance 0.03.
G_CORR = {"a15": {"All": 0.38, "Vision": 0.79, "Prompt text": 0.81, "CoC": 0.41, "Ego history": 0.94},
          "a2": {"All": 0.58, "Vision": 0.71, "Prompt text": 0.73, "CoC": 0.56, "Ego history": 0.92}}
# G6: raw late means printed by paper_figs_a2 (deterministic -> tolerance 0.006)
G_RAW = {"a15": {"All": 0.33, "Vision": 0.72, "Prompt text": 0.75, "CoC": 0.36, "Ego history": 0.86},
         "a2": {"All": 0.54, "Vision": 0.66, "Prompt text": 0.69, "CoC": 0.50, "Ego history": 0.83}}
# G7: A1.5 token shares quoted in the paper text (analysis_v2): (axis, objective, band, types) -> value
G_SHARE = [("mlp", "traj", "lo", ("vis",), 0.90), ("mlp", "coc", "lo", ("vis",), 0.75),
           ("mlp", "coc", "lo", ("pt", "coc"), 0.24), ("mlp", "traj", "lo", ("pt", "coc"), 0.07),
           ("mlp", "coc", "hi", ("pt", "coc"), 0.98), ("mlp", "traj", "hi", ("vis",), 0.26),
           ("mlp", "traj", "hi", ("pt", "coc"), 0.66), ("mlp", "traj", "hi", ("hist",), 0.08),
           ("q", "traj", "hi", ("vis",), 0.35), ("q", "traj", "hi", ("pt", "coc"), 0.49),
           ("q", "traj", "hi", ("hist",), 0.16)]
TYPE_IDX = {"vis": VIS, "hist": HIST, "pt": PT, "coc": COC}


def split_half(t, c, n_split, rng):
    """paper_same_position.split_half: t, c (N, L, U) -> mean self_FM, self_CE, cross over splits, each (L,)"""
    n = len(t)
    st, sc_, cr = [], [], []
    for _ in range(n_split):
        p = rng.permutation(n)
        a, b = p[:n // 2], p[n // 2:]
        ta, tb, ca, cb = t[a].mean(0), t[b].mean(0), c[a].mean(0), c[b].mean(0)
        st.append(rho_layers(ta, tb))
        sc_.append(rho_layers(ca, cb))
        cr.append(0.5 * (rho_layers(ta, cb) + rho_layers(tb, ca)))
    return np.nanmean(st, 0), np.nanmean(sc_, 0), np.nanmean(cr, 0)


def corrected(st, sc_, cr):
    """paper_same_position.corrected"""
    ok = (st > 0.2) & (sc_ > 0.2)
    return np.where(ok, cr / np.sqrt(np.where(ok, st * sc_, 1.0)), np.nan)


def agreement(fm, ce, last, n_split):
    """fm, ce: (N, 5, L, U) signed typed gradients -> {series: (corrected (last,), raw (last,), self_fm, self_ce)}"""
    rng = np.random.default_rng(0)
    out = {}
    for label, ty in (("All", None),) + TYPES:
        t = np.abs(fm.sum(1) if ty is None else fm[:, ty])[:, :last]  # (N, last, U)
        c = np.abs(ce.sum(1) if ty is None else ce[:, ty])[:, :last]
        st, sc_, cr = split_half(t, c, n_split, rng)
        out[label] = {"corrected": corrected(st, sc_, cr), "raw": rho_layers(t.mean(0), c.mean(0)),
                      "self_fm": st, "self_ce": sc_}
    return out


def shares(arr):
    """paper_figs_a2.token_share: arr (N, 5, L, U) -> (5, L) additive share of the layer's score by type"""
    part = np.zeros(arr.shape[1:3])
    for i in range(len(arr)):
        g = arr[i].astype(np.float64)
        part += (g * np.sign(g.sum(0, keepdims=True))).sum(-1)
    part /= len(arr)
    tot = part.sum(0)
    return np.nan_to_num(part / np.where(tot > 0, tot, np.nan))


def plot_agreement(ax, m, axis, series, last):
    depth_marks(ax, m, axis)
    L = np.arange(last)
    for label, s in series.items():
        ax.plot(L, s["corrected"], color=COLORS[label], lw=1.8 if label == "All" else 1.3, label=label)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylim(-0.25, 1.05)
    ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75, 1.0])


def plot_share(ax, m, axis, s, n_layers):
    depth_marks(ax, m, axis)
    L = np.arange(n_layers)
    for label, ty in TYPES:
        ax.plot(L, s[ty][:n_layers], color=COLORS[label], lw=1.5, label=label)
    ax.set_ylim(0, 1.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="iclr27-rebuttal/a2_fig3_token_type")
    args = ap.parse_args()
    out = REPO / "outputs" / args.exp_id
    pfa.OUT = out / "figs"  # export_pair writes here

    res = {}
    for key, m in MODELS.items():
        for axis in ("q", "mlp"):
            fm, ce = load_typed(m, axis)  # (N, 5, L, U)
            last = m["n"] - 1
            res[(key, axis)] = {"agree": agreement(fm, ce, last, N_SPLIT[axis]),
                                "share": {"traj": shares(fm), "coc": shares(ce)}}
            del fm, ce
            print(f"  {key} {axis} done", flush=True)

    # gates
    gates = {"G4_G5_corrected_late_q": {}, "G6_raw_late_q": {}, "G7_a15_share_paper_text": []}
    for key in MODELS:
        ag = res[(key, "q")]["agree"]
        late = list(LATE[key])
        for label in G_CORR[key]:
            got_c = float(np.nanmean(ag[label]["corrected"][late]))
            got_r = float(np.nanmean(ag[label]["raw"][late]))
            gates["G4_G5_corrected_late_q"][f"{key}/{label}"] = {
                "expected": G_CORR[key][label], "got": round(got_c, 3), "ok": abs(got_c - G_CORR[key][label]) <= 0.03}
            gates["G6_raw_late_q"][f"{key}/{label}"] = {
                "expected": G_RAW[key][label], "got": round(got_r, 3), "ok": abs(got_r - G_RAW[key][label]) <= 0.006}
    for axis, obj, band, tys, exp in G_SHARE:
        s = res[("a15", axis)]["share"][obj]
        got = float(sum(s[TYPE_IDX[t]][list(SHARE_BANDS["a15"][band])].mean() for t in tys))
        gates["G7_a15_share_paper_text"].append({"axis": axis, "obj": obj, "band": band, "types": tys,
                                                 "expected": exp, "got": round(got, 3), "ok": abs(got - exp) <= 0.006})
    ok = {g: all(v["ok"] for v in (d.values() if isinstance(d, dict) else d)) for g, d in gates.items()}

    # figures: paper-style pairs (A1.5 left, A2 right) and one 2 x 3 overview per axis
    unit = {"q": "q_head", "mlp": "mlp"}
    for axis in ("q", "mlp"):
        fig, axes = pair()
        for ax, key in zip(axes, MODELS, strict=True):
            plot_agreement(ax, MODELS[key], axis, res[(key, axis)]["agree"], MODELS[key]["n"] - 1)
        axes[0].set_ylabel(r"Corrected $\rho$")
        export_pair(fig, axes, f"fig3a_same_position_corrected_{unit[axis]}", 5)
        for obj in ("traj", "coc"):
            fig, axes = pair()
            for ax, key in zip(axes, MODELS, strict=True):
                n_layers = MODELS[key]["n"] - 1 if obj == "traj" else MODELS[key]["n"]
                plot_share(ax, MODELS[key], axis, res[(key, axis)]["share"][obj], n_layers)
            axes[0].set_ylabel("Share of layer importance")
            export_pair(fig, axes, f"fig3{'b' if obj == 'traj' else 'c'}_token_share_{obj}_{unit[axis]}", 4)

        fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.4))
        for row, key in enumerate(MODELS):
            m, r = MODELS[key], res[(key, axis)]
            plot_agreement(axes[row, 0], m, axis, r["agree"], m["n"] - 1)
            plot_share(axes[row, 1], m, axis, r["share"]["traj"], m["n"] - 1)
            plot_share(axes[row, 2], m, axis, r["share"]["coc"], m["n"])
            axes[row, 0].set_ylabel(f"{m['name']}\nCorrected $\\rho$", fontsize=ps.TEXT)
            axes[row, 1].set_ylabel("Share of layer importance", fontsize=ps.TEXT)
        for ax, t in zip(axes[0], ("(a) Rank correlation", r"(b) $I_{\mathrm{FM}}$ by token type",
                                   r"(c) $I_{\mathrm{CE}}$ by token type"), strict=True):
            ax.set_title(t, fontsize=ps.TEXT)
        for ax in axes.ravel():
            for line in ax.lines:
                if line.get_gid() != "step":
                    line.set_linestyle("-")
            ax.tick_params(labelsize=ps.TEXT - 1)
            ax.xaxis.label.set_fontsize(ps.TEXT)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=ps.TEXT)
        fig.text(0.5, 0.012, f"{'Q heads' if axis == 'q' else 'MLP channels'}; grey = late band, "
                 "dotted = I_traj/I_CoC step", ha="center", va="bottom", fontsize=ps.TEXT - 1, color="0.35")
        fig.tight_layout(rect=(0, 0.05, 1, 0.94))
        for ext_ in ("png", "pdf"):
            fig.savefig(pfa.OUT / f"fig3_overview_{unit[axis]}.{ext_}", dpi=200)
        plt.close(fig)

    # numbers
    band_share = {}
    for (key, axis), r in res.items():
        for obj in ("traj", "coc"):
            for band, ls in SHARE_BANDS[key].items():
                s = r["share"][obj][:, list(ls)].mean(1)
                band_share[f"{key}/{axis}/{obj}/{band}"] = {
                    "vision": float(s[VIS]), "prompt": float(s[PT]), "coc": float(s[COC]), "hist": float(s[HIST])}
    late_agree = {f"{key}/{axis}": {label: {"corrected": float(np.nanmean(s["corrected"][list(LATE[key])])),
                                            "raw": float(np.nanmean(s["raw"][list(LATE[key])])),
                                            "nan_layers": int(np.isnan(s["corrected"][list(LATE[key])]).sum())}
                                    for label, s in r["agree"].items()} for (key, axis), r in res.items()}
    trunk_agree = {f"{key}/{axis}": {label: float(np.nanmean(s["corrected"][: LATE[key].start]))
                                     for label, s in r["agree"].items()} for (key, axis), r in res.items()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "sources": {k: {"q": str(m["anat_q"]), "mlp": str(m["anat_mlp"])} for k, m in MODELS.items()},
        "n_split": N_SPLIT, "seed": 0, "late_band": {k: [v.start, v.stop - 1] for k, v in LATE.items()},
        "share_bands": {k: {b: [r.start, r.stop - 1] for b, r in v.items()} for k, v in SHARE_BANDS.items()},
        "step_tau": {k: m["tau"] for k, m in MODELS.items()}}, indent=2))
    (out / "metrics.json").write_text(json.dumps({
        "gates": gates, "gates_ok": ok, "late_agreement": late_agree, "trunk_agreement_corrected": trunk_agree,
        "band_share": band_share,
        "per_layer": {f"{key}/{axis}": {"agreement": {label: {k: v.tolist() for k, v in s.items()}
                                                      for label, s in r["agree"].items()},
                                        "share": {obj: v.tolist() for obj, v in r["share"].items()}}
                      for (key, axis), r in res.items()}}, indent=2, default=float))

    L = ["Paper Figure 3 on Alpamayo 1.5 and Alpamayo 2 Super", ""]
    for g, v in ok.items():
        L.append(f"{g}: {'PASS' if v else 'FAIL'}")
    for g in ("G4_G5_corrected_late_q", "G6_raw_late_q"):
        L += [f"   {k}: expected {v['expected']} got {v['got']}" for k, v in gates[g].items() if not v["ok"]]
    L += [f"   G7 {v['axis']}/{v['obj']}/{v['band']}/{'+'.join(v['types'])}: expected {v['expected']} got {v['got']}"
          for v in gates["G7_a15_share_paper_text"] if not v["ok"]]
    for axis in ("q", "mlp"):
        L += ["", f"(a) agreement, {'Q heads' if axis == 'q' else 'MLP channels'}: band means of corrected rho (raw)",
              f"{'model':<8}{'band':<8}" + "".join(f"{lab:>18}" for lab in COLORS)]
        for key in MODELS:
            k = f"{key}/{axis}"
            L.append(f"{key:<8}{'trunk':<8}" + "".join(f"{trunk_agree[k][lab]:>18.2f}" for lab in COLORS))
            L.append(f"{key:<8}{'late':<8}" + "".join(
                f"{late_agree[k][lab]['corrected']:>11.2f} ({late_agree[k][lab]['raw']:.2f})" for lab in COLORS))
        L += ["", (f"(b, c) token shares, {'Q heads' if axis == 'q' else 'MLP channels'} "
              "(lo = A1.5 6-17 / A2 11-30, hi = A1.5 27-34 / A2 48-60)"),
              f"{'model':<8}{'obj':<6}{'band':<5}{'vision':>8}{'prompt':>8}{'CoC':>8}{'ego':>8}"]
        for key in MODELS:
            for obj in ("traj", "coc"):
                for band in ("lo", "hi"):
                    v = band_share[f"{key}/{axis}/{obj}/{band}"]
                    L.append(f"{key:<8}{obj:<6}{band:<5}{v['vision']:>8.2f}{v['prompt']:>8.2f}{v['coc']:>8.2f}"
                             f"{v['hist']:>8.2f}")
    (out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
