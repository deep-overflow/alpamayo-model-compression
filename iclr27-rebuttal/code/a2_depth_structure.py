"""VLM depth structure of first-order Taylor importance, Alpamayo 1.5 vs Alpamayo 2 Super.

Re-runs the cvpr27 depth analyses (cvpr27/code/{analyze_layer_concentration,plot_depth_structure}.py)
on both models with one code path, from stored scores (CPU only):

  A1.5  outputs/importance_v2                                   (36 layers, MLP 12,288, Q 32)
  A2    alpamayo2-distill/outputs/a2_importance_v1 (2026-09-27)  (64 layers, MLP 25,600, Q 64)

Both runs: same estimator, same calib_100 clips; importance.npz = mean of importance_perclip.npz.

Per layer and objective (I_traj = FM loss, I_CoC = CoC NLL):
  gini, b50, n50, n90, cut   layer_stats() of analyze_layer_concentration.py (cut = bottom 39.86%)
  overlap                    traj/CoC top-k shared, k = top 0.26% of units (32 / 67)
  rho                        within-layer Spearman(I_traj, I_CoC), Finding 1's measure
  stability                  split-half: share of one half's top-k (0.52%: 64 / 133) that the other half
                             also ranks top-k; 100 clips -> 50/50, 20 random splits (seed 0)

The last layer is left out of I_traj readouts and of rho (I_traj is structurally zero there).
Plan: iclr27-rebuttal/plans/2026-11-07_a2-analysis-replication.md

Usage:
  .venv/bin/python iclr27-rebuttal/code/a2_depth_structure.py [--exp-id iclr27-rebuttal/a2_depth_structure]
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
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "cvpr27" / "code"))
from analyze_layer_concentration import GRID, INK, MUTED, RATIO, layer_stats

C_TRAJ, C_COC = "#2a78d6", "#008300"
TOP_OVERLAP_FRAC = 32 / 12288
TOP_STAB_FRAC = 64 / 12288
N_SPLITS = 20
EARLY_FRAC = 0.15  # "early" = layers 1 .. floor(0.15 n)
GINI_HI, MID_GINI_MAX, OVERLAP_MIN = 0.85, 0.5, 0.6
A2_DIR = Path("/home/cvlab21/project/chan/alpamayo2-distill/outputs/a2_importance_v1")
MODELS = {
    "a15": {"name": "Alpamayo 1.5", "dir": REPO / "outputs" / "importance_v2", "n": 36,
            "bands": {"0": [0], "1-4": list(range(1, 5)), "5": [5], "6-21": list(range(6, 22)),
                      "22-34": list(range(22, 35))},
            "early": "1-4", "mid": "6-21", "trunk": range(22), "late": range(22, 35),
            "split": 21.5, "tau": 23},
    "a2": {"name": "Alpamayo 2 Super", "dir": A2_DIR, "n": 64,
           "bands": {"0": [0], "1-7": list(range(1, 8)), "8-9": [8, 9], "10-38": list(range(10, 39)),
                     "39-62": list(range(39, 63))},
           "early": "1-7", "mid": "10-38", "trunk": range(39), "late": range(39, 63),
           "split": 38.5, "tau": 50},
}
# G2: band-mean rho of the 2026-09-27 report (alpamayo2-distill/reports/2026-09-27_a2-dual-importance.md)
G2_EXPECT = {("a15", "q"): (0.782, 0.335), ("a15", "mlp"): (0.753, 0.339),
             ("a2", "q"): (0.873, 0.537), ("a2", "mlp"): (0.687, 0.378)}
# G1: A1.5 layers 1-4 n50 per objective, outputs/cvpr27/layer_concentration/summary.txt
G1_N50 = {"traj": [86, 40, 48, 151], "coc": [92, 45, 79, 195]}
warnings.filterwarnings("ignore", "All-NaN slice")  # last layer's I_traj is NaN by design


def top_overlap(a, b, k):
    """a, b: (n_units,) scores -> share of a's top-k also in b's top-k"""
    return len(set(np.argpartition(a, -k)[-k:]) & set(np.argpartition(b, -k)[-k:])) / k


def longest_run(layers):
    """longest run of consecutive layer indices"""
    best, cur = [], []
    for li in sorted(layers):
        cur = cur + [li] if cur and li == cur[-1] + 1 else [li]
        best = cur if len(cur) > len(best) else best
    return best


def analyse(key, m):
    imp = np.load(m["dir"] / "importance.npz")
    n, last = m["n"], m["n"] - 1
    res = {"rho": {}, "stats": {}}
    for axis in ("mlp", "q"):
        t = imp[f"traj_vlm_{axis}"].astype(np.float64)  # (n, units)
        c = imp[f"coc_vlm_{axis}"].astype(np.float64)
        res["stats"][axis] = {"traj": [None if li == last else layer_stats(t[li]) for li in range(n)],
                              "coc": [layer_stats(c[li]) for li in range(n)]}
        res["rho"][axis] = [None if li == last else float(spearmanr(t[li], c[li])[0]) for li in range(n)]
        for band in ("trunk", "late"):
            res.setdefault("rho_band_mean", {})[f"{axis}_{band}"] = float(np.mean(
                [res["rho"][axis][li] for li in m[band]]))
    t, c = imp["traj_vlm_mlp"], imp["coc_vlm_mlp"]  # (n, units)
    units = t.shape[1]
    k_ov, k_st = round(TOP_OVERLAP_FRAC * units), round(TOP_STAB_FRAC * units)
    res["k_overlap"], res["k_stability"], res["units"] = k_ov, k_st, units
    res["overlap"] = [None if li == last else top_overlap(t[li], c[li], k_ov) for li in range(n)]
    res["pooled_gini"] = {o: layer_stats(x[:last].ravel().astype(np.float64))["gini"]
                          for o, x in (("traj", t), ("coc", c))}

    pc = np.load(m["dir"] / "importance_perclip.npz")
    n_clips = pc["traj_vlm_mlp"].shape[0]
    rng = np.random.default_rng(0)
    perms = [rng.permutation(n_clips) for _ in range(N_SPLITS)]
    res["stability"] = {}
    for o in ("traj", "coc"):
        x = pc[f"{o}_vlm_mlp"]  # (n_clips, n, units)
        st = np.full((N_SPLITS, n), np.nan)
        for s, p in enumerate(perms):
            ma, mb = x[p[: n_clips // 2]].mean(0), x[p[n_clips // 2:]].mean(0)  # (n, units)
            for li in range(n):
                if not (o == "traj" and li == last):
                    st[s, li] = top_overlap(ma[li], mb[li], k_st)
        res["stability"][o] = st
        print(f"  {key} split-half {o} done", flush=True)
    return res


def per_layer(res, axis, obj, field):
    return np.array([np.nan if r is None else r[field] for r in res["stats"][axis][obj]])


def band_table(m, res):
    rows = {}
    for b, ls in m["bands"].items():
        g = {o: per_layer(res, "mlp", o, "gini")[ls] for o in ("traj", "coc")}
        rows[b] = {"layers": f"{ls[0]}-{ls[-1]}" if len(ls) > 1 else str(ls[0]),
                   "gini_traj": float(np.nanmedian(g["traj"])), "gini_coc": float(np.nanmedian(g["coc"])),
                   "n50_traj": float(np.nanmedian(per_layer(res, "mlp", "traj", "n50")[ls])),
                   "n50_coc": float(np.nanmedian(per_layer(res, "mlp", "coc", "n50")[ls])),
                   "cut_traj": float(np.nanmedian(per_layer(res, "mlp", "traj", "cut")[ls])),
                   "cut_coc": float(np.nanmedian(per_layer(res, "mlp", "coc", "cut")[ls])),
                   "overlap": float(np.nanmedian([np.nan if res["overlap"][li] is None else res["overlap"][li]
                                                  for li in ls])),
                   "rho": float(np.nanmedian([np.nan if res["rho"]["mlp"][li] is None else res["rho"]["mlp"][li]
                                              for li in ls])),
                   "stab_traj": float(np.nanmedian(res["stability"]["traj"][:, ls])),
                   "stab_coc": float(np.nanmedian(res["stability"]["coc"][:, ls]))}
    return rows


def criteria(m, res, bands):
    n = m["n"]
    gt, gc = per_layer(res, "mlp", "traj", "gini"), per_layer(res, "mlp", "coc", "gini")
    early_layers = range(1, int(EARLY_FRAC * n) + 1)
    hi = [li for li in early_layers if gt[li] >= GINI_HI and gc[li] >= GINI_HI]
    run = longest_run(hi)
    mid = bands[m["mid"]]
    c1 = len(run) >= 2 and mid["gini_traj"] <= MID_GINI_MAX and mid["gini_coc"] <= MID_GINI_MAX
    ov = float(np.median([res["overlap"][li] for li in run])) if run else float("nan")
    c2 = bool(run) and ov >= OVERLAP_MIN and ov > mid["overlap"]
    st = {o: float(np.nanmedian(res["stability"][o][:, run])) if run else float("nan") for o in ("traj", "coc")}
    c3 = bool(run) and all(st[o] > mid[f"stab_{o}"] for o in ("traj", "coc"))
    return {"early_layers_searched": [early_layers.start, early_layers.stop - 1], "high_gini_layers": hi,
            "run": run, "C1_concentration": bool(c1), "C2_agreement": bool(c2), "C3_stability": bool(c3),
            "run_overlap": ov, "run_stability": st}


def gates(results):
    a = results["a15"]
    g1 = {"gini_1_4": [min(per_layer(a, "mlp", o, "gini")[1:5].min() for o in ("traj", "coc")),
                       max(per_layer(a, "mlp", o, "gini")[1:5].max() for o in ("traj", "coc"))],
          "n50_1_4": {o: per_layer(a, "mlp", o, "n50")[1:5].astype(int).tolist() for o in ("traj", "coc")},
          "cut_1_4": [min(per_layer(a, "mlp", o, "cut")[1:5].min() for o in ("traj", "coc")),
                      max(per_layer(a, "mlp", o, "cut")[1:5].max() for o in ("traj", "coc"))],
          "top32_overlap_1_4": [round(a["overlap"][li] * a["k_overlap"]) for li in range(1, 5)]}
    ok1 = (0.885 <= g1["gini_1_4"][0] and g1["gini_1_4"][1] < 0.975 and g1["n50_1_4"] == G1_N50
           and 0.0005 <= g1["cut_1_4"][0] and g1["cut_1_4"][1] < 0.0185
           and min(g1["top32_overlap_1_4"]) == 23 and max(g1["top32_overlap_1_4"]) == 28)
    g2 = {}
    for (k, axis), (tr, la) in G2_EXPECT.items():
        got = (results[k]["rho_band_mean"][f"{axis}_trunk"], results[k]["rho_band_mean"][f"{axis}_late"])
        g2[f"{k}_{axis}"] = {"expected": [tr, la], "got": [round(got[0], 4), round(got[1], 4)],
                             "ok": abs(got[0] - tr) < 0.0015 and abs(got[1] - la) < 0.0015}
    return {"G1_reproduces_cvpr27_a15": {"values": g1, "ok": bool(ok1)},
            "G2_reproduces_0927_rho": g2, "G2_ok": all(v["ok"] for v in g2.values())}


def plot(results, out):
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    for row, (key, m) in enumerate(MODELS.items()):
        res, n = results[key], m["n"]
        x = np.arange(n)
        eb = m["bands"][m["early"]]
        for ax in axes[row]:
            ax.axvspan(eb[0] - 0.5, eb[-1] + 0.5, color=GRID, alpha=0.9, lw=0, zorder=0)
            ax.axvline(m["split"], color=MUTED, lw=1, ls="--")
            ax.axvline(m["tau"], color=MUTED, lw=1, ls=":")
            ax.set_xlim(-0.5, n - 0.5)
        ax = axes[row, 0]
        for o, col, lab in (("traj", C_TRAJ, "I_traj (trajectory FM)"), ("coc", C_COC, "I_CoC (reasoning NLL)")):
            ax.plot(x, per_layer(res, "mlp", o, "gini"), color=col, lw=2, label=lab)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"{m['name']}\n({n} layers, {res['units']:,} ch)", fontsize=9)
        ax = axes[row, 1]
        ax.plot(x, [np.nan if r is None else r for r in res["rho"]["mlp"]], color=INK, lw=2)
        ax.set_ylim(0, 1)
        ax = axes[row, 2]
        for o, col in (("traj", C_TRAJ), ("coc", C_COC)):
            v = res["stability"][o] * 100
            ax.fill_between(x, np.nanmin(v, 0), np.nanmax(v, 0), color=col, alpha=0.15, lw=0)
            ax.plot(x, np.nanmedian(v, 0), color=col, lw=2)
        ax.set_ylim(0, 100)
    titles = ("(a) Concentration: Gini of MLP channel scores", "(b) Agreement: Spearman ρ(I_traj, I_CoC)",
              "(c) Stability: split-half top-k kept (%)")
    for ax, t in zip(axes[0], titles, strict=True):
        ax.set_title(t, fontsize=9, loc="left")
    for ax in axes[1]:
        ax.set_xlabel("layer")
    axes[0, 1].annotate("dashed: Finding 1 split\ndotted: I_traj / I_CoC step", xy=(0.12, 0.06),
                        xycoords="axes fraction", fontsize=7, color=MUTED)
    axes[0, 0].annotate("shaded: early band\n(A1.5 1–4, A2 1–7)", xy=(0.2, 0.62),
                        xycoords="axes fraction", fontsize=7, color=MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=2, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "depth_structure_a15_a2.png", dpi=200)
    fig.savefig(out / "depth_structure_a15_a2.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="iclr27-rebuttal/a2_depth_structure")
    args = ap.parse_args()
    out = REPO / "outputs" / args.exp_id
    out.mkdir(parents=True, exist_ok=True)

    results = {k: analyse(k, m) for k, m in MODELS.items()}
    g = gates(results)
    bands = {k: band_table(m, results[k]) for k, m in MODELS.items()}
    crit = {k: criteria(m, results[k], bands[k]) for k, m in MODELS.items()}

    (out / "config.json").write_text(json.dumps({
        "sources": {k: str(m["dir"]) for k, m in MODELS.items()}, "cut_ratio": RATIO,
        "top_overlap_frac": TOP_OVERLAP_FRAC, "top_stability_frac": TOP_STAB_FRAC, "n_splits": N_SPLITS,
        "split_seed": 0, "bands": {k: m["bands"] for k, m in MODELS.items()},
        "criteria": {"early_frac": EARLY_FRAC, "gini_hi": GINI_HI, "mid_gini_max": MID_GINI_MAX,
                     "overlap_min": OVERLAP_MIN},
        "plan": "iclr27-rebuttal/plans/2026-11-07_a2-analysis-replication.md"}, indent=2))
    metrics = {"gates": g, "criteria": crit, "bands": bands,
               "pooled_gini": {k: results[k]["pooled_gini"] for k in MODELS},
               "per_layer": {k: {"mlp": results[k]["stats"]["mlp"], "q_gini": {
                   o: per_layer(results[k], "q", o, "gini").tolist() for o in ("traj", "coc")},
                   "rho_mlp": results[k]["rho"]["mlp"], "rho_q": results[k]["rho"]["q"],
                   "overlap": results[k]["overlap"],
                   "stability_median": {o: np.nanmedian(results[k]["stability"][o], 0).tolist()
                                        for o in ("traj", "coc")}} for k in MODELS}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    L = ["VLM MLP depth structure, Alpamayo 1.5 vs Alpamayo 2 Super (raw first-order Taylor, calib_100)", "",
         (f"G1 (reproduces cvpr27 A1.5 numbers): {'PASS' if g['G1_reproduces_cvpr27_a15']['ok'] else 'FAIL'} "
         f"{g['G1_reproduces_cvpr27_a15']['values']}"),
         f"G2 (reproduces 09-27 band-mean rho): {'PASS' if g['G2_ok'] else 'FAIL'}"]
    L += [f"   {k}: expected {v['expected']} got {v['got']}" for k, v in g["G2_reproduces_0927_rho"].items()]
    for k, m in MODELS.items():
        r = results[k]
        L += ["", (f"== {m['name']}  ({m['n']} layers, {r['units']:,} MLP ch; overlap top-{r['k_overlap']}, "
              f"stability top-{r['k_stability']})"),
              (f"pooled Gini (all layers in one pool): traj {r['pooled_gini']['traj']:.3f}  "
              f"CoC {r['pooled_gini']['coc']:.3f}"),
              (f"{'band':<7}{'layers':>7}{'Gini tr':>9}{'Gini CoC':>9}{'n50 tr':>8}{'n50 CoC':>8}{'cut tr':>8}"
              f"{'cut CoC':>8}{'ovl':>6}{'rho':>6}{'stab tr':>8}{'stab CoC':>9}")]
        for b, v in bands[k].items():
            L.append(f"{b:<7}{v['layers']:>7}{v['gini_traj']:>9.2f}{v['gini_coc']:>9.2f}{v['n50_traj']:>8.0f}"
                     f"{v['n50_coc']:>8.0f}{v['cut_traj'] * 100:>7.1f}%{v['cut_coc'] * 100:>7.1f}%"
                     f"{v['overlap']:>6.2f}{v['rho']:>6.2f}{v['stab_traj'] * 100:>7.0f}%{v['stab_coc'] * 100:>8.0f}%")
        c = crit[k]
        L += [(f"criteria: high-Gini early layers {c['high_gini_layers']} (searched {c['early_layers_searched']}), "
              f"run {c['run']}"),
              (f"  C1 concentration {'PASS' if c['C1_concentration'] else 'FAIL'} | "
              f"C2 agreement {'PASS' if c['C2_agreement'] else 'FAIL'} (run overlap {c['run_overlap']:.2f}) | "
              f"C3 stability {'PASS' if c['C3_stability'] else 'FAIL'} "
              f"(run traj {c['run_stability']['traj']:.2f} CoC {c['run_stability']['coc']:.2f})")]
    L += ["", "Alpamayo 2 Super by layer (MLP):",
          (f"{'L':>3}{'Gini tr':>9}{'Gini CoC':>9}{'n50 tr':>8}{'n50 CoC':>8}{'cut tr':>8}{'ovl':>6}{'rho':>6}"
          f"{'stab tr':>8}{'stab CoC':>9}")]
    r = results["a2"]
    for li in range(MODELS["a2"]["n"]):
        t, c = r["stats"]["mlp"]["traj"][li], r["stats"]["mlp"]["coc"][li]
        st = {o: np.nanmedian(r["stability"][o][:, li]) for o in ("traj", "coc")}
        L.append(f"{li:>3}" + (f"{t['gini']:>9.2f}" if t else f"{'-':>9}") + f"{c['gini']:>9.2f}"
                 + (f"{t['n50']:>8d}" if t else f"{'-':>8}") + f"{c['n50']:>8d}"
                 + (f"{t['cut'] * 100:>7.1f}%" if t else f"{'-':>8}")
                 + (f"{r['overlap'][li]:>6.2f}{r['rho']['mlp'][li]:>6.2f}" if t else f"{'-':>6}{'-':>6}")
                 + (f"{st['traj'] * 100:>7.0f}%" if t else f"{'-':>8}") + f"{st['coc'] * 100:>8.0f}%")
    (out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    plot(results, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
