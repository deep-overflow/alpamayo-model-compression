"""dualr_wl + expert MLP-only: does the expert's free axis survive on a refitted VLM?

plans/2026-08-31_dualrwl-expert-mlp.md. Reads the frozen-protocol rows registered in
paper_numbers.ARMS for baseline, baseline_bw, dualr_wl, dualrwl_em50, dualrwl_em75 and
expertm_u50, plus each recipe's slim_meta.json, and judges the gates:

  G0  recipe integrity: the em arms' VLM kept sets are bit-identical to slim_dualr_wl_u40
      and their expert Q heads are all 16, em50's expert MLP kept set is bit-identical to
      slim_expertm_u50, em75 nests inside em50, removed params exact and additive
  G1  em50 - dualr_wl, paired per clip (median primary, mean beside, Wilcoxon). The
      pre-registered bound is +-0.013, the proportional-cost threshold from the dualexp
      plan -- passing it is NON-INFERIORITY at n=500, not equivalence, so the CI is the
      number to quote.
  G2  em75 - em50: is there a free ceiling above 50%? At 75% the expert intermediate
      falls to 2064, equal to its hidden size, so a cliff is structurally plausible here.
  G3  path separation: the expert cut cannot touch CoC generation, so gen_coc must be
      identical to dualr_wl on every clip. Anything else is a build bug.
  DiD does the refitted cache change the expert's tolerance for MLP width? Cost of the
      same cut on the refitted VLM (em50 - dualr_wl, Ada) minus on the dense VLM
      (expertm_u50 - baseline_bw, Blackwell), each against its own architecture's
      baseline as expert-axis G3 does. NOTE this is a null minus a null: it establishes
      transfer only in the regime where the cut is already free. The Q-head half of the
      axis result (q25 +0.022, q50 +0.116 dense, super-linear in head count) is the
      plausible interaction site and its DiD is untested.

Writes outputs/<out>/{metrics.json, summary.txt, plots/*.png}. Arms whose rows are not
complete yet are skipped, so this can be run while the queue is still draining.

Usage:
  .venv/bin/python experiments/evaluation/analyze_dualrwl_em.py [--out dualrwl_em_analysis]
"""

import argparse
import json
import sys
from itertools import pairwise
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

BG, INK, MUTED, GRID = "#FAF9F5", "#29261B", "#6B6555", "#E8E6DC"
C1, C2, C3 = "#2a78d6", "#008300", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.size": 9, "axes.grid": True, "axes.axisbelow": True,
})
REPO = Path(__file__).resolve().parents[2]
SETS = ("indist", "test", "oodval")
EXPECT = {"indist": 500, "test": 500, "oodval": 262}
WL_REMOVED = 2_657_452_032
# arm -> (recipe dir, label, expected removed params, expert MLP kept per layer)
ARMS = {
    "dualr_wl": ("slim_dualr_wl_u40", "dualr_wl (VLM only)", WL_REMOVED, 8256),
    "dualrwl_em50": ("slim_dualrwl_em50_u40", "+ expert MLP 50%",
                     WL_REMOVED + 913_047_552, 4128),
    "dualrwl_em75": ("slim_dualrwl_em75_u40", "+ expert MLP 75%",
                     WL_REMOVED + 1_369_571_328, 2064),
    "dualrwl_em87p5": ("slim_dualrwl_em87p5_u40", "+ expert MLP 87.5%",
                      WL_REMOVED + 1_597_833_216, 1032),
    "dualrwl_em93p75": ("slim_dualrwl_em93p75_u40", "+ expert MLP 93.75%",
                        WL_REMOVED + 1_711_964_160, 516),
    "dualrwl_em96p875": ("slim_dualrwl_em96p875_u40", "+ expert MLP 96.875%",
                         WL_REMOVED + 1_769_029_632, 258),
    "dualrwl_em98p4375": ("slim_dualrwl_em98p4375_u40", "+ expert MLP 98.4375%",
                          WL_REMOVED + 1_797_562_368, 129),
    "dualrwl_em100": ("slim_dualrwl_em100_u40", "+ expert MLP 100% (no MLP)",
                      WL_REMOVED + 1_826_095_104, 0),
}
COLOR = {"dualr_wl": MUTED, "dualrwl_em50": C2, "dualrwl_em75": C1,
         "dualrwl_em87p5": C3, "dualrwl_em93p75": "#b8442a",
         "dualrwl_em96p875": "#8a6fb0", "dualrwl_em98p4375": "#b0537a",
         "dualrwl_em100": "#7a2d1c"}
# direct per-clip contrasts (all three arms share the slim attention path)
CONTRASTS = [("G1", "dualrwl_em50", "dualr_wl"), ("G2", "dualrwl_em75", "dualrwl_em50"),
             ("G2b", "dualrwl_em75", "dualr_wl"),
             ("G2c", "dualrwl_em87p5", "dualrwl_em75"),
             ("G2d", "dualrwl_em87p5", "dualr_wl"),
             ("G2e", "dualrwl_em93p75", "dualrwl_em87p5"),
             ("G2f", "dualrwl_em93p75", "dualr_wl"),
             ("G2g", "dualrwl_em100", "dualrwl_em93p75"),
             ("G2h", "dualrwl_em100", "dualr_wl"),
             # bisecting the cliff: is the collapse continuous in width or discrete in
             # the sublayer's existence?
             ("G2i", "dualrwl_em96p875", "dualrwl_em93p75"),
             ("G2j", "dualrwl_em98p4375", "dualrwl_em96p875"),
             ("G2k", "dualrwl_em100", "dualrwl_em98p4375")]
THRESHOLD = 0.013


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
    """The em arms must be the exact union of the wl VLM half and the expertm expert half."""
    metas = {a: json.loads(p.read_text()) for a, (d, *_) in ARMS.items()
             if (p := REPO / "outputs" / d / "slim_meta.json").exists()}
    m50 = json.loads((REPO / "outputs" / "slim_expertm_u50" / "slim_meta.json").read_text())
    wl = metas["dualr_wl"]
    g = {}
    for a, m in metas.items():
        g[a] = {
            "removed": m["params"]["removed"], "expected": ARMS[a][2],
            "removed_ok": m["params"]["removed"] == ARMS[a][2],
            # the VLM half must be bit-identical to dualr_wl -- same selection, same refit
            "vlm_identical_to_wl": all(l["q"] == w["q"] and l["mlp"] == w["mlp"]
                                       for l, w in zip(m["vlm"], wl["vlm"])),
            "expert_q_whole": all(len(l["q"]) == 16 for l in m["expert"]),
            # the endpoint keeps zero channels: the MLP sublayer is gone, not narrowed
            "expert_mlp_empty": all(not l["mlp"] for l in m["expert"]),
            "kept_mlp": sorted({len(l["mlp"]) for l in m["expert"]}),
            "kv_intact": not m["kvonly_layers"],
        }
    g["em50_expert_is_expertm_u50"] = all(
        metas["dualrwl_em50"]["expert"][i]["mlp"] == m50["expert"][i]["mlp"]
        for i in range(len(m50["expert"])))
    rungs = [a for a in ("dualrwl_em50", "dualrwl_em75", "dualrwl_em87p5",
                         "dualrwl_em93p75", "dualrwl_em96p875", "dualrwl_em98p4375",
                         "dualrwl_em100") if a in metas]
    g["em75_nested_in_em50"] = all(
        set(metas[b]["expert"][i]["mlp"]) <= set(metas[a]["expert"][i]["mlp"])
        for a, b in pairwise(rungs) for i in range(len(m50["expert"])))
    # the two halves touch different matrices, so their removed counts must simply add
    g["additive"] = all(
        metas[a]["params"]["removed"] - WL_REMOVED == ARMS[a][2] - WL_REMOVED
        for a in rungs)
    g["pass"] = (all(g[a]["removed_ok"] and g[a]["vlm_identical_to_wl"]
                     and g[a]["expert_q_whole"] and g[a]["kv_intact"] for a in metas)
                 and g["em50_expert_is_expertm_u50"] and g["em75_nested_in_em50"]
                 and g["additive"])
    return g


def load_set(s):
    rows = {}
    for arm in ("baseline", "baseline_bw", "expert_m50", *ARMS):
        spec = pn.ARMS.get(arm, {}).get(s)
        if spec is None:
            continue
        try:
            r = pn.load(*spec)
        except SystemExit as e:
            print(f"!! {e}")
            continue
        if len(r) >= EXPECT[s]:
            rows[arm] = r
        elif r:
            print(f"   ({arm}/{s} incomplete: {len(r)}/{EXPECT[s]} clips, skipped)")
    return rows


def analyse_set(s, rows):
    out = {"n_ref": len(rows["dualr_wl"]), "arms": {}, "contrasts": {}, "coc": {},
           "baselines": {}}
    for b in ("baseline", "baseline_bw"):
        if b in rows:
            out["baselines"][b] = {
                "n": len(rows[b]),
                "minADE6_mean": float(np.mean([pn.at6(r, "ade_rollout_k")
                                               for r in rows[b].values()])),
                "minFDE6_mean": float(np.mean([pn.at6(r, "fde_rollout_k")
                                               for r in rows[b].values()]))}
    ade = {}
    for arm in ARMS:
        if arm not in rows:
            continue
        ade[arm] = {i: pn.at6(r, "ade_rollout_k") for i, r in rows[arm].items()}
        entry = {"n": len(ade[arm]),
                 "minADE6_mean": float(np.mean(list(ade[arm].values()))),
                 "minFDE6_mean": float(np.mean([pn.at6(r, "fde_rollout_k")
                                                for r in rows[arm].values()]))}
        for ref in ("baseline", "dualr_wl"):
            if ref not in rows or ref == arm:
                continue
            ids = sorted(set(rows[ref]) & set(rows[arm]))
            entry[f"d_ade_vs_{ref}"] = paired(
                [ade[arm][i] for i in ids],
                [pn.at6(rows[ref][i], "ade_rollout_k") for i in ids])
            entry[f"coc_same_as_{ref}"] = float(np.mean(
                [rows[arm][i]["gen_coc"] == rows[ref][i]["gen_coc"] for i in ids]))
        out["arms"][arm] = entry
    for tag, a, b in CONTRASTS:
        if a in ade and b in ade:
            ids = sorted(set(ade[a]) & set(ade[b]))
            out["contrasts"][f"{tag} {a}-{b}"] = paired([ade[a][i] for i in ids],
                                                        [ade[b][i] for i in ids])
            out["coc"][f"{a}=={b}"] = float(np.mean(
                [rows[a][i]["gen_coc"] == rows[b][i]["gen_coc"] for i in ids]))
    # DiD: the same expert cut on the refitted VLM vs on the dense one, each delta taken
    # against its own architecture's baseline (Ada / Blackwell)
    if all(k in rows for k in ("dualrwl_em50", "dualr_wl", "expert_m50", "baseline_bw")):
        ids = sorted(set(rows["dualrwl_em50"]) & set(rows["dualr_wl"])
                     & set(rows["expert_m50"]) & set(rows["baseline_bw"]))
        refit = np.array([pn.at6(rows["dualrwl_em50"][i], "ade_rollout_k")
                          - pn.at6(rows["dualr_wl"][i], "ade_rollout_k") for i in ids])
        dense = np.array([pn.at6(rows["expert_m50"][i], "ade_rollout_k")
                          - pn.at6(rows["baseline_bw"][i], "ade_rollout_k") for i in ids])
        out["did"] = {"refit": paired(refit, np.zeros_like(refit)),
                      "dense": paired(dense, np.zeros_like(dense)),
                      "refit_minus_dense": paired(refit, dense)}
    out["_ade"] = ade
    return out


def plot_deltas(res, out):
    sets = [s for s in SETS if s in res]
    if not sets:
        return
    fig, axes = plt.subplots(1, len(sets), figsize=(4.6 * len(sets), 3.0), squeeze=False)
    for ax, s in zip(axes[0], sets):
        arms = [a for a in ARMS if a in res[s]["arms"] and a != "dualr_wl"]
        for k, a in enumerate(arms):
            d = res[s]["arms"][a].get("d_ade_vs_dualr_wl")
            if d is None:
                continue
            ax.errorbar(d["med"], k, xerr=[[d["med"] - d["lo"]], [d["hi"] - d["med"]]],
                        fmt="o", color=COLOR[a], capsize=3)
            ax.plot(d["mean"], k, marker="x", color=MUTED, ms=6)
        ax.axvspan(-THRESHOLD, THRESHOLD, color=C3, alpha=0.12, lw=0)
        ax.axvline(0, color=INK, lw=0.8)
        # the endpoint (no MLP at all) is ~100x the threshold, so a linear axis would
        # collapse the four passing rungs onto the zero line
        ax.set_xscale("symlog", linthresh=THRESHOLD)
        ax.set_yticks(np.arange(len(arms)))
        ax.set_yticklabels([ARMS[a][1] for a in arms])
        ax.invert_yaxis()
        ax.set_title(f"{s} (n={res[s]['n_ref']})")
        ax.set_xlabel("median [95% CI] vs dualr_wl, symlog  (x = mean)")
    fig.suptitle("expert MLP-only added to the refitted VLM; shaded = the pre-registered "
                 "+-0.013 non-inferiority bound", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "dualrwl_em_deltas.png", dpi=150)
    plt.close(fig)


def plot_cliff(res, out):
    """minADE@6 against surviving channel count: flat for four rungs, then a cliff."""
    sets = [x for x in SETS if x in res]
    if not sets:
        return
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    colors = {"indist": C2, "test": C1, "oodval": C3}
    for s in sets:
        pts = [(ARMS[a][3], res[s]["arms"][a]["minADE6_mean"])
               for a in ARMS if a in res[s]["arms"]]
        pts.sort(reverse=True)
        # 0 kept has no place on a log axis; draw it at half the smallest positive rung
        floor = min(x for x, _ in pts if x > 0) / 2
        xs = [x if x > 0 else floor for x, _ in pts]
        ax.plot(xs, [y for _, y in pts], marker="o", color=colors[s], label=s)
        ax.axhline(res[s]["baselines"]["baseline"]["minADE6_mean"], color=colors[s],
                   lw=0.7, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([floor, 516, 1032, 2064, 4128, 8256])
    ax.set_xticklabels(["0", "516", "1032", "2064", "4128", "8256"])
    ax.set_xlabel("expert MLP channels kept per layer (8256 = untouched)")
    ax.set_ylabel("minADE@6")
    ax.set_title("width is nearly free; the sublayer is not")
    ax.legend(frameon=False, fontsize=8, title="dotted = unpruned baseline",
              title_fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "expert_mlp_cliff.png", dpi=150)
    plt.close(fig)


def plot_params(res, out):
    s = next((x for x in SETS if x in res), None)
    if s is None:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for a, (_, label, params, _) in ARMS.items():
        if a not in res[s]["arms"]:
            continue
        d = res[s]["arms"][a].get("d_ade_vs_baseline")
        if d is None:
            continue
        ax.errorbar(params / 1e9, d["med"], yerr=[[d["med"] - d["lo"]], [d["hi"] - d["med"]]],
                    fmt="o", color=COLOR[a], capsize=3)
        ax.annotate(label, (params / 1e9, d["med"]), xytext=(4, 4),
                    textcoords="offset points", fontsize=8)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("removed parameters (B)")
    ax.set_ylabel("median paired dminADE@6 vs unpruned")
    ax.set_title(f"cost vs budget ({s}): the expert half is nearly vertical")
    fig.tight_layout()
    fig.savefig(out / "dualrwl_em_params.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dualrwl_em_analysis")
    args = ap.parse_args()
    out = REPO / "outputs" / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)

    g0 = recipe_gates()
    res = {}
    for s in SETS:
        rows = load_set(s)
        if "dualr_wl" in rows and len(rows) > 1:
            res[s] = analyse_set(s, rows)

    lines = [("dualr_wl + expert MLP-only -- does the expert's free axis survive a "
              "refitted VLM?"), "",
             (f"G0 recipes: {'PASS' if g0['pass'] else 'FAIL'}  "
              f"(em50 expert == expertm_u50 {g0['em50_expert_is_expertm_u50']}, "
              f"em75 nested {g0['em75_nested_in_em50']}, additive {g0['additive']})")]
    for a, (_, label, _, _) in ARMS.items():
        if a not in g0:
            continue
        r = g0[a]
        lines.append(f"    {a:14s} {label:22s} removed {r['removed']:>13,} "
                     f"{'OK' if r['removed_ok'] else 'MISMATCH'}  "
                     f"VLM==wl {r['vlm_identical_to_wl']}  expert q whole "
                     f"{r['expert_q_whole']}  kept mlp {r['kept_mlp']}")
    for s, r in res.items():
        lines.append("")
        for b, v in r["baselines"].items():
            lines.append(f"[{s}]  {b:12s} minADE@6 {v['minADE6_mean']:.4f}  "
                         f"minFDE@6 {v['minFDE6_mean']:.4f}  (n={v['n']})")
        for a, v in r["arms"].items():
            lines.append(f"    {a:14s} n={v['n']:3d}  ADE {v['minADE6_mean']:.4f}  "
                         f"FDE {v['minFDE6_mean']:.4f}")
            for ref in ("baseline", "dualr_wl"):
                if f"d_ade_vs_{ref}" in v:
                    lines.append(f"        vs {ref:9s} {fmt(v[f'd_ade_vs_{ref}'])}  "
                                 f"coc==ref {v[f'coc_same_as_{ref}']:.3f}")
        for k, v in r["contrasts"].items():
            verdict = ("PASS" if abs(v["med"]) <= THRESHOLD else "REJECT")
            lines.append(f"    {k:34s} {fmt(v)}  [{verdict} vs +-{THRESHOLD}]")
        for k, v in r["coc"].items():
            lines.append(f"    gen_coc identical {k}: {v:.3f}")
        if "did" in r:
            lines.append("    DiD (does the refitted cache change the expert's tolerance?)")
            for k, v in r["did"].items():
                lines.append(f"        {k:20s} {fmt(v)}")
    text = "\n".join(lines)
    print(text)
    (out / "summary.txt").write_text(text + "\n")
    slim = {s: {k: v for k, v in r.items() if k != "_ade"} for s, r in res.items()}
    (out / "metrics.json").write_text(json.dumps({"g0": g0, "sets": slim,
                                                  "threshold": THRESHOLD}, indent=1))
    plot_deltas(res, out / "plots")
    plot_params(res, out / "plots")
    plot_cliff(res, out / "plots")


if __name__ == "__main__":
    main()
