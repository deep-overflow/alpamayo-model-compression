"""Expert axis decomposition: Q heads vs MLP channels at a matched ratio.

plans/2026-08-28_expert-axis-ablation.md. Reads the frozen-protocol rows registered in
paper_numbers.ARMS for baseline, expert_q25, expert_m25, expert_m_pm (= expertm_c341,
parameter-matched to q25), expert_q50, expert_m50 and expert_both25 (the shipped
expert_u25 = q25 | m25), plus each recipe's slim_meta.json, and judges the gates:

  G0  recipe integrity: removed params exact, VLM/KV intact, q25|m25 == both25 unit for
      unit, 50% cuts nest the 25% cuts, gen_coc identical across the expert arms
      (the expert cannot touch the CoC; all expert arms share the slim attention path)
  G1  q25 - m25, paired per clip (median primary, mean beside, Wilcoxon): CI above 0 means
      the Q-head axis costs more than the MLP axis although it removes 6.05x fewer params
  G2  additivity on val500: both25 - (q25 + m25)
  G3  scaling: q50 vs q25 and m50 vs m25. The 50% arms were evaluated on Blackwell
      (user decision 2026-08-28, to halve the wall-clock) against a Blackwell baseline
      re-measured with per-sample arrays, so G3 compares each arm's delta against its OWN
      architecture's baseline, per clip: (q50 - base_bw) - (q25 - base_ada). Architecture
      bias was measured at +0.0000/+0.0001 (p=0.82), so this costs noise, not bias.
      q50 - m50 (G3x) is a same-architecture pair and stays a direct contrast.
  G4  parameter-matched: q25 - m_pm (same 75.4-75.5M removed, only the axis differs)

Writes outputs/<out>/{metrics.json, summary.txt, plots/*.png}. Arms without rows yet are
skipped, so the script can be run while the queue is still draining.

Usage:
  .venv/bin/python experiments/evaluation/analyze_expert_axis.py [--out expert_axis_analysis]
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
# arm -> (recipe dir, axis, label, expected removed params, same-architecture baseline arm)
ARMS = {
    "expert_q25": ("slim_expertq_u25", "q", "Q heads 25% (4/16)", 75_497_472, "baseline"),
    "expert_m_pm": ("slim_expertm_c341", "m", "MLP 341 ch (param-matched)", 75_423_744,
                    "baseline"),
    "expert_m25": ("slim_expertm_u25", "m", "MLP 25% (2064/8256)", 456_523_776, "baseline"),
    "expert_q50": ("slim_expertq_u50", "q", "Q heads 50% (8/16) [BW]", 150_994_944,
                   "baseline_bw"),
    "expert_m50": ("slim_expertm_u50", "m", "MLP 50% (4128/8256) [BW]", 913_047_552,
                   "baseline_bw"),
    "expert_both25": ("slim_expert_znorm_r25", "both", "Q 25% + MLP 25% (expert_u25)",
                      532_021_248, "baseline"),
}
COLOR = {"q": C1, "m": C2, "both": C4}
# direct per-clip contrasts: both arms on one architecture
CONTRASTS = [("G1", "expert_q25", "expert_m25"), ("G4", "expert_q25", "expert_m_pm"),
             ("G3x", "expert_q50", "expert_m50")]
# delta-vs-delta contrasts across architectures: (a - base_a) - (b - base_b) per clip
DELTA_CONTRASTS = [("G3q", "expert_q50", "expert_q25"), ("G3m", "expert_m50", "expert_m25")]


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
    metas = {a: json.loads((REPO / "outputs" / d / "slim_meta.json").read_text())
             for a, (d, *_) in ARMS.items()}
    g = {}
    for a, m in metas.items():
        vlm_ok = all(len(l["q"]) == 32 and len(l["mlp"]) == 12288 for l in m["vlm"])
        g[a] = {"removed": m["params"]["removed"], "expected": ARMS[a][3],
                "removed_ok": m["params"]["removed"] == ARMS[a][3],
                "vlm_kv_intact": vlm_ok and not m["kvonly_layers"],
                "kept_q": sorted({len(l["q"]) for l in m["expert"]}),
                "kept_mlp": sorted({len(l["mlp"]) for l in m["expert"]})}
    e = {a: m["expert"] for a, m in metas.items()}
    g["union_is_both25"] = all(
        e["expert_q25"][l]["q"] == e["expert_both25"][l]["q"]
        and e["expert_m25"][l]["mlp"] == e["expert_both25"][l]["mlp"] for l in range(36))
    g["nested"] = all(
        set(e["expert_q50"][l]["q"]) <= set(e["expert_q25"][l]["q"])
        and set(e["expert_m50"][l]["mlp"]) <= set(e["expert_m25"][l]["mlp"])
        and set(e["expert_m25"][l]["mlp"]) <= set(e["expert_m_pm"][l]["mlp"])
        for l in range(36))
    g["pass"] = (all(g[a]["removed_ok"] and g[a]["vlm_kv_intact"] for a in ARMS)
                 and g["union_is_both25"] and g["nested"])
    return g


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


def analyse_set(s, rows):
    out = {"n_baseline": len(rows["baseline"]), "arms": {}, "contrasts": {}, "coc": {},
           "baselines": {}}
    for b in ("baseline", "baseline_bw"):
        if b in rows:
            out["baselines"][b] = {
                "n": len(rows[b]),
                "minADE6_mean": float(np.mean([pn.at6(r, "ade_rollout_k")
                                               for r in rows[b].values()])),
                "minFDE6_mean": float(np.mean([pn.at6(r, "fde_rollout_k")
                                               for r in rows[b].values()]))}
    ade, dade = {}, {}
    for arm in ARMS:
        if arm not in rows or ARMS[arm][4] not in rows:
            continue
        base = rows[ARMS[arm][4]]
        ids = sorted(set(base) & set(rows[arm]))
        ade[arm] = {i: pn.at6(rows[arm][i], "ade_rollout_k") for i in ids}
        dade[arm] = {i: ade[arm][i] - pn.at6(base[i], "ade_rollout_k") for i in ids}
        fde = paired([pn.at6(rows[arm][i], "fde_rollout_k") for i in ids],
                     [pn.at6(base[i], "fde_rollout_k") for i in ids])
        out["arms"][arm] = {
            "n": len(ids), "baseline": ARMS[arm][4],
            "minADE6_mean": float(np.mean(list(ade[arm].values()))),
            "minFDE6_mean": float(np.mean([pn.at6(rows[arm][i], "fde_rollout_k")
                                           for i in ids])),
            "d_ade": paired(list(ade[arm].values()), [pn.at6(base[i], "ade_rollout_k")
                                                      for i in ids]),
            "d_fde": fde,
            "coc_same_as_baseline": float(np.mean([rows[arm][i]["gen_coc"] == base[i]["gen_coc"]
                                                   for i in ids])),
        }
    out["baseline"] = out["baselines"]["baseline"]
    for tag, a, b in DELTA_CONTRASTS:
        if a in dade and b in dade:
            ids = sorted(set(dade[a]) & set(dade[b]))
            out["contrasts"][f"{tag} d({a})-d({b})"] = paired(
                [dade[a][i] for i in ids], [dade[b][i] for i in ids])
    for tag, a, b in CONTRASTS:
        if a in ade and b in ade:
            ids = sorted(set(ade[a]) & set(ade[b]))
            out["contrasts"][f"{tag} {a}-{b}"] = paired([ade[a][i] for i in ids],
                                                        [ade[b][i] for i in ids])
            out["coc"][f"{a}=={b}"] = float(np.mean([rows[a][i]["gen_coc"] == rows[b][i]["gen_coc"]
                                                     for i in ids]))
    if all(a in dade for a in ("expert_q25", "expert_m25", "expert_both25")):
        ids = sorted(set(dade["expert_q25"]) & set(dade["expert_m25"])
                     & set(dade["expert_both25"]))
        dq = np.array([dade["expert_q25"][i] for i in ids])
        dm = np.array([dade["expert_m25"][i] for i in ids])
        db = np.array([dade["expert_both25"][i] for i in ids])
        out["contrasts"]["G2 both25-(q25+m25)"] = paired(db, dq + dm)
    out["_ade"] = ade
    return out


def plot_deltas(res, out):
    sets = [s for s in SETS if s in res]
    fig, axes = plt.subplots(1, len(sets), figsize=(4.6 * len(sets), 3.6), squeeze=False)
    for ax, s in zip(axes[0], sets):
        arms = [a for a in ARMS if a in res[s]["arms"]]
        y = np.arange(len(arms))
        for k, a in enumerate(arms):
            d = res[s]["arms"][a]["d_ade"]
            ax.errorbar(d["med"], k, xerr=[[d["med"] - d["lo"]], [d["hi"] - d["med"]]],
                        fmt="o", color=COLOR[ARMS[a][1]], capsize=3)
            ax.plot(d["mean"], k, marker="x", color=MUTED, ms=6)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([ARMS[a][2] for a in arms])
        ax.invert_yaxis()
        ax.set_title(f"{s} (n={res[s]['n_baseline']})")
        ax.set_xlabel("median [95% CI]  (x = mean)")
    fig.suptitle("paired dminADE@6 vs the arm's own-architecture baseline; "
                 "blue = Q heads, green = MLP, orange = both", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "expert_axis_deltas.png", dpi=150)
    plt.close(fig)


def plot_params(res, out):
    s = "indist" if "indist" in res else next(iter(res))
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for a, (_, axis, label, params, _) in ARMS.items():
        if a not in res[s]["arms"]:
            continue
        d = res[s]["arms"][a]["d_ade"]
        ax.errorbar(params / 1e6, d["med"], yerr=[[d["med"] - d["lo"]], [d["hi"] - d["med"]]],
                    fmt="o", color=COLOR[axis], capsize=3)
        ax.annotate(label.split(" (")[0], (params / 1e6, d["med"]), xytext=(4, 4),
                    textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("removed parameters (M, log)")
    ax.set_ylabel("median paired dminADE@6 vs baseline")
    ax.set_title(f"cost vs parameters removed ({s}); blue = Q heads, green = MLP")
    fig.tight_layout()
    fig.savefig(out / "expert_axis_params.png", dpi=150)
    plt.close(fig)


def plot_perclip(res, out):
    s = "indist" if "indist" in res else next(iter(res))
    ade = res[s]["_ade"]
    pairs = [("expert_q25", "expert_m25", "q25 - m25 (ratio-matched)"),
             ("expert_q25", "expert_m_pm", "q25 - m_pm (param-matched)")]
    pairs = [p for p in pairs if p[0] in ade and p[1] in ade]
    if not pairs:
        return
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.6 * len(pairs), 3.4), squeeze=False)
    for ax, (a, b, title) in zip(axes[0], pairs):
        ids = sorted(set(ade[a]) & set(ade[b]))
        d = np.clip(np.array([ade[a][i] - ade[b][i] for i in ids]), -0.6, 0.6)
        ax.hist(d, bins=48, color=C1, alpha=0.8)
        ax.axvline(0, color=INK, lw=0.8)
        ax.axvline(np.median(d), color=C3, lw=1.5, label=f"median {np.median(d):+.4f}")
        ax.set_title(f"{s}: per-clip {title}")
        ax.set_xlabel("dminADE@6 (m, clipped +-0.6)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "expert_axis_perclip.png", dpi=150)
    plt.close(fig)


def plot_mass(out, imp_run="importance_stepexp_sum"):
    """Share of a layer's first-order importance mass (|sum_s dL/dg|, clip-averaged) held
    by the lowest-scoring fraction of units, averaged over the 36 layers -- the reason the
    MLP axis is free: its mass sits in a few channels, the Q heads' does not."""
    z = np.load(REPO / "outputs" / imp_run / "importance.npz")
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    shares = {}
    for key, color, label in (("traj_exp_q", C1, "Q heads (16/layer)"),
                              ("traj_exp_mlp", C2, "MLP channels (8256/layer)")):
        v = np.sort(z[key], axis=1)  # (36, U) ascending within layer
        cum = np.cumsum(v, axis=1) / v.sum(1, keepdims=True)  # (36, U)
        x = (np.arange(v.shape[1]) + 1) / v.shape[1]
        m = cum.mean(0)
        ax.plot(x, m, color=color, lw=2, label=label)
        shares[key] = {str(f): float(np.interp(f, x, m)) for f in (0.25, 0.5)}
    for f in (0.25, 0.5):
        ax.axvline(f, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("lowest-scoring fraction of the layer's units (removed first)")
    ax.set_ylabel("share of the layer's importance mass")
    ax.set_title("importance mass by unit rank: MLP concentrated, Q heads spread")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "expert_axis_mass.png", dpi=150)
    plt.close(fig)
    return shares


def render_template(metrics, template, out_path):
    """Fill {{...}} tokens in the report template from metrics.json, so no number is typed
    by hand. Grammar (S = indist|test|oodval, A = arm key, T = contrast tag G1|G4|G2|G3q|G3m|G3x):
      {{S.A.med}} / {{S.A.mean}} / {{S.A.ade}} / {{S.A.fde}} / {{S.A.n}} / {{S.A.p}} / {{S.A.coc}}
      {{S.base.ade}} / {{S.base.fde}} / {{S.basebw.ade}} / {{S.basebw.fde}} / {{S.base.n}}
      {{S.C.T.med}} / {{S.C.T.mean}} / {{S.C.T.p}}
      {{mass.q.25}} / {{mass.q.50}} / {{mass.mlp.25}} / {{mass.mlp.50}}   (as percent)
    A missing token is an error rather than a blank cell."""
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
        parts = tok.split(".")
        if parts[0] == "mass":
            v = metrics["mass"]["traj_exp_q" if parts[1] == "q" else "traj_exp_mlp"]
            return f"{100 * v[{'25': '0.25', '50': '0.5'}[parts[2]]]:.2f}%"
        S = metrics["sets"][parts[0]]
        if parts[1] in ("base", "basebw"):
            b = S["baselines"]["baseline" if parts[1] == "base" else "baseline_bw"]
            return (f"{b['minADE6_mean']:.4f}" if parts[2] == "ade" else
                    f"{b['minFDE6_mean']:.4f}" if parts[2] == "fde" else str(b["n"]))
        if parts[1] == "C":
            key = next(k for k in S["contrasts"] if k.split()[0] == parts[2])
            return ci(S["contrasts"][key], parts[3])
        a = S["arms"][parts[1]]
        if parts[2] in ("med", "mean", "p"):
            return ci(a["d_ade"], parts[2])
        return {"ade": f"{a['minADE6_mean']:.4f}", "fde": f"{a['minFDE6_mean']:.4f}",
                "n": str(a["n"]), "coc": f"{a['coc_same_as_baseline']:.3f}"}[parts[2]]

    text = Path(template).read_text()
    text = re.sub(r"\{\{([^}]+)\}\}", lambda m: lookup(m.group(1).strip()), text)
    Path(out_path).write_text(text)
    print(f"rendered {template} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="expert_axis_analysis")
    ap.add_argument("--template", default=None,
                    help="report template with {{...}} tokens; rendered next to it as "
                         "<stem>_filled.html for build_report.py")
    args = ap.parse_args()
    out = REPO / "outputs" / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)

    g0 = recipe_gates()
    res = {}
    for s in SETS:
        rows = load_set(s)
        if "baseline" in rows and len(rows) > 1:
            res[s] = analyse_set(s, rows)

    lines = [("expert axis decomposition -- Q heads vs MLP channels, znorm traj_exp_*, "
              "VLM/KV untouched"), "",
             (f"G0 recipes: {'PASS' if g0['pass'] else 'FAIL'}  "
              f"(union==both25 {g0['union_is_both25']}, nested {g0['nested']})")]
    for a, (_, axis, label, _, _) in ARMS.items():
        r = g0[a]
        lines.append(f"    {a:14s} {label:32s} removed {r['removed']:>13,} "
                     f"{'OK' if r['removed_ok'] else 'MISMATCH'}  kept q {r['kept_q']} "
                     f"mlp {r['kept_mlp']}")
    for s, r in res.items():
        lines.append("")
        for b, v in r["baselines"].items():
            lines.append(f"[{s}]  {b:12s} minADE@6 {v['minADE6_mean']:.4f}  "
                         f"minFDE@6 {v['minFDE6_mean']:.4f}  (n={v['n']})")
        for a, v in r["arms"].items():
            lines.append(f"    {a:14s} n={v['n']:3d}  ADE {v['minADE6_mean']:.4f}  "
                         f"FDE {v['minFDE6_mean']:.4f}  dADE vs {v['baseline']:11s} "
                         f"{fmt(v['d_ade'])}  coc==base {v['coc_same_as_baseline']:.3f}")
        for k, v in r["contrasts"].items():
            lines.append(f"    {k:32s} {fmt(v)}")
        for k, v in r["coc"].items():
            lines.append(f"    gen_coc identical {k}: {v:.3f}")
    text = "\n".join(lines)
    print(text)
    (out / "summary.txt").write_text(text + "\n")
    slim = {s: {k: v for k, v in r.items() if k != "_ade"} for s, r in res.items()}
    mass = plot_mass(out / "plots")
    lines += ["", ("importance mass held by the lowest-scoring 25% / 50% of a layer's units "
                   "(|sum_s| Taylor, layer-averaged):")]
    for k, v in mass.items():
        lines.append(f"    {k:13s} {v['0.25']:.4f} / {v['0.5']:.4f}")
    text = "\n".join(lines)
    print("\n".join(lines[-3:]))
    (out / "summary.txt").write_text(text + "\n")
    (out / "metrics.json").write_text(json.dumps({"g0": g0, "sets": slim, "mass": mass},
                                                 indent=1))
    if res:
        plot_deltas(res, out / "plots")
        plot_params(res, out / "plots")
        plot_perclip(res, out / "plots")
    if args.template:
        t = Path(args.template)
        render_template(json.loads((out / "metrics.json").read_text()), t,
                        t.with_name(t.stem + "_filled.html"))


if __name__ == "__main__":
    main()
