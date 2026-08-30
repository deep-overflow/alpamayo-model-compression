"""plans/2026-08-30_dualr-weighted-hessian.md -- the 2x2 Hessian factorial on dualr.

Combines three sources into one metrics file for the report:
  G3a  in-sample reconstruction error per module from the four supernets' metadata
       (run_cache_recon.py: err_{P,D}_{refit,mask}); paired over the 72 modules,
       arm - rep, bootstrap CI of the median
  G1/G2 from analyze_cacheproxy.py (--out cacheproxy_dualrw): cache-only cost, weighted
       shift ratio vs dualr, val500 dminADE
  G3b  LingoQA from analyze_lingo.py runs (per_run accuracy + paired deltas)
Writes outputs/<out>/{metrics_analysis.json, dualrw_summary.txt, plots/}.

Usage:
  .venv/bin/python experiments/head_analysis/analyze_dualrw.py --out dualrw_analysis \
      --proxy cacheproxy_dualrw --lingo lingo_dualrw_vs_dualr
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BG, INK, MUTED, GRID = "#FAF9F5", "#29261B", "#6B6555", "#E8E6DC"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.size": 9, "axes.grid": True, "axes.axisbelow": True,
})
REPO = Path(__file__).resolve().parents[2]
ARMS = {"rep": "dualr_rep_supernet_u40", "d": "dualr_d_supernet_u40",
        "e": "dualr_e_supernet_u40", "w": "dualr_w_supernet_u40"}
LABEL = {"rep": "dualr (uniform, decode 0)", "d": "dualr_d (uniform, decode 0.16)",
         "e": "dualr_e (expert prefill, decode 0)", "w": "dualr_w (expert prefill, decode 0.16)"}


def med_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = np.median(x[rng.integers(0, len(x), (n, len(x)))], 1)
    return float(np.median(x)), *[float(q) for q in np.percentile(b, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dualrw_analysis")
    ap.add_argument("--proxy", default="cacheproxy_dualrw")
    ap.add_argument("--lingo", default=None, help="analyze_lingo exp-id with dualr as baseline")
    ap.add_argument("--extra-arm", nargs="*", default=[],
                    help="name=supernet exp-id to add to the recon-error section (e.g. wl=...)")
    ap.add_argument("--lingo-vs-w", default=None,
                    help="analyze_lingo exp-id with dualr_w as baseline; the wl comparison "
                         "is exported as `wl_vs_w`")
    args = ap.parse_args()
    for spec in args.extra_arm:
        name, sup = spec.split("=")
        ARMS[name] = sup
        LABEL[name] = name
    out = REPO / "outputs" / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)
    res = {"recon": {}, "lingo": {}, "proxy": {}, "g0": {}}
    lines = ["dualr x Hessian 2x2 (plans/2026-08-30_dualr-weighted-hessian.md)", ""]

    # --- G0: weight-space reproduction (pre-registered, expected to fail on o_proj) and the
    # function-space check on a few layers
    rep_dir = REPO / "outputs" / ARMS["rep"]
    gw = rep_dir / "g0_reproduction.json"
    if gw.exists():
        rel = json.loads(gw.read_text())["rel"]
        o = [v for k, v in rel.items() if "o_proj" in k]
        m = [v for k, v in rel.items() if "down_proj" in k]
        res["g0"]["weight"] = {"o_proj_median": float(np.median(o)), "o_proj_max": float(max(o)),
                               "down_proj_median": float(np.median(m)),
                               "down_proj_max": float(max(m)),
                               "verdict": "FAIL" if max(rel.values()) >= 1e-2 else "PASS"}
        lines.append(f"G0 weight-space: o_proj rel diff median {np.median(o):.3f} max {max(o):.1f}; "
                     f"down_proj median {np.median(m):.3f} max {max(m):.3f} -> "
                     f"{res['g0']['weight']['verdict']}")
    gf = rep_dir / "g0_function_space.json"
    if gf.exists():
        fs = json.loads(gf.read_text())
        res["g0"]["fn"] = {k.replace("layers.", "L").replace(".self_attn.o_proj", "_o_proj")
                           .replace(".mlp.down_proj", "_down_proj"): v for k, v in fs.items()}
        res["g0"]["fn_between_max"] = float(max(v["between"] for v in fs.values()))
        res["g0"]["fn_between_median"] = float(np.median([v["between"] for v in fs.values()]))
        lines.append("G0 function-space ||(W'_rep - W'_dualr)X|| / ||WX|| (dense-input H, 10 clips):")
        for k, v in fs.items():
            lines.append(f"  {k:28s} between {v['between']:.4f} | rep err {v['rep_err']:.4f} "
                         f"dualr err {v['dualr_err']:.4f} | weight rel diff {v['weight_rel_diff']:.3f}")

    # --- G3a reconstruction errors
    errs = {}
    for a, sup in ARMS.items():
        p = REPO / "outputs" / sup / "metadata.json"
        if p.exists():
            errs[a] = json.loads(p.read_text()).get("recon_errors", {})
    mods = sorted(set.intersection(*[set(e) for e in errs.values()])) if errs else []
    for a, e in errs.items():
        r = {}
        for kind in ("o_proj", "down_proj"):
            ms = [m for m in mods if kind in m]
            for st in ("P", "D", "L"):
                if not all(f"err_{st}_refit" in e[m] for m in ms):
                    continue
                refit = np.array([e[m][f"err_{st}_refit"] for m in ms])
                mask = np.array([e[m][f"err_{st}_mask"] for m in ms])
                r[f"{kind}_{st}_refit_median"] = float(np.median(refit))
                r[f"{kind}_{st}_mask_median"] = float(np.median(mask))
                r[f"{kind}_{st}_refit_minus_mask"] = med_ci(refit - mask)
                if a != "rep" and "rep" in errs and all(f"err_{st}_refit" in errs["rep"][m] for m in ms):
                    ref = np.array([errs["rep"][m][f"err_{st}_refit"] for m in ms])
                    r[f"{kind}_{st}_minus_rep"] = med_ci(refit - ref)
        res["recon"][a] = r
    lines.append("G3a in-sample reconstruction error (median over modules; refit - mask, and arm - rep):")
    for a, r in res["recon"].items():
        for kind in ("o_proj", "down_proj"):
            s = (f"  {a:4s} {kind:9s} P refit {r[f'{kind}_P_refit_median']:.4f} (mask "
                 f"{r[f'{kind}_P_mask_median']:.4f}) | D refit {r[f'{kind}_D_refit_median']:.4f} "
                 f"(mask {r[f'{kind}_D_mask_median']:.4f})")
            if f"{kind}_D_minus_rep" in r:
                d = r[f"{kind}_D_minus_rep"]; pp = r[f"{kind}_P_minus_rep"]
                s += (f" | vs rep: D {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}] "
                      f"P {pp[0]:+.4f} [{pp[1]:+.4f},{pp[2]:+.4f}]")
            lines.append(s)

    # --- G1/G2 from the cache proxy analysis
    pp = REPO / "outputs" / args.proxy / "metrics_analysis.json"
    if pp.exists():
        res["proxy"] = json.loads(pp.read_text())
        lines.append("")
        lines.append(f"G1/G2 from {args.proxy}: see arms/val500 sections")
        for name, r in res["proxy"].get("arms", {}).items():
            s = (f"  {name:10s} A10-A00 {r['A10_minus_A00'][0]:+.4f} [{r['A10_minus_A00'][1]:+.4f},"
                 f"{r['A10_minus_A00'][2]:+.4f}] wshift {r['wshift_v'][0]:.3f}")
            if "wshift_ratio_vs_ref" in r:
                s += f" ratio vs ref {r['wshift_ratio_vs_ref'][0]:.3f}"
            lines.append(s)
        for name, r in res["proxy"].get("val500", {}).items():
            lines.append(f"  val500 {name:10s} dADE {r['d_ade'][0]:+.4f} [{r['d_ade'][1]:+.4f},"
                         f"{r['d_ade'][2]:+.4f}] degen {r['degen']:.3f}")

    # --- G3b LingoQA
    if args.lingo:
        lp = REPO / "outputs" / args.lingo / "metrics.json"
        if lp.exists():
            m = json.loads(lp.read_text())
            res["lingo"] = {"per_run": m["per_run"], "comparisons": m["comparisons"]}
            lines.append("")
            lines.append(f"G3b LingoQA ({args.lingo}):")
            for run, r in m["per_run"].items():
                lines.append(f"  {run:32s} acc {100 * r['accuracy']:.1f}%")
            for c in m["comparisons"]:
                lines.append(f"  {c['a']} - {c['b']}: {100 * c['delta']:+.1f}pp "
                             f"[{100 * c['ci_lo']:+.1f}, {100 * c['ci_hi']:+.1f}] McNemar p={c['mcnemar_p']:.2g}")

    for name, sup in ARMS.items():
        mp = REPO / "outputs" / sup / "metadata.json"
        if mp.exists() and name in res["recon"]:
            md = json.loads(mp.read_text())
            res["recon"][name]["tokens"] = md.get("tokens", {})
            res["recon"][name]["lingo"] = md.get("lingo")
            res["recon"][name]["damp_escalations"] = sum(
                1 for v in md.get("recon_errors", {}).values() if v.get("damp", 0.01) > 0.01)
    if res["recon"]:
        # the template's {{recon.tokens.*}} read the last extra arm's token counts
        last = list(ARMS)[-1]
        res["recon"]["tokens"] = res["recon"].get(last, {}).get("tokens", {})
    if args.lingo_vs_w:
        lp = REPO / "outputs" / args.lingo_vs_w / "metrics.json"
        if lp.exists():
            m = json.loads(lp.read_text())
            for c in m["comparisons"]:
                if "dualr_wl" in c["a"]:
                    res["wl_vs_w"] = c
                    lines.append(f"wl - w (LingoQA, baseline w): {100 * c['delta']:+.1f}pp "
                                 f"[{100 * c['ci_lo']:+.1f}, {100 * c['ci_hi']:+.1f}] p={c['mcnemar_p']:.2g}")
    text = "\n".join(lines)
    print(text)
    (out / "dualrw_summary.txt").write_text(text + "\n")
    (out / "metrics_analysis.json").write_text(json.dumps(res, indent=1))

    if errs and mods:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, kind in zip(axes, ("o_proj", "down_proj")):
            ms = [m for m in mods if kind in m]
            layers = [int(m.split(".")[1]) for m in ms]
            for a, color in zip(list(ARMS), (C1, C2, C3, C4, "#7b3fa0", MUTED)):
                if a in errs:
                    ax.plot(layers, [errs[a][m]["err_D_refit"] for m in ms], "o-", ms=3,
                            color=color, label=f"{a} refit")
            ax.plot(layers, [errs["rep"][m]["err_D_mask"] for m in ms], "k--", lw=1,
                    label="mask-only")
            ax.set_title(f"{kind}: decode-stream reconstruction error by layer")
            ax.set_xlabel("VLM layer")
            ax.set_ylabel("rel err on own-CoC tokens")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "plots" / "dualrw_decode_error.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
