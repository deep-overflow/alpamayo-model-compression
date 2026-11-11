"""Alpamayo 2 Super on test500: dense vs MoOP (ICLR27 recipe) vs MoOP l0keep, open loop.

Plan: iclr27-rebuttal/plans/2026-11-09_a2-moop-test500.md. Rows come from run_a2_openloop.py (alpamayo2_s{0,1}of2.json per arm).
Readouts per arm: minADE@6 / minFDE@6 (first six of eight samples) at 1.6 / 3.2 / 6.4 s, median minADE@6 6.4 s, CoC
degeneracy, generated length. Paired deltas (MoOP - dense, l0keep - dense, l0keep - MoOP) with bootstrap 200,000 (95% CI
of the mean, and of the median for minADE@6 6.4 s) and Wilcoxon p; per-bucket deltas. The A1.5 test500 rows (Ada, single
card, same clip IDs) are shown as context only — a different model on different hardware.

--partial evaluates on the clips every available arm has so far (progress check); without it all 500 are required.

--preset u50 (plans/2026-11-09_a2-u50-test500.md): the u50 arms and their own dense, all on Blackwell (GPUs 0-3, `_bw`), with
the Ada u40 rows as context; the pairs are against Blackwell dense, plus dense_bw - dense Ada as the measured hardware shift.

--preset u40_oodval / u40_oodtrain (plans/2026-11-10_a2-u40-ood-gh200.md): the u40 arms on OOD-Val262 / OOD-train1271, all on
KISTI Hangang GH200 (outputs/iclr27-rebuttal/a2_gh200, pulled by pull_a2_ood_gh200.sh), with A1.5 rows as context (OOD-train: the
11-09 GH200 run; OOD-Val: the paper's Ada rows).

--preset u50q8_test500 / u50q8_oodval / u50q8_oodtrain (plans/2026-11-11_a2-u50-q8-gh200.md): u50 (32 Q heads + 12,800 MLP
channels cut per layer) vs u50-q8 (8 heads + 14,848 channels, the same 15,267,266,560 parameters) for MoOP / FM-only / CE-only,
all on GH200 with a GH200 dense (OOD: job 1251209's rows, or dense_r if the reproducibility check failed); OOD adds the GH200 u40
rows, test500 the Blackwell u50 rows, both as context.

Usage: .venv/bin/python iclr27-rebuttal/code/analyze_a2_test500.py [--preset u40|u50|u40_oodval|u40_oodtrain|u50q8_<set>] [--partial]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "cvpr27" / "code"))
from analyze_early_mlp_openloop import DRAWS, INK, MUTED, at6, boot, rows

A2_ARMS = {"A2 dense": "iclr27-rebuttal/a2_test500_dense",
           "A2 + MoOP": "iclr27-rebuttal/a2_test500_moop",
           "A2 + MoOP l0keep": "iclr27-rebuttal/a2_test500_moop_l0keep",
           "A2 + FM-only": "iclr27-rebuttal/a2_test500_fm_only",
           "A2 + CE-only": "iclr27-rebuttal/a2_test500_ce_only"}
A15_ARMS = {"A1.5 dense": "baseline_ada_ps_test", "A1.5 + MoOP": "dual_u40_v2_ps_test",
            "A1.5 + MoOP l0keep": "swap_ship_L0fullM_500",
            "A1.5 + FM-only": "traj_u40_v2_test", "A1.5 + CE-only": "coc_u40_v2_test"}
PAIRS = [("A2 + MoOP", "A2 dense"), ("A2 + MoOP l0keep", "A2 dense"), ("A2 + FM-only", "A2 dense"),
         ("A2 + CE-only", "A2 dense"), ("A2 + MoOP l0keep", "A2 + MoOP"), ("A2 + MoOP", "A2 + FM-only"),
         ("A2 + MoOP", "A2 + CE-only"),
         ("A1.5 + MoOP", "A1.5 dense"), ("A1.5 + MoOP l0keep", "A1.5 dense"), ("A1.5 + FM-only", "A1.5 dense"),
         ("A1.5 + CE-only", "A1.5 dense"), ("A1.5 + MoOP", "A1.5 + FM-only"), ("A1.5 + MoOP", "A1.5 + CE-only")]
HORIZONS = {"1.6s": "_h16", "3.2s": "_h32", "6.4s": ""}
BUCKETS = ("cruise", "decel_stop", "accel", "turn")
REMOVED = {**{k: "12.21B / 35.81B (34.1%)" for k in A2_ARMS if k != "A2 dense"},
           **{k: "2.66B / 11.08B (24.0%)" for k in A15_ARMS if k != "A1.5 dense"}}

U50_ARMS = {"A2 dense (BW)": "iclr27-rebuttal/a2_test500_dense_bw",
            "A2 + MoOP u50": "iclr27-rebuttal/a2_test500_moop_u50_bw",
            "A2 + FM-only u50": "iclr27-rebuttal/a2_test500_fm_only_u50_bw",
            "A2 + CE-only u50": "iclr27-rebuttal/a2_test500_ce_only_u50_bw"}
U40_ADA = {"A2 dense (Ada)": A2_ARMS["A2 dense"], "A2 + MoOP u40 (Ada)": A2_ARMS["A2 + MoOP"],
           "A2 + FM-only u40 (Ada)": A2_ARMS["A2 + FM-only"], "A2 + CE-only u40 (Ada)": A2_ARMS["A2 + CE-only"]}
PRESETS = {
    "u40": {"arms": A2_ARMS, "context": A15_ARMS, "pairs": PAIRS, "removed": REMOVED,
            "exp_id": "iclr27-rebuttal/a2_test500_analysis", "plan": "iclr27-rebuttal/plans/2026-11-09_a2-moop-test500.md",
            "title": "Alpamayo 2 Super, test500 open loop"},
    "u50": {"arms": U50_ARMS, "context": U40_ADA,
            "pairs": [("A2 + MoOP u50", "A2 dense (BW)"), ("A2 + FM-only u50", "A2 dense (BW)"),
                      ("A2 + CE-only u50", "A2 dense (BW)"), ("A2 + MoOP u50", "A2 + FM-only u50"),
                      ("A2 + MoOP u50", "A2 + CE-only u50"), ("A2 dense (BW)", "A2 dense (Ada)"),
                      ("A2 + MoOP u40 (Ada)", "A2 dense (Ada)"), ("A2 + FM-only u40 (Ada)", "A2 dense (Ada)"),
                      ("A2 + CE-only u40 (Ada)", "A2 dense (Ada)"),
                      # u50 - u40 of the same criterion: across cards, valid because dense_bw - dense_ada is ~0
                      ("A2 + MoOP u50", "A2 + MoOP u40 (Ada)"), ("A2 + FM-only u50", "A2 + FM-only u40 (Ada)"),
                      ("A2 + CE-only u50", "A2 + CE-only u40 (Ada)")],
            "removed": {**{k: "15.27B / 35.81B (42.6%)" for k in U50_ARMS if "dense" not in k},
                        **{k: "12.21B / 35.81B (34.1%)" for k in U40_ADA if "dense" not in k}},
            "exp_id": "iclr27-rebuttal/a2_test500_u50_analysis", "plan": "iclr27-rebuttal/plans/2026-11-09_a2-u50-test500.md",
            "title": "Alpamayo 2 Super u50 (Blackwell) + u40 (Ada context), test500 open loop"},
}
G = "iclr27-rebuttal/a2_gh200"
OOD_PAIRS = [("A2 + MoOP", "A2 dense"), ("A2 + FM-only", "A2 dense"), ("A2 + CE-only", "A2 dense"),
             ("A2 + MoOP", "A2 + FM-only"), ("A2 + MoOP", "A2 + CE-only"),
             ("A1.5 + MoOP", "A1.5 dense"), ("A1.5 + FM-only", "A1.5 dense"), ("A1.5 + CE-only", "A1.5 dense"),
             ("A1.5 + MoOP", "A1.5 + FM-only"), ("A1.5 + MoOP", "A1.5 + CE-only")]
for _set, _n, _ctx, _hw in (
        ("oodval", 262, {"A1.5 dense": "baseline_pred_oodval", "A1.5 + MoOP": "dual_u40_v2_pred_oodval",
                         "A1.5 + FM-only": "iclr27-rebuttal/ada_fill/fm_only_oodval262",
                         "A1.5 + CE-only": "iclr27-rebuttal/ada_fill/ce_only_oodval262"}, "A1.5 rows Ada"),
        ("oodtrain", 1271, {"A1.5 dense": "iclr27-rebuttal/gh200/dense_oodtrain",
                            "A1.5 + MoOP": "iclr27-rebuttal/gh200/moop_oodtrain",
                            "A1.5 + FM-only": "iclr27-rebuttal/gh200/fm_only_oodtrain",
                            "A1.5 + CE-only": "iclr27-rebuttal/gh200/ce_only_oodtrain"}, "A1.5 rows GH200")):
    PRESETS[f"u40_{_set}"] = {
        "arms": {"A2 dense": f"{G}/a2_{_set}_dense", "A2 + MoOP": f"{G}/a2_{_set}_moop",
                 "A2 + FM-only": f"{G}/a2_{_set}_fm_only", "A2 + CE-only": f"{G}/a2_{_set}_ce_only"},
        "context": _ctx, "pairs": OOD_PAIRS, "n": _n,
        "removed": {**{k: "12.21B / 35.81B (34.1%)" for k in ("A2 + MoOP", "A2 + FM-only", "A2 + CE-only")},
                    **{k: "2.66B / 11.08B (24.0%)" for k in ("A1.5 + MoOP", "A1.5 + FM-only", "A1.5 + CE-only")}},
        "exp_id": f"iclr27-rebuttal/a2_{_set}_u40_analysis", "plan": "iclr27-rebuttal/plans/2026-11-10_a2-u40-ood-gh200.md",
        "title": f"Alpamayo 2 Super u40 on {'OOD-Val262' if _set == 'oodval' else 'OOD-train1271'}, GH200 ({_hw}, context)"}

_CRIT = (("MoOP", "moop"), ("FM-only", "fm_only"), ("CE-only", "ce_only"))
for _set, _n, _name in (("test500", 500, "test500"), ("oodval", 262, "OOD-Val262"), ("oodtrain", 1271, "OOD-train1271")):
    _dense = f"{G}/a2_{_set}_dense_r" if (REPO / "outputs" / G / f"a2_{_set}_dense_r").exists() else f"{G}/a2_{_set}_dense"
    _arms = {"A2 dense": _dense}
    for _alloc, _sfx in (("u50", "_u50"), ("u50-q8", "_u50_q8")):
        for _lab, _key in _CRIT:
            _arms[f"A2 + {_lab} {_alloc}"] = f"{G}/a2_{_set}_{_key}{_sfx}"
    if _set == "test500":
        _ctx = {f"A2 + {_lab} u50 (BW)": f"iclr27-rebuttal/a2_test500_{_key}_u50_bw" for _lab, _key in _CRIT}
    else:
        _ctx = {f"A2 + {_lab} u40": f"{G}/a2_{_set}_{_key}" for _lab, _key in _CRIT}
    _pairs = [(a, "A2 dense") for a in _arms if a != "A2 dense"]
    _pairs += [(f"A2 + {_lab} u50-q8", f"A2 + {_lab} u50") for _lab, _ in _CRIT]
    for _alloc in ("u50", "u50-q8"):
        _pairs += [(f"A2 + MoOP {_alloc}", f"A2 + FM-only {_alloc}"), (f"A2 + MoOP {_alloc}", f"A2 + CE-only {_alloc}")]
    if _set != "test500":
        _pairs += [(f"A2 + {_lab} u50", f"A2 + {_lab} u40") for _lab, _ in _CRIT]
    PRESETS[f"u50q8_{_set}"] = {
        "arms": _arms, "context": _ctx, "pairs": _pairs, "n": _n,
        "removed": {**{k: "15.27B / 35.81B (42.6%)" for k in _arms if k != "A2 dense"},
                    **{k: ("15.27B / 35.81B (42.6%)" if "u50" in k else "12.21B / 35.81B (34.1%)") for k in _ctx}},
        "exp_id": f"iclr27-rebuttal/a2_u50q8_{_set}_analysis", "plan": "iclr27-rebuttal/plans/2026-11-11_a2-u50-q8-gh200.md",
        "title": f"Alpamayo 2 Super u50 vs u50-q8 on {_name}, GH200 (context: {'Blackwell u50' if _set == 'test500' else 'GH200 u40'})"}


def readouts(rr, ids):
    """per-clip readouts; a horizon the run did not store (older A1.5 rows: no _h16/_h32) is NaN"""
    out = {}
    one = rr[ids[0]]
    for h, suf in HORIZONS.items():
        for kind in ("ade", "fde"):
            key = f"{kind}_rollout_k{suf}"
            out[f"{kind}_{h}"] = at6(rr, ids, key) if key in one else np.full(len(ids), np.nan)  # (n,)
    out["degen"] = np.array([float(rr[c]["coc_degenerate"]) for c in ids])
    out["gen_len"] = np.array([float(rr[c].get("gen_len", np.nan)) for c in ids])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="u40", choices=sorted(PRESETS))
    ap.add_argument("--exp-id", default=None, help="default: the preset's")
    ap.add_argument("--partial", action="store_true")
    args = ap.parse_args()
    P = PRESETS[args.preset]
    arms, context, pairs = P["arms"], P["context"], P["pairs"]
    out = REPO / "outputs" / (args.exp_id or P["exp_id"])
    out.mkdir(parents=True, exist_ok=True)

    data = {k: rows(v) for k, v in {**arms, **context}.items()}
    data = {k: v for k, v in data.items() if v}
    a2_present = [k for k in arms if k in data]
    if args.partial:
        ids = sorted(set.intersection(*(set(data[k]) for k in a2_present)))
    else:
        n_exp = P.get("n", 500)
        assert all(len(data.get(k, {})) == n_exp for k in arms), {k: len(data.get(k, {})) for k in arms}
        ids = sorted(data[a2_present[0]])
    for k in context:
        assert set(ids) <= set(data[k]), k
    ro = {k: readouts(v, ids) for k, v in data.items()}
    bucket = {c: data[a2_present[0]][c]["bucket"] for c in ids}
    rng = np.random.default_rng(0)

    res = {"n_clips": len(ids), "partial": args.partial, "arms": {}, "pairs": {}}
    for k, r in ro.items():
        res["arms"][k] = {"mean": {m: float(np.nanmean(v)) for m, v in r.items()},
                          "median_ade_6.4s": float(np.median(r["ade_6.4s"]))}
    for a, b in pairs:
        if a not in ro or b not in ro:
            continue
        e = {}
        for m in ("ade_1.6s", "ade_3.2s", "ade_6.4s", "fde_1.6s", "fde_3.2s", "fde_6.4s", "degen"):
            d = ro[a][m] - ro[b][m]
            e[m] = ({"mean": float("nan"), "ci": [float("nan"), float("nan")]} if np.isnan(d).any()
                    else {"mean": float(d.mean()), "ci": boot(d, rng)})
        d = ro[a]["ade_6.4s"] - ro[b]["ade_6.4s"]
        e["ade_6.4s"].update({"median": float(np.median(d)), "median_ci": boot(d, rng, stat=np.median),
                              "wilcoxon_p": float(wilcoxon(d).pvalue), "rel_mean": float(d.mean() / ro[b]["ade_6.4s"].mean())})
        e["bucket"] = {bk: {"n": int(sum(bucket[c] == bk for c in ids)),
                            "mean": float(np.mean([d[i] for i, c in enumerate(ids) if bucket[c] == bk]))}
                       for bk in BUCKETS}
        res["pairs"][f"{a} - {b}"] = e

    (out / "config.json").write_text(json.dumps({
        **({"a2_arms": arms, "a15_context": context} if args.preset == "u40" else
           {"preset": args.preset, "arms": arms, "context": context}),
        "pairs": pairs, "removed": P["removed"],
        "metric": "minADE@6 / minFDE@6 = min over the first 6 of 8 samples", "bootstrap_draws": DRAWS, "seed": 0,
        "plan": P["plan"]}, indent=2))
    (out / "metrics.json").write_text(json.dumps(res, indent=2))

    w = max(20, max(len(k) for k in [*arms, *context]) + 2)
    L = [f"{P['title']} ({len(ids)} clips{', PARTIAL' if args.partial else ''})", "",
         (f"{'arm':<{w}}{'minADE@6 1.6/3.2/6.4 s':>26}{'minFDE@6 1.6/3.2/6.4 s':>26}{'median':>8}{'degen':>7}"
          f"{'gen tok':>8}")]
    for k in [*arms, *context]:
        if k not in ro:
            continue
        mm = res["arms"][k]["mean"]
        L.append(f"{k:<{w}}{mm['ade_1.6s']:>8.3f}{mm['ade_3.2s']:>9.3f}{mm['ade_6.4s']:>9.3f}{mm['fde_1.6s']:>8.3f}"
                 f"{mm['fde_3.2s']:>9.3f}{mm['fde_6.4s']:>9.3f}{res['arms'][k]['median_ade_6.4s']:>8.3f}"
                 f"{mm['degen'] * 100:>6.1f}%{mm['gen_len']:>8.1f}")
    L += ["", "paired deltas (95% CI, bootstrap 200k), minADE@6 / minFDE@6 at 6.4 s:"]
    for name, e in res["pairs"].items():
        a, f = e["ade_6.4s"], e["fde_6.4s"]
        L.append(f"  {name:<36} dADE {a['mean']:+.3f} [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}] ({a['rel_mean']:+.1%}) "
                 f"median {a['median']:+.3f} [{a['median_ci'][0]:+.3f}, {a['median_ci'][1]:+.3f}] p={a['wilcoxon_p']:.2g} | "
                 f"dFDE {f['mean']:+.3f} [{f['ci'][0]:+.3f}, {f['ci'][1]:+.3f}] | ddegen {e['degen']['mean'] * 100:+.1f}%p")
        L.append(f"  {'':<36} 1.6 s dADE {e['ade_1.6s']['mean']:+.3f} [{e['ade_1.6s']['ci'][0]:+.3f}, "
                 f"{e['ade_1.6s']['ci'][1]:+.3f}]  3.2 s dADE {e['ade_3.2s']['mean']:+.3f} [{e['ade_3.2s']['ci'][0]:+.3f}, "
                 f"{e['ade_3.2s']['ci'][1]:+.3f}]  buckets " + " ".join(
                     f"{bk} {v['mean']:+.3f}({v['n']})" for bk, v in e["bucket"].items()))
    (out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    shown = [p for p in res["pairs"] if p.startswith("A2")]
    if shown:
        fig, axes = plt.subplots(1, 2, figsize=(10, 0.8 + 0.6 * len(shown)), sharey=True)
        for ax, m, lab in ((axes[0], "ade_6.4s", "minADE@6 6.4 s"), (axes[1], "fde_6.4s", "minFDE@6 6.4 s")):
            for i, p in enumerate(shown):
                d = res["pairs"][p][m]
                ax.errorbar(d["mean"], i, xerr=[[d["mean"] - d["ci"][0]], [d["ci"][1] - d["mean"]]], fmt="o",
                            color=INK, capsize=3)
            ax.axvline(0, color=MUTED, lw=1)
            ax.set_xlabel(f"Δ {lab} (m), mean and 95% CI")
        axes[0].set_yticks(range(len(shown)))
        axes[0].set_yticklabels(shown)
        axes[0].invert_yaxis()
        fig.tight_layout()
        fig.savefig(out / "deltas.png", dpi=180)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
