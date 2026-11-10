"""Combination rules under the paper's pre-l0keep recipe: rank-max (MoOP), rank-mean, raw-mean, raw-max -- test500 and LingoQA.

Plan: iclr27-rebuttal/plans/2026-11-08_comb-rules-pre-l0keep-test500.md (amended 2026-11-10: LingoQA added, rank-mean re-run).
All rows Ada, model revision 7aba829.
  test500  k=8, seed 42 per clip, max_gen 256, the 500 clip IDs of manifest test_500. Per arm: minADE@6 / minFDE@6 (first six of
           eight samples) at 1.6 / 3.2 / 6.4 s, CoC degeneracy; paired delta vs rank-max (bootstrap 200,000, 95% CI of the
           mean; Wilcoxon p for minADE@6 6.4 s), per-bucket delta, each arm's layer-0 Q heads (the pre-l0keep confound).
           The rank-mean re-run is compared clip by clip with the September run (reproducibility).
  LingoQA  Lingo-Judge accuracy over 500 questions (scored.json); CIs resample the 100 segments
           (ci_openloop_lingoqa.lingo_block); paired deltas vs rank-max and dense, and vs the always-"yes" floor.
Arms whose runs are missing are skipped, so the script also works as a progress check.

Usage: .venv/bin/python iclr27-rebuttal/code/analyze_comb_pre_l0keep.py [--exp-id iclr27-rebuttal/comb_pre_l0keep_analysis]
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
sys.path.insert(0, str(REPO / "iclr27-rebuttal" / "code"))
from analyze_early_mlp_openloop import DRAWS, INK, MUTED, at6, boot, rows
from ci_openloop_lingoqa import lingo_block

C = "iclr27-rebuttal/comb_pre_l0keep"
# label -> (test500 run, slim dir, config, LingoQA run dir)
ARMS = {"rank-max (MoOP)": ("dual_u40_v2_ps_test", "slim_dual_u40_v2", "dual_u40_v2", "lingo_vqa_slim_dual_u40_v2"),
        "rank-mean": (f"{C}/rank_mean_test500", "slim_dualsum_u40_v2", "dualsum_u40_v2", f"{C}/lingo_vqa_rank_mean"),
        "raw-mean": (f"{C}/raw_mean_test500", f"{C}/slim_rawabsavg_u40_v2", "rawabsavg_u40_v2", f"{C}/lingo_vqa_raw_mean"),
        "raw-max": (f"{C}/raw_max_test500", f"{C}/slim_rawabsmean_u40_v2", "rawabsmean_u40_v2", f"{C}/lingo_vqa_raw_max")}
REF = "rank-max (MoOP)"
DENSE = ("baseline_ada_ps_test", "lingo_vqa_baseline")
RANK_MEAN_SEPT = "dualsum_u40_v2_test"
HORIZONS = {"1.6s": "_h16", "3.2s": "_h32", "6.4s": ""}
BUCKETS = ("cruise", "decel_stop", "accel", "turn")


def readouts(rr, ids):
    out = {}
    for h, suf in HORIZONS.items():
        out[f"ade_{h}"] = at6(rr, ids, "ade_rollout_k" + suf)  # (n,)
        out[f"fde_{h}"] = at6(rr, ids, "fde_rollout_k" + suf)
    out["degen"] = np.array([float(rr[c]["coc_degenerate"]) for c in ids])
    return out


def reproducibility(new, old):
    """clips whose rollout samples and generated CoC match the September run exactly"""
    keys = ("ade_rollout_k", "fde_rollout_k", "gen_coc")
    same = sum(all(new[c][k] == old[c][k] for k in keys) for c in new)
    return {"clips": len(new), "identical": same, "fields": list(keys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="iclr27-rebuttal/comb_pre_l0keep_analysis")
    ap.add_argument("--rank-mean-run", default=None, help="override the rank-mean test500 run (e.g. the September one)")
    args = ap.parse_args()
    out = REPO / "outputs" / args.exp_id
    out.mkdir(parents=True, exist_ok=True)
    arms = dict(ARMS)
    if args.rank_mean_run:
        arms["rank-mean"] = (args.rank_mean_run, *arms["rank-mean"][1:])

    # ---- test500
    data = {k: rows(v[0]) for k, v in arms.items()}
    data = {k: v for k, v in data.items() if len(v) == 500}
    dense = rows(DENSE[0])
    ids = sorted(data[REF])
    assert all(set(v) == set(ids) for v in data.values()) and set(dense) == set(ids), "clip sets differ"
    bucket = {c: data[REF][c]["bucket"] for c in ids}
    ro = {k: readouts(v, ids) for k, v in data.items()}
    ro_dense = readouts(dense, ids)
    rng = np.random.default_rng(0)
    ref_meta = json.loads((REPO / "outputs" / arms[REF][1] / "slim_meta.json").read_text())

    res = {}
    for k, r in ro.items():
        meta = json.loads((REPO / "outputs" / arms[k][1] / "slim_meta.json").read_text())
        l0 = sorted(meta["vlm"][0]["q"])
        e = {"run": arms[k][0], "config": arms[k][2], "removed_params": meta["params"]["removed"],
             "layer0_q_kept": l0, "layer0_shared_with_ref": len(set(l0) & set(ref_meta["vlm"][0]["q"])),
             "mean": {m: float(v.mean()) for m, v in r.items()}, "median_ade_6.4s": float(np.median(r["ade_6.4s"])),
             "delta_dense_ade_6.4s": float((r["ade_6.4s"] - ro_dense["ade_6.4s"]).mean())}
        if k != REF:
            e["delta_ref"] = {}
            for m in ("ade_1.6s", "ade_3.2s", "ade_6.4s", "fde_1.6s", "fde_3.2s", "fde_6.4s", "degen"):
                d = r[m] - ro[REF][m]
                e["delta_ref"][m] = {"mean": float(d.mean()), "ci": boot(d, rng)}
            d = r["ade_6.4s"] - ro[REF]["ade_6.4s"]
            e["delta_ref"]["ade_6.4s"]["median"] = float(np.median(d))
            e["delta_ref"]["ade_6.4s"]["wilcoxon_p"] = float(wilcoxon(d).pvalue)
            e["delta_ref_bucket"] = {b: {"n": int(sum(bucket[c] == b for c in ids)),
                                         "mean": float(np.mean([d[i] for i, c in enumerate(ids) if bucket[c] == b]))}
                                     for b in BUCKETS}
        res[k] = e
    repro = None
    if "rank-mean" in data and arms["rank-mean"][0] != RANK_MEAN_SEPT:
        repro = reproducibility(data["rank-mean"], rows(RANK_MEAN_SEPT))

    # ---- LingoQA
    ldirs = {k: REPO / "outputs" / v[3] for k, v in arms.items()}
    ldirs["dense"] = REPO / "outputs" / DENSE[1]
    ldirs = {k: d for k, d in ldirs.items() if (d / "scored.json").exists()}
    lingo = lingo_block(ldirs, [REF, "dense"]) if REF in ldirs else None

    (out / "config.json").write_text(json.dumps({
        "arms": {k: {"run": v[0], "slim": v[1], "config": v[2], "lingoqa": v[3]} for k, v in arms.items()},
        "reference": REF, "dense": {"test500": DENSE[0], "lingoqa": DENSE[1]}, "rank_mean_september": RANK_MEAN_SEPT,
        "metric": "minADE@6 / minFDE@6 = min over the first 6 of 8 samples; LingoQA = Lingo-Judge accuracy, segment bootstrap",
        "horizons": list(HORIZONS), "bootstrap_draws": DRAWS, "seed": 0,
        "plan": "iclr27-rebuttal/plans/2026-11-08_comb-rules-pre-l0keep-test500.md"}, indent=2))
    (out / "metrics.json").write_text(json.dumps({"dense_mean": {m: float(v.mean()) for m, v in ro_dense.items()},
                                                  "arms": res, "rank_mean_reproducibility": repro, "lingoqa": lingo},
                                                 indent=2))

    L = ["Combination rules, pre-l0keep recipe (layer 0 cut 13/32 like every layer), Ada", "", "test500 (500 clips)",
         (f"dense: minADE@6 {ro_dense['ade_1.6s'].mean():.3f} / {ro_dense['ade_3.2s'].mean():.3f} / "
          f"{ro_dense['ade_6.4s'].mean():.3f}  minFDE@6 {ro_dense['fde_1.6s'].mean():.3f} / {ro_dense['fde_3.2s'].mean():.3f} / "
          f"{ro_dense['fde_6.4s'].mean():.3f}"), "",
         (f"{'rule':<17}{'minADE@6 1.6/3.2/6.4 s':>26}{'minFDE@6 1.6/3.2/6.4 s':>26}{'median':>8}{'degen':>7}"
          f"{'  dADE@6 6.4s vs rank-max [CI]':<34}{'dFDE@6 6.4s [CI]':<26}{'p':>8}")]
    for k, e in res.items():
        mm = e["mean"]
        s = (f"{k:<17}{mm['ade_1.6s']:>8.3f}{mm['ade_3.2s']:>9.3f}{mm['ade_6.4s']:>9.3f}{mm['fde_1.6s']:>8.3f}"
             f"{mm['fde_3.2s']:>9.3f}{mm['fde_6.4s']:>9.3f}{e['median_ade_6.4s']:>8.3f}{mm['degen'] * 100:>6.1f}%")
        if k != REF:
            a, f = e["delta_ref"]["ade_6.4s"], e["delta_ref"]["fde_6.4s"]
            s += (f"  {a['mean']:+.3f} [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}]{'':<8}"
                  f"{f['mean']:+.3f} [{f['ci'][0]:+.3f}, {f['ci'][1]:+.3f}]{a['wilcoxon_p']:>10.2g}")
        L.append(s)
    L += ["", "delta vs rank-max at 1.6 / 3.2 s (minADE@6, minFDE@6):"]
    for k, e in res.items():
        if k != REF:
            dd = e["delta_ref"]
            L.append(f"  {k:<15}" + "  ".join(
                f"{m} {dd[m]['mean']:+.3f} [{dd[m]['ci'][0]:+.3f}, {dd[m]['ci'][1]:+.3f}]"
                for m in ("ade_1.6s", "ade_3.2s", "fde_1.6s", "fde_3.2s")))
    L += ["", "delta minADE@6 6.4 s vs rank-max by bucket (n):"]
    for k, e in res.items():
        if k != REF:
            L.append(f"  {k:<15}" + "  ".join(f"{b} {v['mean']:+.3f} ({v['n']})" for b, v in e["delta_ref_bucket"].items()))
    L += ["", "layer-0 Q heads kept (19 of 32; the pre-l0keep confound):"]
    for k, e in res.items():
        L.append(f"  {k:<15}{e['layer0_q_kept']}  shared with rank-max {e['layer0_shared_with_ref']}/19")
    if repro:
        L += ["", (f"rank-mean re-run vs September run ({RANK_MEAN_SEPT}): {repro['identical']}/{repro['clips']} clips identical "
                  f"on {repro['fields']}")]
    if lingo:
        st = lingo["stats"]
        L += ["", (f"LingoQA ({lingo['n_questions']} questions, {lingo['n_segments']} segments; always-yes floor "
                  f"{lingo['floor_yes_accuracy'] * 100:.1f}%)"),
              f"{'arm':<17}{'acc %':>7}  {'[CI]':<16}{'d vs rank-max pp [CI]':<28}{'d vs dense pp [CI]':<28}{'d vs floor pp':<14}"]
        for k in [*[a for a in arms if a in ldirs], "dense"]:
            if k not in ldirs:
                continue
            a = st[f"{k}|abs"]
            s = f"{k:<17}{a['mean'] * 100:>7.1f}  [{a['ci'][0] * 100:.1f}, {a['ci'][1] * 100:.1f}]   "
            for ref in (REF, "dense", "floor_yes"):
                key = f"{k}|{ref}"
                s += (f"{st[key]['mean'] * 100:+.1f} [{st[key]['ci'][0] * 100:+.1f}, {st[key]['ci'][1] * 100:+.1f}] p={st[key]['p']:.2g}  "
                      if key in st else f"{'—':<28}")
            L.append(s)
    (out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    others = [k for k in res if k != REF]
    if not others:
        print(f"wrote {out} (no rule rows yet)")
        return
    lrows = [k for k in others if lingo and f"{k}|{REF}" in lingo["stats"]]
    ncol = 3 if lrows else 2
    fig, axes = plt.subplots(1, ncol, figsize=(4.5 * ncol, 0.6 + 0.55 * len(others)), sharey=True)
    for ax, m, lab in ((axes[0], "ade_6.4s", "minADE@6 6.4 s (m)"), (axes[1], "fde_6.4s", "minFDE@6 6.4 s (m)")):
        for i, k in enumerate(others):
            d = res[k]["delta_ref"][m]
            ax.errorbar(d["mean"], i, xerr=[[d["mean"] - d["ci"][0]], [d["ci"][1] - d["mean"]]], fmt="o", color=INK,
                        capsize=3)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_xlabel(f"{lab}: rule − rank-max")
    if lrows:
        for k in lrows:
            i = others.index(k)
            d = lingo["stats"][f"{k}|{REF}"]
            axes[2].errorbar(d["mean"] * 100, i, xerr=[[(d["mean"] - d["ci"][0]) * 100], [(d["ci"][1] - d["mean"]) * 100]],
                             fmt="o", color=INK, capsize=3)
        axes[2].axvline(0, color=MUTED, lw=1)
        axes[2].set_xlabel("LingoQA accuracy (pp): rule − rank-max")
    axes[0].set_yticks(range(len(others)))
    axes[0].set_yticklabels(others)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "delta_vs_rankmax.png", dpi=180)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
