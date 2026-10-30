"""MoOP (slim_dual_u40_v2) vs dense Alpamayo 1.5 in closed loop, paired per scene on the s600 sample.

Plan: plans/2026-10-28_dual-closedloop-s600.md (§3). Both arms ran on the patched AlpaSim (hand-over fix), one
rollout per scene, every scene on the same card in both arms:
  dense  alpasim-runs/s600_merged_baseline          (analysis: outputs/alpasim_dense_s600)
  MoOP   alpasim-runs/d600_merged_slim_dual_u40_v2

  * paired difference MoOP - dense per scene, stratified over the 9 sampling cells (weight = cell size / 912),
    CI from a bootstrap within cells, Wilcoxon alongside; each arm's own stratified mean is reported first
  * scores re-derived from metrics.parquet (gate G0) for the end-of-route-aware 4 m variant (`score_dev`)
  * aborted rollouts (forced 0.0) per arm, and the difference without the union of their scenes
  * subsets: Origin150 / Hard100 / rest, and per card architecture
  * CoC degeneracy per rollout (empty or garbled step fraction, the rule of head_analysis/analyze_alpasim.py)
  * reference: the same pair before the hand-over fix (m2601_ on Origin150, h100_ on Hard100, two rollouts
    each) on the scenes this sample shares with them

Reads rollout.asl for the CoC, so run it with alpasim's venv:
  /mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python experiments/evaluation/analyze_dual_s600.py
Writes outputs/alpasim_dual_s600/{config.json, metrics.json, summary.txt, rollouts.parquet, *.png}.
Any other pair of runs on the same pinned s600 lists takes --base/--arm (run names) with labels and --out; the
before-the-fix reference only exists for dense vs MoOP, so other pairs pass --no-pre-fix, e.g.
  ... --arm l600_merged_slim_dual_u40_l0keep_v2 --arm-label "MoOP l0keep" --out outputs/alpasim_l0keep_s600 \
      --no-pre-fix
"""

import argparse
import asyncio
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import analyze_closedloop_gap as G
import analyze_dense_s600 as D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
import analyze_alpasim as AA

RUNS = Path("/home/cvlab21/project/chan/alpasim-runs")
BOUT = REPO / "outputs" / "alpasim_dense_s600"   # scene list, sampling cells and card per scene
PRE = [("origin150", "m2601_merged_baseline", "m2601_merged_slim_dual_u40_v2"),
       ("hard100", "h100_merged_baseline", "h100_merged_slim_dual_u40_v2")]
COLS = ["score", "score_dev", "passed", "progress_clipped_rel", "collision_any", "collision_at_fault", "offroad",
        "dist_traveled_m", "plan_deviation", "coc_degenerate"]


def coc(asl):
    """Degenerate CoC step fraction of one rollout; NaN for an aborted rollout, which leaves no log."""
    return AA.coc_stats(asyncio.run(AA._read_coc(asl)))["degenerate_frac"] if asl.exists() else np.nan


def arm(run, tag):
    """One rollout per scene (the main-stage one if a smoke scene was re-run) with re-derived scores and CoC."""
    ro = D.per_rollout(run)
    ro = ro.assign(main=~ro.run_name.str.contains("smoke")).sort_values("main").groupby("scene").tail(1)
    uuid = pd.read_csv(G.SCENES).set_index("scene_id")["uuid"]
    rec, g0 = [], []
    for r in ro.itertuples():
        p = run / "rollouts" / r.scene / r.rollout / "metrics.parquet"
        a = G.anatomy(p, G.gt_track(uuid[r.scene])) if p.exists() else None
        a = a[0] if a else None
        if a is not None and not np.isnan(r.score):
            g0.append(abs(a["score"] - r.score))
        rec.append({"score_dev": np.nan if a is None else a["score_dev"]})
    ro = pd.concat([ro.reset_index(drop=True), pd.DataFrame(rec)], axis=1)
    asls = [run / "rollouts" / r.scene / r.rollout / "rollout.asl" for r in ro.itertuples()]
    with Pool(12) as pool:
        ro["coc_degenerate"] = pool.map(coc, asls, chunksize=4)
    ro["aborted"] = ~ro.failure_reason.isin(D.GATE_FAIL) & ro.score.notna()
    print(f"  {tag}: {len(ro)} scenes, G0 max {max(g0):.1e}", flush=True)
    return ro.set_index("scene"), float(max(g0))


def pre_fix(rng, scenes):
    """dense - MoOP pair before the hand-over fix, two rollouts each, on the given scenes."""
    out = {}
    for name, b, m in PRE:
        sb = D.per_rollout(RUNS / b).groupby("scene").score.mean()
        sm = D.per_rollout(RUNS / m).groupby("scene").score.mean()
        common = sorted(set(sb.index) & set(sm.index) & set(scenes[name]))
        x = (sm[common] - sb[common]).to_numpy()
        out[name] = {"n": len(common), "dense": float(sb[common].mean()), "moop": float(sm[common].mean()),
                     "diff": D.boot(x, rng), "scenes": common}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="s600_merged_baseline")
    ap.add_argument("--base-label", default="dense")
    ap.add_argument("--arm", default="d600_merged_slim_dual_u40_v2")
    ap.add_argument("--arm-label", default="MoOP")
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "alpasim_dual_s600")
    ap.add_argument("--no-pre-fix", action="store_true", help="skip the dense-vs-MoOP before-the-fix reference")
    ap.add_argument("--exclude", type=Path, default=None,
                    help="CSV with a scene_id column: scenes left out of every estimate (e.g. ones the two arms "
                         "ran on different card types)")
    args = ap.parse_args()
    BASE, ARM, OUT, A, B = RUNS / args.base, RUNS / args.arm, args.out, args.base_label, args.arm_label
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    sc = pd.read_csv(BOUT / "scenes_s600.csv")
    sc["cell"] = sc.sg_level.astype(str) + "/" + sc.hs_tertile
    cells = pd.DataFrame(json.loads((BOUT / "draw.json").read_text())["cells"])
    cells["cell"] = cells.sg_level.astype(str) + "/" + cells.hs_tertile
    sc = sc.merge(cells[["cell", "N"]], on="cell").assign(W=lambda x: x.N / D.FRAME)
    sc = sc.merge(pd.read_csv(BOUT / "scene_gpu.csv")[["scene_id", "gpu", "arch"]], on="scene_id")

    print("reading rollouts", flush=True)
    b, g0b = arm(BASE, A)
    m, g0m = arm(ARM, B)
    b.add_prefix("b_").join(m.add_prefix("m_")).to_parquet(OUT / "rollouts.parquet")
    d = sc.set_index("scene_id")
    for c in COLS + ["aborted"]:
        d["b_" + c], d["m_" + c] = b[c].reindex(d.index), m[c].reindex(d.index)
    for c in COLS:
        d["d_" + c] = d["m_" + c] - d["b_" + c]
    d = d.reset_index()
    excluded = set(pd.read_csv(args.exclude).scene_id) if args.exclude else set()
    d = d[~d.scene_id.isin(excluded)].reset_index(drop=True)

    ab_b, ab_m = set(d.scene_id[d.b_aborted == True]), set(d.scene_id[d.m_aborted == True])
    res = {"n_scenes": len(d), "n_paired": int(d.d_score.notna().sum()), "G0_max": max(g0b, g0m),
           "aborted_base": sorted(ab_b), "aborted_arm": sorted(ab_m)}

    est = {}
    for c in COLS:
        dd = d.dropna(subset=["d_" + c])
        est[c] = {"base": D.strat_boot(d, "b_" + c, rng), "arm": D.strat_boot(d, "m_" + c, rng),
                  "diff": D.strat_boot(d, "d_" + c, rng),
                  "wilcoxon_p": float(wilcoxon(dd["d_" + c]).pvalue) if (dd["d_" + c] != 0).any() else 1.0}
    keep = d[~d.scene_id.isin(ab_b | ab_m)]
    est["score_no_aborted"] = {"base": D.strat_boot(keep, "b_score", rng), "arm": D.strat_boot(keep, "m_score", rng),
                               "diff": D.strat_boot(keep, "d_score", rng), "n": len(keep)}
    res["estimates"] = est

    sub = {}
    for name, msk in (("origin150", d.origin150), ("hard100", d.hard100), ("rest", ~d.origin150 & ~d.hard100),
                      ("blackwell", d.arch == "Blackwell"), ("ada", d.arch == "Ada")):
        g = d[msk]
        sub[name] = {"n": len(g), "base": float(g.b_score.mean()), "arm": float(g.m_score.mean()),
                     "diff": D.boot(g.d_score, rng), "diff_progress": D.boot(g.d_progress_clipped_rel, rng)}
    res["subsets"] = sub
    res["arch_diff_of_diffs"] = (sub["blackwell"]["diff"][0] - sub["ada"]["diff"][0])

    pf = {} if args.no_pre_fix else pre_fix(rng, {"origin150": set(d.scene_id[d.origin150]),
                                                  "hard100": set(d.scene_id[d.hard100])})
    for name, v in pf.items():
        g = d.set_index("scene_id").loc[v.pop("scenes")]
        v["post_fix_same_scenes"] = {"dense": float(g.b_score.mean()), "moop": float(g.m_score.mean()),
                                     "diff": D.boot(g.d_score, rng)}
    res["pre_fix_reference"] = pf

    cfg = {"base": str(BASE), "base_label": A, "arm": str(ARM), "arm_label": B,
           "scenes": str(BOUT / "scenes_s600.csv"), "rollouts_per_scene": 1,
           "cards": "each scene on the same card in both arms (outputs/alpasim_dense_s600/scene_gpu.csv)",
           "bootstrap": D.NB, "excluded_scenes": sorted(excluded)}
    (OUT / "config.json").write_text(json.dumps(cfg, indent=1))
    (OUT / "metrics.json").write_text(json.dumps(res, indent=1, default=float))

    f = D.fmt
    L = [f"{B} vs {A}, paired per scene: {res['n_paired']}/{res['n_scenes']} scenes; G0 max {res['G0_max']:.1e}",
         (f"aborted {A} {len(ab_b)}, {B} {len(ab_m)}, both {len(ab_b & ab_m)}; {B} only "
         f"{sorted(s[7:15] for s in ab_m - ab_b)}; {A} only {sorted(s[7:15] for s in ab_b - ab_m)}"),
         "", f"stratified over 912 scenes [95% CI]:       {A} | {B} | {B} - {A} (Wilcoxon p)",
         *[f"  {k:22s} {f(v['base'])} | {f(v['arm'])} | {f(v['diff'], sign=True)} (p={v['wilcoxon_p']:.2g})"
           for k, v in est.items() if "wilcoxon_p" in v],
         (f"  {'score w/o aborted':22s} {f(est['score_no_aborted']['base'])} | {f(est['score_no_aborted']['arm'])}"
          f" | {f(est['score_no_aborted']['diff'], sign=True)} (n={est['score_no_aborted']['n']})"),
         "", f"subsets (plain means): {A} | {B} | score diff | progress diff",
         *[f"  {k:10s} n={v['n']:3d}  {v['base']:.3f} | {v['arm']:.3f} | {f(v['diff'], sign=True)} | "
           f"{f(v['diff_progress'], sign=True)}" for k, v in sub.items()],
         *(["", "before the hand-over fix (2 rollouts each) vs this run, same scenes: MoOP - dense"] if pf else []),
         *[f"  {k:10s} n={v['n']:3d}  before {v['dense']:.3f} -> {v['moop']:.3f} {f(v['diff'], sign=True)} | after "
           f"{v['post_fix_same_scenes']['dense']:.3f} -> {v['post_fix_same_scenes']['moop']:.3f} "
           f"{f(v['post_fix_same_scenes']['diff'], sign=True)}" for k, v in pf.items()]]
    (OUT / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    rows = [("suite (stratified)", est["score"]["diff"], G.CAT[0]),
            ("  4 m rule ends with the route", est["score_dev"]["diff"], G.CAT[0]),
            ("  without aborted scenes", est["score_no_aborted"]["diff"], G.CAT[0]),
            ("Origin150 scenes", sub["origin150"]["diff"], G.MUTED), ("Hard100 scenes", sub["hard100"]["diff"], G.MUTED),
            ("other scenes", sub["rest"]["diff"], G.MUTED), ("Blackwell cards", sub["blackwell"]["diff"], G.MUTED),
            ("Ada cards", sub["ada"]["diff"], G.MUTED)]
    rows += [(f"before fix, {k}", v["diff"], G.CAT[1]) for k, v in pf.items()]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, (lab, c, col) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([c[1], c[2]], [y, y], color=col, linewidth=2, solid_capstyle="round")
        ax.plot(c[0], y, "o", color=col, markersize=8, markeredgecolor=G.BG, markeredgewidth=2)
        ax.text(c[2] + 0.005, y, f"{c[0]:+.3f}", va="center", fontsize=9, color=G.INK)
    ax.axvline(0, color=G.MUTED, linewidth=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_xlabel(f"closed-loop score, {B} - {A} (paired per scene, 95% CI)")
    ax.grid(axis="x", color=G.GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "paired_diff.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
