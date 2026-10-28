"""Dense Alpamayo 1.5 closed loop on the stratified 600-scene public_2601 sample, one rollout per scene.

Plan: plans/2026-10-28_dense-closedloop-full2601.md (§5). Run: alpasim-runs/s600_merged_baseline (the
584-scene main run merged with the 16-scene smoke run, same settings), scene list and strata from
make_strat_suite.py. Patched AlpaSim (hand-over fix), eight cards: shard i ran on GPU i, Blackwell 0-3,
Ada 4-7.

  * suite estimate: stratified mean over the 9 cells (weight = cell size / 912 frame), CI from a bootstrap
    that resamples scenes within each cell; plain mean alongside
  * scoring: shipped score re-derived from metrics.parquet (gate G0, |diff| < 1e-6 on every scored rollout)
    so the end-of-route-aware 4 m variant (`score_dev` in analyze_closedloop_gap) can be reported
  * aborted rollouts (failure_reason other than collision / offroad, forced to 0.0): counted, and the
    estimate repeated without their scenes
  * subsets: Hard100 and Origin150 inside the sample
  * stability: Origin150 scenes, the new rollout vs the mean of hofix_merged_baseline's two (same harness)
  * rollout noise: the same scenes with three rollouts -> within- / between-scene variance, ICC, and the
    standard error of a suite mean for other scene and rollout counts
  * card architecture: Blackwell minus Ada, post-stratified over the 9 cells (all scenes, without the aborted
    ones, and on new - sangoh), and within scene on Origin150: new rollout - hofix's two Ada rollouts, by card
  * reference only: paired against sangoh's 913 x 1 run (before the hand-over fix, its settings unknown)

    .venv/bin/python experiments/evaluation/analyze_dense_s600.py
Writes outputs/alpasim_dense_s600/{config.json, metrics.json, summary.txt, rollouts.parquet, *.png}.
"""

import json
import re
from pathlib import Path

import analyze_closedloop_gap as G
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "alpasim_dense_s600"
RUNS = Path("/home/cvlab21/project/chan/alpasim-runs")
RUN = RUNS / "s600_merged_baseline"
HOFIX = RUNS / "hofix_merged_baseline"
SANGOH = G.SANGOH
FRAME = 912
NB = 20000
METRICS = ["collision_at_fault", "collision_any", "offroad", "progress_clipped_rel", "dist_traveled_m",
           "plan_deviation", "dist_to_gt_trajectory"]
GATE_FAIL = {"None", "collision_at_fault", "offroad"}


def per_rollout(run):
    """results-summary rollouts -> one row each: shipped score, gate metrics, failure reason."""
    rows = []
    for r in json.loads((run / "aggregate" / "results-summary.json").read_text())["rollouts"]:
        m = r.get("metrics") or {}
        rows.append({"scene": r["clipgt_id"], "rollout": r["rollout_id"], "run_name": r.get("run_name", ""),
                     "score": np.nan if r.get("score") is None else float(r["score"]),
                     "passed": float(bool(r.get("passed"))), "failure_reason": str(r.get("failure_reason")),
                     **{k: np.nan if m.get(k) is None else float(m[k]) for k in METRICS}})
    return pd.DataFrame(rows)


def strat_boot(df, col, rng):
    """Stratified mean of a per-scene column and its within-cell bootstrap CI.

    Weights are renormalised over the cells present, so a subset that empties a cell (scenes dropped)
    still estimates a mean rather than a fraction of one."""
    d = df.dropna(subset=[col])
    cells = [(g[col].to_numpy(float), g["W"].iloc[0]) for _, g in d.groupby("cell")]
    wsum = sum(w for _, w in cells)
    est = sum(w * x.mean() for x, w in cells) / wsum
    b = np.zeros(NB)
    for x, w in cells:
        b += w * x[rng.integers(0, len(x), size=(NB, len(x)))].mean(1)   # (NB,)
    b /= wsum
    return float(est), float(np.quantile(b, .025)), float(np.quantile(b, .975))


def boot(x, rng):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    m = x[rng.integers(0, len(x), size=(NB, len(x)))].mean(1)              # (NB,)
    return float(x.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))


def fmt(c, d=3, sign=False):
    f = f"{{:{'+' if sign else ''}.{d}f}}"
    return f"{f.format(c[0])} [{f.format(c[1])}, {f.format(c[2])}]"


def main():
    rng = np.random.default_rng(0)
    sc = pd.read_csv(OUT / "scenes_s600.csv")
    sc["cell"] = sc.sg_level.astype(str) + "/" + sc.hs_tertile
    cells = pd.DataFrame(json.loads((OUT / "draw.json").read_text())["cells"])
    cells["cell"] = cells.sg_level.astype(str) + "/" + cells.hs_tertile
    sc = sc.merge(cells[["cell", "N"]], on="cell")
    sc["W"] = sc["N"] / FRAME
    shard = pd.concat([pd.read_csv(RUNS / f"shard_suites_{p}_baseline.csv") for p in ("s600smoke", "s600")])
    sc["gpu"] = sc.scene_id.map(dict(zip(shard.scene_id, shard.test_suite_id.str.extract(r"_sh(\d)$")[0].astype(int))))
    sc["arch"] = np.where(sc.gpu < 4, "Blackwell", "Ada")
    # the card each scene ran on: a later arm paired against this baseline should run each scene on the same kind
    sc[["scene_id", "uuid", "gpu", "arch"]].to_csv(OUT / "scene_gpu.csv", index=False)

    ro = per_rollout(RUN)
    # a smoke scene whose rollout aborted was run again by the main stage; keep the main-stage rollout
    ro["main"] = ro.run_name.str.contains("s600_baseline")
    ro = ro.sort_values("main").groupby("scene").tail(1)
    uuid = pd.read_csv(G.SCENES).set_index("scene_id")["uuid"]
    rec, g0 = [], []
    for r in ro.itertuples():
        p = RUN / "rollouts" / r.scene / r.rollout / "metrics.parquet"
        a = G.anatomy(p, G.gt_track(uuid[r.scene])) if p.exists() else None
        if a is not None:
            a = a[0]
            g0.append(abs(a["score"] - r.score) if not np.isnan(r.score) else np.nan)
        rec.append({"score_re": np.nan if a is None else a["score"],
                    "score_dev": np.nan if a is None else a["score_dev"]})
    ro = pd.concat([ro.reset_index(drop=True), pd.DataFrame(rec)], axis=1)
    ro.to_parquet(OUT / "rollouts.parquet")
    d = sc.merge(ro, left_on="scene_id", right_on="scene", how="left")
    g0_max = float(np.nanmax(g0)) if g0 else np.nan

    aborted = d[~d.failure_reason.isin(GATE_FAIL) & d.score.notna()]
    missing = d[d.score.isna()]
    # the runtime retries a failed rollout up to twice (runtime.max_rollout_retries); only the launch logs
    # say which scenes needed it, the summary keeps the final attempt
    retried = {}
    for log in sorted(RUNS.glob("s600*_baseline_sh*.launch.log")):
        for m in re.finditer(r"Retrying failed rollout: scene=(\S+) .*?retry=(\d)/\d error=([^:\n]+)",
                             log.read_text(errors="replace")):
            retried.setdefault(m[1], []).append(f"{m[2]}: {m[3]}")
    res = {"n_scenes": len(d), "n_scored": int(d.score.notna().sum()), "missing": missing.scene_id.tolist(),
           "aborted": aborted[["scene_id", "failure_reason"]].values.tolist(), "retried": retried,
           "G0_max_abs_diff": g0_max, "G0_pass": bool(g0_max < 1e-6)}

    est = {}
    for col in ["score", "score_dev", "passed", "progress_clipped_rel", "collision_any", "collision_at_fault",
                "offroad", "dist_traveled_m", "plan_deviation"]:
        est[col] = {"stratified": strat_boot(d, col, rng), "plain": boot(d[col], rng)}
    est["score_no_aborted"] = {"stratified": strat_boot(d[~d.scene_id.isin(aborted.scene_id)], "score", rng)}
    res["estimates"] = est

    sub = {}
    for name, m in (("hard100", d.hard100), ("origin150", d.origin150), ("rest", ~d.hard100 & ~d.origin150)):
        sub[name] = {"n": int(m.sum()), "score": boot(d.score[m], rng), "progress": boot(d.progress_clipped_rel[m], rng)}
    res["subsets"] = sub
    res["cells"] = {c: {"n": len(g), "N": int(g.N.iloc[0]), "score": boot(g.score, rng)}
                    for c, g in d.groupby("cell")}

    # stability and rollout noise on the Origin150 scenes: two hofix rollouts + the new one
    hf = per_rollout(HOFIX)
    o = d[d.origin150][["scene_id", "score"]].rename(columns={"score": "new"})
    hm = hf.groupby("scene").score.agg(["mean", "count"]).rename(columns={"mean": "old"})
    o = o.join(hm, on="scene_id").dropna()
    diff = (o.new - o.old).to_numpy()
    res["stability"] = {"n": len(o), "new_minus_old": boot(diff, rng),
                        "wilcoxon_p": float(wilcoxon(diff).pvalue) if np.any(diff) else 1.0}
    three = pd.concat([hf[hf.scene.isin(o.scene_id)][["scene", "score"]],
                       d[d.origin150][["scene_id", "score"]].rename(columns={"scene_id": "scene"})])
    piv = three.groupby("scene").score.apply(list)
    piv = np.array([x for x in piv if len(x) == 3])                            # (S, 3)
    within = float(piv.var(1, ddof=1).mean())
    between = float(piv.mean(1).var(ddof=1) - within / 3)
    res["noise"] = {"scenes": len(piv), "within_var": within, "between_var": between,
                    "icc": between / (between + within),
                    "se_table": {f"{s}x{r}": float(np.sqrt(between / s + within / (s * r)))
                                 for s in (300, 600, 912) for r in (1, 2, 3)}}

    sg = per_rollout(SANGOH).groupby("scene").score.mean()
    d["sangoh_run"] = d.scene_id.map(sg)

    def arch_gap(df, col):
        """Blackwell - Ada of a per-scene column, post-stratified over the cells (weight = cell size)."""
        est, b, wsum = 0.0, np.zeros(NB), 0.0
        for _, g in df.dropna(subset=[col]).groupby("cell"):
            bw, ad = g[col][g.arch == "Blackwell"].to_numpy(), g[col][g.arch == "Ada"].to_numpy()
            if len(bw) and len(ad):
                w = g.W.iloc[0]
                wsum += w
                est += w * (bw.mean() - ad.mean())
                b += w * (bw[rng.integers(0, len(bw), (NB, len(bw)))].mean(1)
                          - ad[rng.integers(0, len(ad), (NB, len(ad)))].mean(1))
        return est / wsum, float(np.quantile(b / wsum, .025)), float(np.quantile(b / wsum, .975))

    # card architecture. The between-scene gap is confounded by which scenes each card drew and by the
    # aborts (scene properties) landing where they did; the cleaner test is within scene: Origin150 scenes
    # were also run twice on Ada by hofix, so (new - hofix) per scene splits by the card the new one ran on
    d["did"] = d.score - d.sangoh_run
    clean = d[~d.scene_id.isin(aborted.scene_id)]
    w = d[d.origin150].assign(old=lambda x: x.scene_id.map(hf.groupby("scene").score.mean()))
    w["diff"] = w.score - w.old
    wb, wa = w["diff"][w.arch == "Blackwell"].to_numpy(), w["diff"][w.arch == "Ada"].to_numpy()
    wbb = (wb[rng.integers(0, len(wb), (NB, len(wb)))].mean(1) - wa[rng.integers(0, len(wa), (NB, len(wa)))].mean(1))
    res["arch"] = {"n_blackwell": int((d.arch == "Blackwell").sum()), "n_ada": int((d.arch == "Ada").sum()),
                   "aborted_on_blackwell": int((aborted.scene_id.map(d.set_index("scene_id").arch) == "Blackwell").sum()),
                   "score_gap": arch_gap(d, "score"), "score_gap_no_aborted": arch_gap(clean, "score"),
                   "gap_in_new_minus_sangoh": arch_gap(d, "did"),
                   "gap_in_new_minus_sangoh_no_aborted": arch_gap(clean, "did"),
                   "per_gpu_score": d.groupby("gpu").score.mean().round(4).to_dict(),
                   "within_scene": {"n_blackwell": len(wb), "n_ada": len(wa),
                                    "blackwell_new_minus_hofix": boot(wb, rng), "ada_new_minus_hofix": boot(wa, rng),
                                    "difference": (float(wb.mean() - wa.mean()), float(np.quantile(wbb, .025)),
                                                   float(np.quantile(wbb, .975)))}}

    pr = d[["scene_id", "score"]].join(sg.rename("sangoh"), on="scene_id").dropna()
    res["vs_sangoh"] = {"n": len(pr), "new": float(pr.score.mean()), "sangoh": float(pr.sangoh.mean()),
                        "new_minus_sangoh": boot(pr.score - pr.sangoh, rng),
                        "wilcoxon_p": float(wilcoxon(pr.score - pr.sangoh).pvalue)}
    res["hofix_origin150_score"] = float(hf.score.mean())

    cfg = {"run": str(RUN), "scenes": str(OUT / "scenes_s600.csv"), "frame": FRAME, "rollouts_per_scene": 1,
           "gpus": "0-3 RTX PRO 5000 Blackwell, 4-7 RTX 5880 Ada, one shard per card", "driver_omp_threads": 8,
           "alpasim": "patched (hand-over fix, minitron/alpasim/handover_fix.patch)", "topology": "pair_a15",
           "bootstrap": NB, "plan": "plans/2026-10-28_dense-closedloop-full2601.md"}
    (OUT / "config.json").write_text(json.dumps(cfg, indent=1))
    (OUT / "metrics.json").write_text(json.dumps(res, indent=1, default=float))

    e = est
    L = [(f"dense Alpamayo 1.5, public_2601 stratified sample: {res['n_scored']}/{res['n_scenes']} scenes scored, "
         f"1 rollout each; G0 max |recomputed - shipped| {g0_max:.1e} -> {'PASS' if res['G0_pass'] else 'FAIL'}"),
         f"aborted (forced 0.0): {len(aborted)} {aborted.scene_id.tolist()}; missing: {len(missing)}",
         f"retried by the runtime: {len(retried)} scenes "
         + "; ".join(f"{s[7:15]} [{', '.join(v)}]" for s, v in retried.items()),
         "", "suite estimate (stratified [95% CI]) | plain mean",
         *[f"  {k:22s} {fmt(v['stratified'])} | {fmt(v['plain'])}" for k, v in e.items() if "plain" in v],
         f"  {'score w/o aborted':22s} {fmt(e['score_no_aborted']['stratified'])}",
         "", "subsets (mean [95% CI])",
         *[f"  {k:10s} n={v['n']:3d}  score {fmt(v['score'])}  progress {fmt(v['progress'])}" for k, v in sub.items()],
         "", "cells sangoh-level/hard-tertile: n/N  score",
         *[f"  {c:9s} {v['n']:3d}/{v['N']:3d}  {fmt(v['score'])}" for c, v in res["cells"].items()],
         "", (f"stability (Origin150 scenes, n={len(o)}): new - mean(hofix x2) {fmt(res['stability']['new_minus_old'], sign=True)}"
             f", Wilcoxon p={res['stability']['wilcoxon_p']:.3f}"),
         f"rollout noise ({len(piv)} scenes x 3): within {within:.4f}, between {between:.4f}, ICC {res['noise']['icc']:.2f}",
         "  SE of a suite mean, scenes x rollouts: "
         + "  ".join(f"{k} {v:.4f}" for k, v in res["noise"]["se_table"].items()),
         (f"card architecture: Blackwell n={res['arch']['n_blackwell']}, Ada n={res['arch']['n_ada']}, "
          f"aborted on Blackwell {res['arch']['aborted_on_blackwell']}/{len(aborted)}; Blackwell - Ada post-stratified:"),
         f"  score {fmt(res['arch']['score_gap'], sign=True)}, w/o aborted {fmt(res['arch']['score_gap_no_aborted'], sign=True)}",
         (f"  in (new - sangoh) {fmt(res['arch']['gap_in_new_minus_sangoh'], sign=True)}, "
          f"w/o aborted {fmt(res['arch']['gap_in_new_minus_sangoh_no_aborted'], sign=True)}"),
         "  per GPU score " + " ".join(f"{k}:{v:.3f}" for k, v in res["arch"]["per_gpu_score"].items()),
         (f"  within scene (Origin150, new - hofix Ada x2): Blackwell n={len(wb)} "
          f"{fmt(res['arch']['within_scene']['blackwell_new_minus_hofix'], sign=True)}, Ada n={len(wa)} "
          f"{fmt(res['arch']['within_scene']['ada_new_minus_hofix'], sign=True)}, difference "
          f"{fmt(res['arch']['within_scene']['difference'], sign=True)}"),
         (f"reference, sangoh 913x1 (before the fix) on the same {len(pr)} scenes: {res['vs_sangoh']['sangoh']:.3f} -> "
         f"{res['vs_sangoh']['new']:.3f}, diff {fmt(res['vs_sangoh']['new_minus_sangoh'], sign=True)}, "
         f"p={res['vs_sangoh']['wilcoxon_p']:.2g}"),
         f"reference, hofix Origin150 x2: {res['hofix_origin150_score']:.3f}"]
    (OUT / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    rows = [("suite (stratified)", e["score"]["stratified"], G.CAT[0]),
            ("  4 m rule ends with the route", e["score_dev"]["stratified"], G.CAT[0]),
            ("Origin150 scenes in sample", sub["origin150"]["score"], G.MUTED),
            ("Hard100 scenes in sample", sub["hard100"]["score"], G.MUTED),
            ("other scenes", sub["rest"]["score"], G.MUTED)]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for i, (lab, c, col) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([c[1], c[2]], [y, y], color=col, linewidth=2, solid_capstyle="round")
        ax.plot(c[0], y, "o", color=col, markersize=8, markeredgecolor=G.BG, markeredgewidth=2)
        ax.text(c[2] + 0.01, y, f"{c[0]:.3f}", va="center", fontsize=9, color=G.INK)
    ax.axvline(res["hofix_origin150_score"], color=G.MUTED, linestyle=":", linewidth=1)
    ax.text(res["hofix_origin150_score"], len(rows) - 0.45, "Origin150 x2 (10-26)", fontsize=8, color=G.MUTED,
            ha="center")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_xlabel("closed-loop score (mean, 95% CI)")
    ax.grid(axis="x", color=G.GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "score_estimates.png", dpi=160)
    plt.close(fig)

    order = [c for c in (f"{a}/{b}" for a in ("0", "mid", "1") for b in ("lo", "mid", "hi")) if c in res["cells"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for i, c in enumerate(order):
        v = res["cells"][c]["score"]
        ax.plot([i, i], [v[1], v[2]], color=G.CAT[0], linewidth=2, solid_capstyle="round")
        ax.plot(i, v[0], "o", color=G.CAT[0], markersize=8, markeredgecolor=G.BG, markeredgewidth=2)
    ax.set_xticks(range(len(order)), [f"{c}\nn={res['cells'][c]['n']}" for c in order], fontsize=8)
    ax.set_xlabel("stratum: sangoh 913x1 score level / hard_score tertile")
    ax.set_ylabel("score (mean, 95% CI)")
    ax.grid(axis="y", color=G.GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "score_by_stratum.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
