"""Four rollouts per scene on the 76 picked s600 scenes (cvpr27/plans/2026-11-05_rollout4-76scenes.md).

The scenes were picked on one s600 rollout per arm: MoOP l0keep had no at-fault collision (front/lateral) and no
offroad, tyr_off or lp_m24 had one. Picking on a single noisy rollout favours the reference, so the s600 rollouts are
not reused here: only the four fresh rollouts per scene per arm count. Per rollout, event = at-fault collision or
offroad (rear-end ignored, user rule 2026-11-04).
Writes outputs/cvpr27/rollout4_76/{per_scene.csv, metrics.json, summary.txt, event_rate.png}.

  python3 cvpr27/code/analyze_rollout4_76.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUNS = Path("/home/cvlab21/project/chan/alpasim-runs")
OUT = REPO / "outputs" / "cvpr27" / "rollout4_76"
ARMS = [("r4l0_merged_slim_dual_u40_l0keep_v2", "MoOP_l0keep"), ("r4ty_merged_slim_tyr_off", "tyr_off"),
        ("r4lp_merged_slim_lp_m24", "lp_m24"), ("r4dn_merged_baseline", "dense")]
REF, B = "MoOP_l0keep", 20000


def load(run):
    rows = []
    for r in json.loads((RUNS / run / "aggregate" / "results-summary.json").read_text())["rollouts"]:
        m = r.get("metrics") or {}
        ok = m.get("collision_any") is not None
        hit = {k: ok and float(m.get(k) or 0) > 0
               for k in ("collision_at_fault", "offroad", "collision_rear", "wrong_lane")}
        num = {k: float(m.get(k) or 0) if ok else np.nan
               for k in ("progress_clipped_rel", "open_loop_collision", "dist_traveled_m")}
        rows.append({"scene": r["clipgt_id"], "rollout": r["rollout_id"], "aborted": not ok, "score": r["score"],
                     "at_fault": hit["collision_at_fault"], "offroad": hit["offroad"], "rear": hit["collision_rear"],
                     "wrong_lane": hit["wrong_lane"], "progress": num["progress_clipped_rel"],
                     "olc": num["open_loop_collision"], "dist": num["dist_traveled_m"]})
    d = pd.DataFrame(rows)
    d["event"] = d.at_fault | d.offroad
    coc = OUT / f"coc_{run}.csv"   # written by coc_rollout4_76.py (alpasim venv), NaN until then
    d = d.merge(pd.read_csv(coc), on=["scene", "rollout"], how="left") if coc.exists() else d.assign(coc_degenerate=np.nan)
    return d


def main():
    sel = pd.read_csv(REPO / "outputs/cvpr27/moop_clean_scenes/selected76_tyroff_lpm24.csv", keep_default_na=False)
    group = dict(zip(sel.scene, sel.which))
    D = {lab: load(run) for run, lab in ARMS if (RUNS / run / "aggregate" / "results-summary.json").exists()}
    per = []
    for lab, d in D.items():
        v = d[~d.aborted]
        g = v.groupby("scene").agg(n=("event", "size"), events=("event", "sum"), at_fault=("at_fault", "sum"),
                                   offroad=("offroad", "sum"), rear=("rear", "sum"), score=("score", "mean"))
        g["arm"] = lab
        g["aborted"] = d[d.aborted].groupby("scene").size().reindex(g.index).fillna(0).astype(int)
        per.append(g.reset_index())
    per = pd.concat(per)
    per["group"] = per.scene.map(group)
    per.to_csv(OUT / "per_scene.csv", index=False)

    rate = per.assign(r=per.events / per.n).pivot(index="scene", columns="arm", values="r")
    score = per.pivot(index="scene", columns="arm", values="score")
    rng = np.random.default_rng(0)
    res, lines = {"n_scenes": len(rate), "arms": {}}, []
    lines.append("76 picked scenes x 4 rollouts; event = at-fault collision or offroad; 95% CI = scene bootstrap")
    for lab, d in D.items():
        v = d[~d.aborted]
        e = {"rollouts": len(d), "aborted": int(d.aborted.sum()), "event_rate": float(v.event.mean()),
             "at_fault_rate": float(v.at_fault.mean()), "offroad_rate": float(v.offroad.mean()),
             "rear_rate": float(v.rear.mean()), "score": float(d.score.mean()),
             "progress": float(v.progress.mean()), "wrong_lane_rate": float(v.wrong_lane.mean()),
             "open_loop_collision": float(v.olc.mean()), "coc_degenerate": float(v.coc_degenerate.mean()),
             "scenes_with_any_event": int((per[per.arm == lab].events > 0).sum()),
             "scenes_all_clean": int((per[per.arm == lab].events == 0).sum())}
        km = v.groupby("scene").agg(dist=("dist", "sum"), ev=("event", "sum"))   # AlpaSim score (at-fault), km/event
        ii = rng.integers(0, len(km), (B, len(km)))
        e["alpasim_at_fault_km"] = {
            "score": float(km.dist.sum() / km.ev.sum() / 1000), "events": int(km.ev.sum()),
            "ci": np.percentile(km.dist.to_numpy()[ii].sum(1) / km.ev.to_numpy()[ii].sum(1) / 1000, [2.5, 97.5]).tolist()}
        if lab != REF and REF in rate:
            for name, tab in (("event_rate", rate), ("score", score)):
                x = (tab[lab] - tab[REF]).dropna().to_numpy()
                bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
                e[f"{name}_minus_{REF}"] = {"delta": float(x.mean()), "ci": np.percentile(bs, [2.5, 97.5]).tolist()}
            for grp in ("tyr_off", "lp_m24", "both"):
                idx = [s for s in rate.index if group.get(s) == grp]
                x = (rate.loc[idx, lab] - rate.loc[idx, REF]).dropna().to_numpy()
                e[f"event_rate_minus_{REF}_{grp}_scenes"] = float(x.mean()) if len(x) else None
        res["arms"][lab] = e
        a = e["alpasim_at_fault_km"]
        line = (f"{lab:12s} event {e['event_rate']:.3f} (at-fault {e['at_fault_rate']:.3f}, offroad {e['offroad_rate']:.3f}"
                f", rear {e['rear_rate']:.3f}) | score {e['score']:.3f} | progress {e['progress']:.3f}"
                f" | AlpaSim at-fault {a['score']:.3f} km [{a['ci'][0]:.3f}, {a['ci'][1]:.3f}]"
                f" | wrong lane {e['wrong_lane_rate']:.3f} | OLC {e['open_loop_collision']:.3f}"
                f" | CoC degen {e['coc_degenerate']:.3f} | scenes clean in all 4: "
                f"{e['scenes_all_clean']}/{len(rate)} | aborted {e['aborted']}")
        if f"event_rate_minus_{REF}" in e:
            dd, sd = e[f"event_rate_minus_{REF}"], e[f"score_minus_{REF}"]
            line += (f" | event - {REF} {dd['delta']:+.3f} [{dd['ci'][0]:+.3f}, {dd['ci'][1]:+.3f}]"
                     f" | score - {REF} {sd['delta']:+.3f} [{sd['ci'][0]:+.3f}, {sd['ci'][1]:+.3f}]")
        lines.append(line)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=1))
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labs = list(D)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    vals = [res["arms"][k]["event_rate"] for k in labs]
    ax.bar(labs, vals, color="#4a6fa5")
    ax.set_ylabel("rollouts with at-fault collision or offroad")
    ax.set_title("76 picked s600 scenes x 4 rollouts")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "event_rate.png", dpi=150)


if __name__ == "__main__":
    main()
