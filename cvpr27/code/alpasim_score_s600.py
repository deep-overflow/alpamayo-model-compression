"""AlpaSim score (km driven per event) for every s600 closed-loop arm (cvpr27/reports/2026-11-02_alpasim-score-s600.md).

Alpamayo-R1 (arXiv 2511.00088, Sec. 6.1) defines the AlpaSim score as the average distance driven in km between
events -- offroad or close encounter -- and an at-fault variant that drops rear-end encounters (offroad still counts);
the public benchmark and the Alpamayo 1.5 model card (1.37 +- 0.10 on 913 scenarios) report the at-fault one.
AlpaSim computes it in eval/aggregation/processing.py:get_avg_dist_between_incidents as
sum(dist_traveled_m) / sum(event) / 1000 per rollout index, on the per-rollout rows after the aggregation modifiers
(time steps after the first event, and after a 4 m deviation from the GT path, removed):
  all       offroad_or_collision           = collision_any or offroad
  at-fault  offroad_or_collision_at_fault  = collision_front or collision_lateral or offroad
With one rollout per scene there is no spread over rollout indices, so the CI here is a scene bootstrap of the ratio.
Scenes are the ones every arm finished (aborted rollouts carry no metrics).

  python3 cvpr27/code/alpasim_score_s600.py
"""

import json
from pathlib import Path

import numpy as np

RUNS = Path("/home/cvlab21/project/chan/alpasim-runs")
OUT = Path(__file__).resolve().parents[2] / "outputs" / "cvpr27" / "alpasim_score_s600"
ARMS = [("s600_merged_baseline", "dense"), ("d600_merged_slim_dual_u40_v2", "MoOP"),
        ("l600_merged_slim_dual_u40_l0keep_v2", "MoOP_l0keep"), ("f600_merged_slim_trajfix_u40_l0keep_v2", "FM-only"),
        ("c600_merged_slim_coc_u40_l0keep_v2", "CE-only"), ("m95_merged_slim_dual_u40_l0keep_l14m95_v2", "l14m95"),
        ("m90_merged_slim_dual_u40_l0keep_l14m90_v2", "l14m90"), ("m80_merged_slim_dual_u40_l0keep_l14m80_v2", "l14m80"),
        ("p24_merged_slim_lp_m24", "lp_m24"), ("r50_merged_lp_r50", "lp_r50"), ("tyo_merged_slim_tyr_off", "tyr_off"),
        ("wsp_merged_slim_wandasp_u40", "wanda_sp"), ("t40_merged_slim_tyr_u40_r", "tyr_u40_r")]
B = 20000


def main():
    rows = {}
    for run, label in ARMS:
        f = RUNS / run / "aggregate" / "results-summary.json"
        if not f.exists():
            continue
        d = {}
        for r in json.loads(f.read_text())["rollouts"]:
            m = r.get("metrics") or {}
            if m.get("collision_any") is None:
                continue
            d[r["clipgt_id"]] = (float(m["dist_traveled_m"]), float(m["offroad_or_collision"] > 0),
                                 float(m["offroad_or_collision_at_fault"] > 0), float(r["score"]))
        rows[label] = d
    scenes = sorted(set.intersection(*[set(d) for d in rows.values()]))
    A = {k: np.array([d[s] for s in scenes]) for k, d in rows.items()}   # (n, 4): dist m, event, at-fault event, score
    idx = np.random.default_rng(0).integers(0, len(scenes), (B, len(scenes)))

    def ratio(a, col, i=None):
        a = a if i is None else a[i]
        return a[..., 0].sum(-1) / a[..., col].sum(-1) / 1000

    res = {"n_scenes": len(scenes), "arms": {}}
    lines = [f"AlpaSim score (km per event) over the {len(scenes)} scenes every arm finished; 95% CI = scene bootstrap"]
    for k, a in A.items():
        e = {"km_driven": a[:, 0].sum() / 1000, "scene_score_mean": a[:, 3].mean()}
        for col, name in ((1, "all"), (2, "at_fault")):
            e[name] = {"score": ratio(a, col), "ci": np.percentile(ratio(a, col, idx), [2.5, 97.5]).tolist(),
                       "events": int(a[:, col].sum())}
            for base in ("dense", "MoOP"):
                if base in A and base != k:
                    d = ratio(a, col, idx) - ratio(A[base], col, idx)
                    e[name][f"minus_{base}"] = {"delta": ratio(a, col) - ratio(A[base], col),
                                                "ci": np.percentile(d, [2.5, 97.5]).tolist()}
        res["arms"][k] = e
        lines.append(f"{k:12s} all {e['all']['score']:.3f} [{e['all']['ci'][0]:.3f}, {e['all']['ci'][1]:.3f}] "
                     f"({e['all']['events']} ev) | at-fault {e['at_fault']['score']:.3f} "
                     f"[{e['at_fault']['ci'][0]:.3f}, {e['at_fault']['ci'][1]:.3f}] ({e['at_fault']['events']} ev) | "
                     f"{e['km_driven']:.1f} km | scene score {e['scene_score_mean']:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=1))
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
