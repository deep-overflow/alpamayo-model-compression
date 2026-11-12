"""u50 closed loop on Origin150 x 2 (iclr27-rebuttal/plans/2026-11-09_u50-closedloop-origin150.md): 95% CIs per arm and
paired 95% CIs against MoOP u50 for scene score, at-fault collision rate and offroad rate.

Same conventions as cvpr27/code/closedloop_ci.py (paper Tables 2 / 8): a rollout's offroad / at-fault is 1 if the event
happened; an aborted rollout scores 0 with no event (none occurred here); per-scene means, scene-cluster bootstrap
(the two rollouts of a scene are correlated), one resample shared by every arm and pair. Paired sign as in Table 8:
score = MoOP u50 - method, rates = method - MoOP u50 (positive favours MoOP). Arms whose merged run is missing
(e.g. CE-only still running) are skipped, so a rerun after they finish gives the full table.

  .venv/bin/python iclr27-rebuttal/code/u50_closedloop_ci.py [--seed 0] [--draws 200000]
writes outputs/iclr27-rebuttal/u50_origin150/ci/{metrics_seed<k>.json, table_seed<k>.md}
"""

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RUNS = Path("/home/cvlab21/project/chan/alpasim-runs")
OUT = REPO / "outputs" / "iclr27-rebuttal" / "u50_origin150" / "ci"
REF = "MoOP u50"
ARMS = [("Baseline (dense)", "hofix_merged_baseline"), ("MoOP u50", "u50o_merged_slim_dual_u50_v2"),
        ("MoOP u50 l0keep", "u50o_merged_slim_dual_u50_l0keep_v2"), ("Tyr u50", "u50o_merged_slim_tyr_u50_r"),
        ("LLM-Pruner u50", "u50o_merged_slim_lp_r60"), ("FM-only u50", "u50o_merged_slim_traj_u50_v2"),
        ("CE-only u50", "u50o_merged_slim_coc_u50_v2"), ("Wanda u50", "u50o_merged_slim_wanda_u50_v2")]
METRICS = ["score", "at_fault", "offroad"]


def per_scene(run):
    """results-summary -> {metric: {scene: mean over that scene's rollouts}}, plus rollout / abort counts."""
    rs = json.loads((RUNS / run / "aggregate" / "results-summary.json").read_text())["rollouts"]
    acc = {m: {} for m in METRICS}
    for r in rs:
        mt = r.get("metrics") or {}
        v = {"score": r["score"] or 0.0, "at_fault": float((mt.get("collision_at_fault") or 0) > 0),
             "offroad": float((mt.get("offroad") or 0) > 0)}
        for m in METRICS:
            acc[m].setdefault(r["clipgt_id"], []).append(v[m])
    return {m: {s: float(np.mean(x)) for s, x in acc[m].items()} for m in METRICS}, len(rs), \
        sum(r["score"] is None for r in rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=200_000)
    args = ap.parse_args()
    arms = [(lab, run) for lab, run in ARMS if (RUNS / run / "aggregate" / "results-summary.json").exists()]
    data = {lab: per_scene(run) for lab, run in arms}
    scenes = sorted(data[REF][0]["score"])
    for lab, (d, n, _) in data.items():
        assert sorted(d["score"]) == scenes and n == 2 * len(scenes), (lab, n)
    V = {(lab, m): np.array([data[lab][0][m][s] for s in scenes]) for lab, _ in arms for m in METRICS}  # (S,)
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(scenes), (args.draws, len(scenes)))  # (B, S), shared by all arms and pairs

    def ci(x):
        b = x[idx].mean(1)  # (B,)
        lo, hi = np.percentile(b, [2.5, 97.5])
        return [float(x.mean()), float(lo), float(hi)]

    res = {"seed": args.seed, "draws": args.draws, "n_scenes": len(scenes), "abs": {}, "paired_vs_moop_u50": {}}
    for lab, _ in arms:
        res["abs"][lab] = {"rollouts": data[lab][1], "aborted": data[lab][2], **{m: ci(V[lab, m]) for m in METRICS}}
        if lab != REF:
            res["paired_vs_moop_u50"][lab] = {
                m: ci(V[REF, m] - V[lab, m]) if m == "score" else ci(V[lab, m] - V[REF, m]) for m in METRICS}

    f = lambda c, sig=False: (f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]" + ("*" if c[1] > 0 or c[2] < 0 else "")
                              if sig else f"{c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}]")
    lines = [f"Origin150, {len(scenes)} scenes x 2 rollouts; scene-cluster bootstrap, {args.draws:,} draws, seed {args.seed}",
             "", "Absolute (95% CI)", "",
             "| Method | rollouts | Scene score | At-fault coll. | Offroad |", "|---|---|---|---|---|"]
    for lab, _ in arms:
        a = res["abs"][lab]
        lines.append(f"| {lab} | {a['rollouts']} | " + " | ".join(f(a[m]) for m in METRICS) + " |")
    lines += ["", f"Paired vs {REF} (score = MoOP - method; rates = method - MoOP; * = CI excludes 0)", "",
              "| Method | Scene score | At-fault coll. | Offroad |", "|---|---|---|---|"]
    for lab, p in res["paired_vs_moop_u50"].items():
        lines.append(f"| {lab} | " + " | ".join(f(p[m], sig=True) for m in METRICS) + " |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"metrics_seed{args.seed}.json").write_text(json.dumps(res, indent=1))
    (OUT / f"table_seed{args.seed}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
