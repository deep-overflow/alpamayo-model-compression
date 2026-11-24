"""AlpaSim score (km driven per event, Alpamayo-R1 Sec. 6.1) for the ICLR-paper arms at u40 and u50, Origin150 and Hard100.

Definition, as AlpaSim computes it (eval/aggregation/processing.py:get_avg_dist_between_incidents): within each scene the
rollouts are ranked by "run_uuid_rollout_id" (rollout index 1, 2); per rollout index, sum(dist_traveled_m) / sum(events)
/ 1000 over all scenes; reported as mean +- std (ddof 1) over the rollout indices. Per-rollout values are the aggregated
ones in results-summary.json (after AlpaSim's modifiers: time steps after the first event and after a 4 m deviation from
the GT path removed), so each rollout carries at most one event.
  all       offroad_or_collision           = collision_any (incl. rear) or offroad
  at-fault  offroad_or_collision_at_fault  = collision_front or collision_lateral or offroad   (the public / model-card one)
Aborted rollouts (route sanity check) have no metrics and drop out, as in AlpaSim. Checked against AlpaSim's own
metrics_results.txt on u50h_slim_dual_u50_v2_sh1 (1.47 +- 0.35 / 1.93 +- 0.30, equal).
95% CI: scene-cluster bootstrap of the same statistic; paired differences vs MoOP of the same budget use the same
resample (MoOP - arm, positive = MoOP drives further per event).
Runs: u40 = the ICLR runs (m2601_merged_* / h100_merged_*, pre-hand-over-fix harness); Origin150 LLM-Pruner u40 = the
09-20 cache outputs/alpasim_all_metrics/rollout_metrics.parquet (run dir deleted; no rollout_id, so rollout index = cache
order within the scene). u50 = u50o_merged_* / u50h_merged_* (patched harness); dense = the same ICLR run in both budgets.

  python iclr27-rebuttal/code/alpasim_score_u40_u50.py [--seed 0] [--draws 20000]
writes outputs/iclr27-rebuttal/alpasim_score_u40_u50/{metrics_seed<k>.json, table_seed<k>.md}
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUNS = Path("/home/cvlab21/project/chan/advla/alpasim-runs")
OUT = REPO / "outputs" / "iclr27-rebuttal" / "alpasim_score_u40_u50"
LP150 = REPO / "outputs" / "alpasim_all_metrics" / "rollout_metrics.parquet"
SETS = {"origin150": ("Origin150", {"u40": "m2601_merged_", "u50": "u50o_merged_"}),
        "hard100": ("Hard100", {"u40": "h100_merged_", "u50": "u50h_merged_"})}
ARMS = {"u40": [("Baseline (dense)", "baseline"), ("MoOP", "slim_dual_u40_v2"), ("MoOP+E", "slim_dualexp_u40_em93p75"),
                ("FM-only", "slim_traj_u40_v2"), ("CE-only", "slim_coc_u40_v2"), ("Tyr", "slim_tyr_u40_r"),
                ("LLM-Pruner", "lp_r50"), ("Wanda", "slim_wanda_u40_v2")],
        "u50": [("Baseline (dense)", None), ("MoOP", "slim_dual_u50_v2"), ("FM-only", "slim_traj_u50_v2"),
                ("CE-only", "slim_coc_u50_v2"), ("Tyr", "slim_tyr_u50_r"), ("LLM-Pruner", "slim_lp_r60"),
                ("Wanda", "slim_wanda_u50_v2")]}
REF = "MoOP"


def load(set_, budget, cfg):
    """-> DataFrame (scene, idx, d [m], e_all, e_af) of scored rollouts, and the number of aborted ones."""
    if cfg is None:                                                    # dense: the ICLR run, same in both budgets
        return load(set_, "u40", "baseline")
    if set_ == "origin150" and cfg == "lp_r50":
        c = pd.read_parquet(LP150)
        c = c[(c.set == "origin150") & (c.arm == "llm-pruner")].copy()
        df = pd.DataFrame({"scene": c.scene.values, "d": c.dist_traveled_m.values.astype(float),
                           "e_all": (c.offroad_or_collision.values > 0).astype(float),
                           "e_af": (c.offroad_or_collision_at_fault.values > 0).astype(float)})
        df["idx"] = df.groupby("scene").cumcount() + 1
        return df, 0
    rs = json.loads((RUNS / f"{SETS[set_][1][budget]}{cfg}" / "aggregate" / "results-summary.json").read_text())["rollouts"]
    rows, aborted = [], 0
    for r in rs:
        m = r.get("metrics") or {}
        if m.get("dist_traveled_m") is None:
            aborted += 1
            continue
        rows.append({"scene": r["clipgt_id"], "key": f"{r.get('run_uuid')}_{r['rollout_id']}", "d": float(m["dist_traveled_m"]),
                     "e_all": float(m["offroad_or_collision"] > 0), "e_af": float(m["offroad_or_collision_at_fault"] > 0)})
    df = pd.DataFrame(rows)
    df["idx"] = df.groupby("scene")["key"].rank(method="dense").astype(int)
    return df.drop(columns="key"), aborted


def to_array(df, scenes):
    """-> (S, 2, 3) per scene and rollout index: dist m, all events, at-fault events (0 where a rollout is missing)."""
    a = np.zeros((len(scenes), 2, 3))
    pos = {s: i for i, s in enumerate(scenes)}
    for r in df.itertuples():
        a[pos[r.scene], r.idx - 1] += (r.d, r.e_all, r.e_af)
    return a


def score(a):
    """(..., S, 2, 3) -> (..., 2) [all, at-fault]: mean over rollout indices of km per event."""
    s = a.sum(-3)                                                      # (..., 2 idx, 3)
    with np.errstate(divide="ignore"):
        r = s[..., 0:1] / s[..., 1:3] / 1000                           # (..., 2 idx, 2 variants)
    return r.mean(-2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=20_000)
    args = ap.parse_args()
    res = {"seed": args.seed, "draws": args.draws, "sets": {}}
    lines = [(f"AlpaSim score = km driven per event (R1 Sec. 6.1), mean +- std over the 2 rollout indices; "
              f"95% CI = scene-cluster bootstrap ({args.draws:,} draws, seed {args.seed})")]
    f2 = lambda c: f"{c[0]:.2f} [{c[1]:.2f}, {c[2]:.2f}]"
    fd = lambda c: f"{c[0]:+.2f} [{c[1]:+.2f}, {c[2]:+.2f}]" + ("*" if c[1] > 0 or c[2] < 0 else "")
    for set_, (name, _) in SETS.items():
        for budget, arms in ARMS.items():
            data = {lab: load(set_, budget, cfg) for lab, cfg in arms}
            scenes = sorted(set().union(*[set(d.scene) for d, _ in data.values()]))
            A = {lab: to_array(d, scenes) for lab, (d, _) in data.items()}
            rng = np.random.default_rng(args.seed)
            idx = rng.integers(0, len(scenes), (args.draws, len(scenes)))   # (B, S), shared by every arm
            boot = {lab: score(a[idx]) for lab, a in A.items()}            # (B, 2)
            R = res["sets"].setdefault(set_, {})[budget] = {"n_scenes": len(scenes), "arms": {}, "paired_vs_moop": {}}
            lines += ["", f"## {name}, {budget} ({len(scenes)} scenes x 2 rollouts)", "",
                      ("| Method | scored / aborted | km driven | events all / at-fault | AlpaSim score (all) | "
                       "AlpaSim score (at-fault) | at-fault 95% CI | MoOP - arm, at-fault | MoOP - arm, all |"),
                      "|---|---|---|---|---|---|---|---|---|"]
            for lab, (d, ab) in data.items():
                s = A[lab].sum(0)                                         # (2 idx, 3)
                per = s[:, 0:1] / s[:, 1:3] / 1000                        # (2 idx, 2)
                e = {"scored": len(d), "aborted": int(ab), "km": float(s[:, 0].sum() / 1000),
                     "events_all": int(s[:, 1].sum()), "events_at_fault": int(s[:, 2].sum())}
                for j, v in enumerate(["all", "at_fault"]):
                    ci = np.percentile(boot[lab][:, j], [2.5, 97.5])
                    e[v] = {"mean": float(per[:, j].mean()), "std": float(per[:, j].std(ddof=1)),
                            "per_index": per[:, j].tolist(), "ci": [float(per[:, j].mean()), float(ci[0]), float(ci[1])]}
                R["arms"][lab] = e
            for lab, e in R["arms"].items():
                if lab != REF:
                    R["paired_vs_moop"][lab] = {
                        v: [R["arms"][REF][v]["mean"] - e[v]["mean"],
                            *np.percentile(boot[REF][:, j] - boot[lab][:, j], [2.5, 97.5]).tolist()]
                        for j, v in enumerate(["all", "at_fault"])}
                p = R["paired_vs_moop"].get(lab)
                lines.append(f"| {lab} | {e['scored']} / {e['aborted']} | {e['km']:.1f} | {e['events_all']} / "
                             f"{e['events_at_fault']} | {e['all']['mean']:.2f} ± {e['all']['std']:.2f} | "
                             f"{e['at_fault']['mean']:.2f} ± {e['at_fault']['std']:.2f} | {f2(e['at_fault']['ci'])} | "
                             + (f"{fd(p['at_fault'])} | {fd(p['all'])} |" if p else "— | — |"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"metrics_seed{args.seed}.json").write_text(json.dumps(res, indent=1))
    (OUT / f"table_seed{args.seed}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
