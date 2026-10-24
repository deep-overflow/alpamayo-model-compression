"""Closed-loop per-scene failure analysis: m2601 150 scenes and hard100, arms baseline / traj / coc / dual.
Per rollout: score + metrics; per scene: mean over rollouts; paired deltas vs baseline; worst-decile
overlap; failure-mode decomposition (gates, progress, deviation); scene kinematic features."""
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon, spearmanr, mannwhitneyu, fisher_exact
RUNS = Path("/home/cvlab21/project/chan/alpasim-runs")
REPO = Path("/home/cvlab21/project/chan/alpamayo-model-compression")
SETS = {"150": "m2601_merged_{}", "hard100": "h100_merged_{}"}
ARMS = {"base": "baseline", "traj": "slim_traj_u40_v2", "coc": "slim_coc_u40_v2", "dual": "slim_dual_u40_v2"}
MET = ["collision_at_fault", "collision_any", "offroad", "progress_rel", "dist_to_gt_trajectory", "plan_deviation", "min_distance_to_obstacle_m", "dist_traveled_m"]
rng = np.random.default_rng(0)
feats = pd.read_csv(REPO / "outputs" / "scene_difficulty" / "scene_difficulty_public2601.csv")
print("scene feature columns:", list(feats.columns)[:14])
sid_col = [c for c in feats.columns if "scene" in c.lower()][0]


def per_scene(run):
    d = json.load(open(RUNS / run / "aggregate" / "results-summary.json"))
    rows = []
    for r in d["rollouts"]:
        m = r["metrics"] or {}
        rows.append({"scene": r["clipgt_id"], "score": r["score"], **{k: m.get(k) for k in MET}})
    df = pd.DataFrame(rows)
    return df.groupby("scene").mean(numeric_only=True), df


def ci(x, n=3000):
    x = np.asarray(x, float); m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]); return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


L = []
for sname, pat in SETS.items():
    sc, roll = {}, {}
    for a, cfg in ARMS.items():
        sc[a], roll[a] = per_scene(pat.format(cfg))
    scenes = sorted(set.intersection(*(set(s.index) for s in sc.values())))
    n = len(scenes); k = max(3, round(n / 10))
    df = pd.DataFrame(index=scenes)
    for a in ARMS:
        df[f"score_{a}"] = sc[a].loc[scenes, "score"]
        for m in MET:
            df[f"{m}_{a}"] = sc[a].loc[scenes, m]
    for a in ("traj", "coc", "dual"):
        df[f"d_{a}"] = df[f"score_{a}"] - df["score_base"]
    L.append(f"\n=================== closed loop {sname}: {n} scenes x {len(roll['base']) // n} rollouts ===================")
    for a in ("traj", "coc", "dual"):
        mn, lo, hi = ci(df[f"d_{a}"]); p = wilcoxon(df[f"d_{a}"])[1] if (df[f"d_{a}"] != 0).any() else 1
        worse = (df[f"d_{a}"] < -0.05).mean(); better = (df[f"d_{a}"] > 0.05).mean()
        L.append(f"  {a:4s} score {df[f'score_{a}'].mean():.3f} vs base {df['score_base'].mean():.3f}: d mean {mn:+.3f} [{lo:+.3f},{hi:+.3f}] Wilcoxon p={p:.2g} | worse(<-0.05) {worse:.0%} better(>+0.05) {better:.0%}")
    # gates and continuous failure modes, rollout level
    L.append("  failure modes (rollout-level rates / means): " + ", ".join(MET[:6]))
    for a in ARMS:
        r = roll[a]
        L.append(f"    {a:4s} at-fault {r['collision_at_fault'].mean():.3f} any-coll {r['collision_any'].mean():.3f} offroad {r['offroad'].mean():.3f} | progress_rel {r['progress_rel'].mean():.3f} dist_to_gt {r['dist_to_gt_trajectory'].mean():.2f} plan_dev {r['plan_deviation'].mean():.3f} | min obstacle dist {r['min_distance_to_obstacle_m'].mean():.2f}")
    # overlap of worst scenes
    tails = {a: df[f"d_{a}"].rank() <= k for a in ("traj", "coc", "dual")}
    jac = lambda x, y: (x & y).sum() / max((x | y).sum(), 1)
    rho = spearmanr(df["d_traj"], df["d_coc"])[0]
    L.append(f"  [overlap] Spearman(d_traj, d_coc) {rho:+.3f}; worst-{k} Jaccard traj/coc {jac(tails['traj'], tails['coc']):.2f} (chance {k / (2 * n - k):.2f}); "
             f"shared worst scenes {int((tails['traj'] & tails['coc']).sum())}")
    for a, o in (("traj", "coc"), ("coc", "traj")):
        t = tails[a]
        L.append(f"  on the {a} worst-{k}: d_{a} med {df.loc[t, f'd_{a}'].median():+.2f} | d_{o} med {df.loc[t, f'd_{o}'].median():+.2f} (within -0.05: {(df.loc[t, f'd_{o}'] > -0.05).mean():.0%}) | d_dual med {df.loc[t, 'd_dual'].median():+.2f} (within -0.05: {(df.loc[t, 'd_dual'] > -0.05).mean():.0%})")
        # what happened in those scenes
        L.append(f"      in-tail gates: at-fault {a} {df.loc[t, f'collision_at_fault_{a}'].mean():.2f} (base {df.loc[t, 'collision_at_fault_base'].mean():.2f}), offroad {df.loc[t, f'offroad_{a}'].mean():.2f} (base {df.loc[t, 'offroad_base'].mean():.2f}), "
                 f"progress_rel {df.loc[t, f'progress_rel_{a}'].mean():.2f} (base {df.loc[t, 'progress_rel_base'].mean():.2f}), dist_to_gt {df.loc[t, f'dist_to_gt_trajectory_{a}'].mean():.1f} (base {df.loc[t, 'dist_to_gt_trajectory_base'].mean():.1f}), plan_dev {df.loc[t, f'plan_deviation_{a}'].mean():.2f} (base {df.loc[t, 'plan_deviation_base'].mean():.2f})")
        # scene features
        f = feats.set_index(sid_col).reindex(scenes)
        for c in [c for c in ("v_mean", "yaw_total_deg", "hard_score", "path_len_m", "n_agents") if c in f.columns]:
            x, y = f.loc[t, c].dropna(), f.loc[~t, c].dropna()
            if len(x) > 2 and len(y) > 2:
                L.append(f"      {c}: tail median {x.median():.2f} vs rest {y.median():.2f} (MWU p={mannwhitneyu(x, y)[1]:.2g})")
    df.to_parquet(f"closedloop_{sname}.parquet")
Path("closedloop_summary.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
