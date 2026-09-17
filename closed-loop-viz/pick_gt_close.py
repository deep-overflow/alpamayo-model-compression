"""The scenes where an arm's driven path stayed closest to alpasim's GT trajectory.

`dist_to_gt_trajectory` is the sim's own per-rollout measure of that (the `d2gt` column in
analyze_alpasim), so this ranks on it rather than re-deriving a distance from the poses --
the point is to select on the quantity the evaluation already uses.

Ranking on that alone is a trap, and the guard against it is the point of this script. A
rollout that stops early cannot leave the GT curve: the closest scene by raw d2gt drove
16.2 m of a 105.9 m route, scored 0.000 and went offroad on both rollouts, yet its
distance to the GT trajectory is 0.000. Half of a raw top-10 was that failure mode. So a
scene qualifies only when the ego actually drove the route -- `dist_traveled_m /
gt_dist_traveled_m >= --min-progress-frac` -- and only then is it ranked by d2gt.

Scenes are ranked by the mean over their rollouts, and score, gates, driven fraction and
GT path length are printed with each one. `--min-gt-m` additionally drops scenes whose GT
path is too short for the distance to mean much.

Also reports which of the picks the renderer can actually serve: it only loads the usdz its
--artifact-glob matched, and a sceneset directory holds whatever the last run materialised,
not the whole suite.
"""
import argparse
import collections
import json
from pathlib import Path

RUNS = Path("/home/cvlab21/project/chan/alpasim-runs")
ARTIFACTS = Path("/mnt/nvme1n1/ad_vla/data/nre-artifacts")
SUITES_CSV = Path("/home/cvlab21/project/chan/alpasim/data/scenes/sim_suites.csv")


def run_dir(config, prefix="m2601_merged_"):
    p = Path(config)
    return p if p.is_absolute() else RUNS / f"{prefix}{config}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="slim_dual_u40_v2")
    ap.add_argument("--prefix", default="m2601_merged_",
                    help="run-name prefix; h100_merged_ for the hard100 suite")
    ap.add_argument("--parent-suite", default="public_2601",
                    help="suite whose scene_id -> uuid mapping resolves the usdz check")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--min-gt-m", type=float, default=20.0,
                    help="drop scenes whose GT path is shorter than this")
    ap.add_argument("--min-progress-frac", type=float, default=0.9,
                    help="require dist_traveled / gt_dist_traveled at least this; a rollout "
                         "that stopped early sits at d2gt ~ 0 without having driven the route")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    d = json.loads(
        (run_dir(args.config, args.prefix) / "aggregate/results-summary.json").read_text())
    per = collections.defaultdict(list)
    for r in d["rollouts"]:
        m = r.get("metrics") or {}
        v = m.get("dist_to_gt_trajectory")
        if v is None:
            continue
        gt_m = float(m.get("gt_dist_traveled_m") or 0)
        frac = float(m.get("dist_traveled_m") or 0) / gt_m if gt_m else 0.0
        per[r["clipgt_id"]].append(
            (float(v), float(r["score"]), gt_m, float(m.get("offroad") or 0),
             float(m.get("collision_at_fault") or 0), frac))

    rows = []
    for s, vals in per.items():
        n = len(vals)
        rows.append((sum(v[0] for v in vals) / n, s, sum(v[1] for v in vals) / n,
                     max(v[2] for v in vals), sum(v[3] for v in vals),
                     sum(v[4] for v in vals), n, min(v[5] for v in vals)))
    long_enough = [r for r in rows if r[3] >= args.min_gt_m]
    kept = [r for r in long_enough if r[7] >= args.min_progress_frac]
    print(f"{args.config}: 씬 {len(rows)}개  ->  GT {args.min_gt_m:.0f} m 이상 "
          f"{len(long_enough)}개  ->  경로의 {args.min_progress_frac:.0%} 이상 주행 "
          f"{len(kept)}개")
    kept.sort()

    print(f"\n{'#':>2s} {'scene':44s} {'d2gt':>6s} {'score':>6s} {'주행/GT':>7s} "
          f"{'GT m':>7s}  게이트")
    picks = []
    for i, (d2, s, sc, gt, off, col, n, frac) in enumerate(kept[:args.top], 1):
        gates = []
        if off:
            gates.append(f"offroad {int(off)}/{n}")
        if col:
            gates.append(f"at-fault {int(col)}/{n}")
        print(f"{i:2d} {s:44s} {d2:6.3f} {sc:6.3f} {frac:7.2f} {gt:7.1f}  "
              f"{', '.join(gates) or '-'}")
        picks.append(s)

    print(f"\n비교 — 전체 중앙값 d2gt {sorted(r[0] for r in kept)[len(kept) // 2]:.3f}, "
          f"최댓값 {max(r[0] for r in kept):.3f}")

    # can the renderer serve these? it loads only what its glob matched
    print("\n렌더러 서빙 가능 여부 (씬셋별 usdz 보유):")
    try:
        import csv
        uuid_of = {}
        with open(SUITES_CSV) as fh:
            for row in csv.DictReader(fh):
                if row.get("test_suite_id") == args.parent_suite:
                    uuid_of[row["scene_id"]] = row["uuid"]
        for ss in sorted(ARTIFACTS.glob("scenesets/*")):
            have = sum(1 for s in picks
                       if s in uuid_of and (ss / f"{uuid_of[s]}.usdz").exists())
            print(f"  {ss.name}  usdz {len(list(ss.glob('*.usdz'))):4d}개 중 "
                  f"선정 씬 {have}/{len(picks)}")
        missing = [s for s in picks if s not in uuid_of]
        if missing:
            print(f"  (suite csv 에 없는 씬 {len(missing)}개)")
    except Exception as e:  # noqa: BLE001 -- availability check is advisory
        print(f"  확인 실패: {e}")

    if args.out:
        args.out.write_text("\n".join(picks) + "\n")
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
