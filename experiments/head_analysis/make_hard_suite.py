"""Draw the hard-N closed-loop scene set and write it as an alpasim test suite CSV.

The 150-scene matrix is `sorted(scene_id)[:150]` of `public_2601`. Those scenes are **excluded
here** so the hard set is a genuinely new sample -- the whole point is to stop reporting on a
sample that turned out easier than the suite it came from (0.742 vs 0.660, p=0.039; at-fault
collision 2.0% vs 5.5%). Scenes whose GT ego travels < 5 m are excluded too: `progress_score` is
overridden to 1.0 for them, so they carry no information.

Ranking is `hard_score = z(v_mean) + z(yaw_total_deg)` from `extract_scene_feats.py`. It was
validated out-of-sample on exactly the scenes this script draws from -- rho=-0.293 (p=1.5e-16)
against sangoh's 913-scene unpruned run, with the top 100 measuring 0.499 and the bottom 100
0.826. It **saturates above the 80th percentile** (the 80-90% and 90-100% bands both sit at
~0.50), so `--band` draws from that plateau instead of the extreme tail when N needs to grow
past ~100 without getting easier.

Rows are written with the suite's own uuid, never a bare scene_id: 159 of `public_2601`'s
scene_ids also exist in `public_2604` and `query_by_scene_ids` silently resolves those to the
newer 26.04 render. The output is appended to `scenes.suites_csv` by `launch_alpasim_shards.sh`.

Usage:
  python make_hard_suite.py --out outputs/scene_difficulty/hard100_suite.csv
  python make_hard_suite.py -n 183 --band 80 100 --suite-id public_2601_hard183 --out ...
"""

import argparse
import csv
import json
from pathlib import Path

import extract_scene_feats
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SUITES = Path("/home/cvlab21/project/chan/alpasim/data/scenes/sim_suites.csv")
MATRIX_N = 150          # the shipped closed-loop matrix is sorted(scene_id)[:150]


def suite_rows(suite):
    rows = [r for r in csv.DictReader(SUITES.open()) if r["test_suite_id"] == suite]
    return sorted(rows, key=lambda r: r["scene_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", type=Path,
                    default=REPO / "outputs/scene_difficulty/scene_feats_public2601.json",
                    help="output of extract_scene_feats.py")
    ap.add_argument("--suite", default="public_2601")
    ap.add_argument("-n", "--num-scenes", type=int, default=100)
    ap.add_argument("--band", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="draw uniformly from this hard_score percentile band of the whole suite "
                         "instead of taking the top N (use 80 100 -- the score saturates there)")
    ap.add_argument("--suite-id", default=None, help="test_suite_id to write (default hard<N>)")
    ap.add_argument("--exclude-runs", type=Path, default=None,
                    help="a merged run dir; its scenes are cross-checked against the first-150 "
                         "definition and the run must not disagree")
    ap.add_argument("--min-gt-path-m", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=20260903, help="only used with --band")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = suite_rows(args.suite)
    uuid_of = {r["scene_id"]: r["uuid"] for r in rows}
    excluded = {r["scene_id"] for r in rows[:MATRIX_N]}
    print(f"{args.suite}: {len(rows)} scenes | 매트릭스 첫 {MATRIX_N}개 제외")

    # If a run dir is given, the exclusion set must match what was actually evaluated. A silent
    # mismatch here is exactly how an overlapping "new" sample would get shipped.
    if args.exclude_runs:
        agg = json.loads((args.exclude_runs / "aggregate" / "results-summary.json").read_text())
        ran = {r["clipgt_id"] for r in agg["rollouts"]}
        if ran != excluded:
            raise SystemExit(
                f"이미 평가된 씬 집합이 첫 {MATRIX_N}개 정의와 다릅니다 "
                f"(런에만 {len(ran - excluded)}개, 정의에만 {len(excluded - ran)}개). "
                "겹침 위험이 있어 중단합니다.")
        print(f"  교차 확인 OK: {args.exclude_runs.name} 의 {len(ran)}개가 정의와 일치")

    feats = {r["scene_id"]: r for r in json.loads(args.feats.read_text())}
    missing = [s for s in uuid_of if s not in feats]
    if missing:
        raise SystemExit(f"피처가 없는 씬 {len(missing)}개 -- extract_scene_feats.py 를 먼저 실행")

    # Recomputed from the features rather than read back from the file: the score definition then
    # lives in exactly one place, and an older feature dump without the field still works.
    order = sorted(uuid_of)
    hs = extract_scene_feats.hard_score([feats[s] for s in order])
    H = dict(zip(order, hs))

    pool = [s for s in order
            if s not in excluded and feats[s]["gt_path_m"] >= args.min_gt_path_m]
    print(f"  후보 풀: {len(pool)}개 "
          f"(제외 {len(excluded)} + GT 주행 <{args.min_gt_path_m:g}m "
          f"{len(order) - len(excluded) - len(pool)})")

    if args.band:
        lo, hi = np.percentile(hs, args.band)
        band = [s for s in pool if lo <= H[s] <= hi]
        if len(band) < args.num_scenes:
            raise SystemExit(f"밴드 {args.band} 에 {len(band)}개뿐 -- N={args.num_scenes} 불가")
        rng = np.random.RandomState(args.seed)
        picked = [band[i] for i in sorted(rng.choice(len(band), args.num_scenes, replace=False))]
        how = f"{args.band[0]:g}-{args.band[1]:g}% 밴드({len(band)}개)에서 seed={args.seed} 추출"
    else:
        picked = sorted(pool, key=lambda s: -H[s])[:args.num_scenes]
        how = "hard_score 상위"

    suite_id = args.suite_id or f"{args.suite}_hard{args.num_scenes}"

    # --- 불변식: 여기서 걸리면 절대 쓰면 안 되는 스위트다
    assert len(picked) == args.num_scenes, "요청한 씬 수와 다름"
    assert len(set(picked)) == len(picked), "중복 씬"
    assert not (set(picked) & excluded), "이미 평가한 150씬과 겹침"
    assert all(uuid_of[s] for s in picked), "uuid 없는 씬"

    out = args.out if args.out.is_absolute() else REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["test_suite_id", "scene_id", "uuid"])
        for s in picked:
            w.writerow([suite_id, s, uuid_of[s]])

    sel = np.array([H[s] for s in picked])
    def med(k):
        return np.median([feats[s][k] for s in picked])

    print(f"\n{suite_id}: {len(picked)}개 -> {out}   ({how})")
    print(f"  겹침 검사: 매트릭스 150씬과 교집합 {len(set(picked) & excluded)}개")
    print(f"  hard_score {sel.min():+.2f} .. {sel.max():+.2f} "
          f"(스위트 {100 * (hs < sel.min()).mean():.0f}퍼센타일부터)")
    print(f"  중앙값  v_mean {med('v_mean'):.1f} m/s | 선회 {med('yaw_total_deg'):.0f} deg | "
          f"경로 {med('gt_path_m'):.0f} m   (스위트 중앙값 "
          f"{np.median([feats[s]['v_mean'] for s in order]):.1f} / "
          f"{np.median([feats[s]['yaw_total_deg'] for s in order]):.0f} / "
          f"{np.median([feats[s]['gt_path_m'] for s in order]):.0f})")
    print(f"\n실행: SUITE={suite_id} PREFIX=h{args.num_scenes}_ SCENES_CSV={out} \\")
    print("        DRIVER_OMP_THREADS=8 bash experiments/head_analysis/launch_alpasim_shards.sh \\")
    print(f'        <config> {args.num_scenes} 2 "4 5 6 7"')


if __name__ == "__main__":
    main()
