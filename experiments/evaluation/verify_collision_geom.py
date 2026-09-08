"""Phase 0 gate: does the GT trajectory collide with the labelled obstacles?

It must not. The ego drove that path and the obstacles are what its own sensors saw, so a
GT collision rate above a couple of percent means the geometry is wrong -- the frame
transform, the time interpolation, or the box sizes -- not that the data is interesting.
Running this before anything else costs no GPU and decides whether the rest of
plans/2026-09-08_openloop-collision-proxy.md is worth doing.

G0 PASS < 2%, FAIL >= 5%; in between, look at the diagnostics before continuing.

calib_100 is the only set whose obstacle chunks are downloaded (96/100), which is why the
gate runs there -- it is calibration data and is never evaluated on, so using it here
costs nothing.

    python experiments/evaluation/verify_collision_geom.py --set calib_100
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import collision_lib as cl
import sample_cache as sc

O = Path("/mnt/nvme1n1/ad_vla/outputs/chan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="which", default="calib_100")
    ap.add_argument("--cache", default="calib")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=O / "collision_geom_check")
    args = ap.parse_args()

    man = pd.read_parquet(O / "eval_sets" / f"{args.which}.parquet")
    if args.limit:
        man = man.head(args.limit)

    rows, skipped = [], Counter()
    for i, m in enumerate(man.itertuples()):
        chunk = int(m.chunk)
        obs = cl.load_obstacles(str(m.clip_id), chunk)
        if obs is None or obs.empty:
            skipped["장애물 라벨 없음"] += 1
            continue
        size = cl.load_ego_size(str(m.clip_id), chunk)
        if size is None:
            skipped["자차 크기 없음"] += 1
            continue
        d = sc.load_cached(sc.path_for(args.cache, str(m.clip_id), int(m.t0_us)))
        fx = np.asarray(d["ego_future_xyz"])[0, 0]     # (64, 3)
        fr = np.asarray(d["ego_future_rot"])[0, 0]     # (64, 3, 3)
        drop = cl.ego_self_tracks(obs, int(m.t0_us), size)
        r = cl.score_path(fx[:, :2], fx, fr, obs, int(m.t0_us), size, drop_tracks=drop)
        r.update(clip_id=str(m.clip_id), chunk=chunk, n_tracks=int(obs.track_id.nunique()),
                 n_self_labelled=len(drop))
        rows.append(r)
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(man)}] 충돌 {sum(x['collide'] for x in rows)}", flush=True)

    n = len(rows)
    hits = [r for r in rows if r["collide"]]
    rate = 100 * len(hits) / n if n else float("nan")
    verdict = "PASS" if rate < 2 else ("FAIL" if rate >= 5 else "INCONCLUSIVE")
    print(f"\n채점한 클립 {n}  (건너뜀 {dict(skipped)})")
    print(f"GT 궤적 충돌률 {len(hits)}/{n} = {rate:.1f}%   G0 {verdict}")
    if hits:
        print("\n충돌 클래스:", Counter(h["hit_class"] for h in hits).most_common())
        st = [h["hit_step"] for h in hits]
        print(f"충돌 시점(step): 중앙값 {np.median(st):.0f} / 64, "
              f"범위 {min(st)}-{max(st)}")
    self_n = sum(r["n_self_labelled"] for r in rows)
    print(f"\n자차로 오라벨된 트랙 제거: {self_n}개 "
          f"({sum(1 for r in rows if r['n_self_labelled'])} 클립)")
    md = [r["min_center_dist"] for r in rows if r["min_center_dist"] is not None]
    if md:
        print(f"\n최소 중심거리: 중앙값 {np.median(md):.2f} m, "
              f"5퍼센타일 {np.percentile(md, 5):.2f} m")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(
        {"set": args.which, "n": n, "collide": len(hits), "rate_pct": rate,
         "verdict": verdict, "skipped": dict(skipped), "rows": rows},
        indent=2, default=float))
    (args.out / "summary.txt").write_text(
        f"GT 충돌 {len(hits)}/{n} = {rate:.1f}%  G0 {verdict}\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
