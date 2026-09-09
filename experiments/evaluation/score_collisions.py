"""Score an arm's saved trajectories against the labelled obstacles, and gate on G1.

Three numbers per clip, because any one of them alone misleads:
  collide_any   at least one of the k samples hits            -- the worst case
  collide_frac  fraction of the k that hit                    -- the distribution
  collide_best  whether the lowest-ADE sample hits            -- the one you would pick
The GT path is scored alongside on the same clips: it is the noise floor, since the ego
demonstrably drove it, and the difference is the reading.

G1 asks whether the metric discriminates at all. If a pruned model's paths never come
near an obstacle, the proxy is measuring nothing and the remaining arms are not worth
6 GPU-hours. It is a McNemar test on the paired per-clip indicators, which is the right
test here: the same clip, the same labels, two paths.

    python experiments/evaluation/score_collisions.py --arm baseline_pred
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
SETS = [("test", "test_500", "test", "test500"),
        ("indist", "indist_500", "eval", "val500"),
        ("oodval", "ood_val", "ood", "OOD-val")]


def rows_of(arm, suffix):
    d = O / f"{arm}_{suffix}"
    out = {}
    for f in sorted(d.glob("*_s*of*.json")):
        for r in json.loads(f.read_text()):
            out[r["clip_id"]] = r
    return out


def mcnemar(a, b):
    """Exact McNemar on paired binary outcomes: only the discordant pairs carry signal."""
    from scipy.stats import binomtest

    n01 = int(np.sum((~np.asarray(a)) & np.asarray(b)))
    n10 = int(np.sum(np.asarray(a) & (~np.asarray(b))))
    if n01 + n10 == 0:
        return 1.0, n10, n01
    return float(binomtest(n10, n01 + n10, 0.5).pvalue), n10, n01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out or O / f"{args.arm}_collision"

    report = {}
    for suffix, man_name, cache, label in SETS:
        rows = rows_of(args.arm, suffix)
        if not rows:
            print(f"{label}: 결과 없음, 건너뜀")
            continue
        man = pd.read_parquet(O / "eval_sets" / f"{man_name}.parquet")
        if args.limit:
            man = man.head(args.limit)

        per, skipped = [], Counter()
        for m in man.itertuples():
            cid = str(m.clip_id)
            if cid not in rows or "pred_xy_k" not in rows[cid]:
                skipped["궤적 없음"] += 1
                continue
            obs = cl.load_obstacles(cid, int(m.chunk))
            size = cl.load_ego_size(cid, int(m.chunk))
            if obs is None or obs.empty or size is None:
                skipped["라벨 없음"] += 1
                continue
            d = sc.load_cached(sc.path_for(cache, cid, int(m.t0_us)))
            fx = np.asarray(d["ego_future_xyz"])[0, 0]
            fr = np.asarray(d["ego_future_rot"])[0, 0]
            drop = cl.ego_self_tracks(obs, int(m.t0_us), size)

            # one interpolation + frame transform for the clip, reused by all 9 paths
            prep = cl.prepare_clip(obs, fx, fr, int(m.t0_us), len(fx), drop_tracks=drop)
            gt = cl.score_path(fx[:, :2], fx, fr, obs, int(m.t0_us), size, prepared=prep)
            preds = [cl.score_path(np.asarray(p), fx, fr, obs, int(m.t0_us), size,
                                   prepared=prep)
                     for p in rows[cid]["pred_xy_k"]]
            hits = [p["collide"] for p in preds]
            best = int(np.argmin(rows[cid]["ade_rollout_k"]))
            # a clip can have labels and still have no obstacle inside the 6.4 s window --
            # every track's observed span falls outside it, or the only track was the ego
            # self-label. There is then no distance to report, and it is not a collision.
            dists = [p["min_center_dist"] for p in preds if p["min_center_dist"] is not None]
            per.append({
                "clip_id": cid,
                "gt_collide": gt["collide"],
                "collide_any": any(hits),
                "collide_frac": float(np.mean(hits)),
                "collide_best": hits[best],
                "min_dist": min(dists) if dists else None,
                "n_in_window": len(dists),
                "hit_class": next((p["hit_class"] for p in preds if p["collide"]), None),
            })
            if len(per) % 50 == 0:
                print(f"  {label} [{len(per)}] any={sum(x['collide_any'] for x in per)}",
                      flush=True)

        n = len(per)
        gt_r = 100 * np.mean([x["gt_collide"] for x in per])
        e = {"n": n, "skipped": dict(skipped), "gt_pct": gt_r,
             "any_pct": 100 * np.mean([x["collide_any"] for x in per]),
             "frac_pct": 100 * np.mean([x["collide_frac"] for x in per]),
             "best_pct": 100 * np.mean([x["collide_best"] for x in per]),
             "min_dist_median": float(np.median(
                 [x["min_dist"] for x in per if x["min_dist"] is not None])),
             "no_obstacle_in_window": sum(1 for x in per if x["min_dist"] is None),
             "classes": dict(Counter(x["hit_class"] for x in per if x["hit_class"])),
             "rows": per}
        p, n10, n01 = mcnemar([x["collide_any"] for x in per],
                              [x["gt_collide"] for x in per])
        e.update(mcnemar_p=p, pred_only=n10, gt_only=n01)
        report[label] = e

        print(f"\n=== {label}  (n={n}, 건너뜀 {dict(skipped)}) ===")
        print(f"  GT 궤적          {gt_r:5.1f}%")
        print(f"  예측 any(8중1)   {e['any_pct']:5.1f}%   "
              f"McNemar p={p:.2e}  (예측만 {n10} / GT만 {n01})")
        print(f"  예측 frac(평균)  {e['frac_pct']:5.1f}%")
        print(f"  예측 best(최저ADE) {e['best_pct']:5.1f}%")
        print(f"  최소 중심거리 중앙값 {e['min_dist_median']:.2f} m"
              f"  (구간 내 장애물 없음 {e['no_obstacle_in_window']}클립)")
        if e["classes"]:
            print(f"  충돌 클래스 {sorted(e['classes'].items(), key=lambda x: -x[1])[:4]}")

    if report:
        t = report.get("test500")
        verdict = "PASS" if t and t["any_pct"] > t["gt_pct"] and t["mcnemar_p"] < 0.05 \
            else "FAIL"
        print(f"\nG1 {verdict}  (test500에서 예측 충돌률이 GT보다 유의하게 높은가)")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(
            json.dumps({"arm": args.arm, "G1": verdict, "sets": report},
                       indent=2, default=float))
        (out_dir / "summary.txt").write_text("\n".join(
            f"{k}: GT {v['gt_pct']:.1f}%  any {v['any_pct']:.1f}%  "
            f"frac {v['frac_pct']:.1f}%  best {v['best_pct']:.1f}%  p={v['mcnemar_p']:.2e}"
            for k, v in report.items()) + f"\nG1 {verdict}\n")
        print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
