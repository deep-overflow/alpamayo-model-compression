"""CoC 퇴화율을 씬 단위 페어드로 검정할 수 있게 씬별 값을 남긴다.

`analyze_alpasim.py` 는 arm 별 rollout 평균만 저장하므로 "5.92% vs 4.66% 가 유의한가" 를
답할 수 없다. 같은 파서를 재사용해 rollout 단위 값을 얻고, 씬당 rollout 을 평균해 씬별
배열로 저장한다. 통계는 저장하지 않는다 -- 리포트 쪽에서 이 배열로 다시 계산하므로 숫자를
손으로 옮길 일이 없다.

alpasim venv 로 돌려야 한다. ASL 프로토버프를 읽는 데 alpasim_utils 가 필요하다:

  cd /home/cvlab21/project/chan/alpasim && uv run python \
      <repo>/experiments/evaluation/coc_paired_tyrk.py --out <abs>/outputs/tyrK_pairs

`--arm NAME RUNDIR` 를 반복하면 임의의 arm 집합을 쓸 수 있다 (생략하면 150씬 tyrK 3 arm).
씬 x rollout x arm 당 ASL 하나를 읽으므로 150씬 3 arm 이면 900개, ~10분 걸린다.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/head_analysis"))
import analyze_alpasim as A

DEFAULT_ARMS = [("baseline", "m2601_merged_baseline"),
                ("tyr_c100", "m2601_merged_slim_tyr_u40_r"),
                ("tyrK", "m2601_merged_slim_tyrK")]
KEYS = ("degenerate_frac", "empty_frac", "soup_frac")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path,
                    default=Path("/home/cvlab21/project/chan/alpasim-runs"))
    ap.add_argument("--out", type=Path, default=REPO / "outputs/tyrK_pairs")
    ap.add_argument("--arm", nargs=2, action="append", metavar=("NAME", "RUNDIR"),
                    help="arm 하나 (반복 가능). 생략하면 150씬 tyrK 3 arm")
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    arms = [tuple(a) for a in args.arm] if args.arm else DEFAULT_ARMS

    per_scene = {}
    for name, d in arms:
        run = args.runs_root / d
        acc = {}
        for r in A.load_rollouts(run):
            asl = run / "rollouts" / r["scene"] / r["rollout_id"] / "rollout.asl"
            texts = asyncio.run(A._read_coc(asl)) if asl.exists() else []
            st = A.coc_stats(texts)
            for k in KEYS:
                acc.setdefault(k, {}).setdefault(r["scene"], []).append(st[k])
        per_scene[name] = {k: {s: float(np.nanmean(v)) for s, v in acc[k].items()} for k in KEYS}
        mean = np.nanmean(list(per_scene[name]["degenerate_frac"].values()))
        print(f"{name}: 씬 {len(per_scene[name]['degenerate_frac'])}, 평균 퇴화 {mean:.4f}")

    scenes = sorted(set.intersection(*(set(v["degenerate_frac"]) for v in per_scene.values())))
    blob = {"n_scenes": len(scenes), "scenes": scenes,
            "per_scene": {n: {k: [v[k][s] for s in scenes] for k in KEYS}
                          for n, v in per_scene.items()}}
    (out / "coc_paired.json").write_text(json.dumps(blob))
    print(f"-> {out}/coc_paired.json  (공통 씬 {len(scenes)})")


if __name__ == "__main__":
    main()
