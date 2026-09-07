"""전체 run 과 그 앞부분 run 만으로 **나머지 부분**의 importance run 을 만든다. GPU 불필요.

`make_block_importance.py` 는 `importance_perclip.npz` 를 읽어 임의의 행 부분집합을 평균낸다.
그 파일은 클립 하나마다 전체 importance 배열을 통째로 들고 있어서 4,000 클립이면 20 GB 급이고,
측정을 돌린 기계(여기서는 cvlab20)에만 남아 있는 일이 많다. 그런데 저장되는 값은 단순 평균이므로,
앞 k 개와 전체 n 개의 평균이 있으면 나머지 n-k 개의 평균은 산술로 정확히 나온다:

    mean_all = (k*mean_head + (n-k)*mean_tail) / n
    ->  mean_tail = (n*mean_all - k*mean_head) / (n - k)

부동소수점 오차 말고는 근사가 아니다. 그래서 per-clip 파일을 옮기지 않고도 서로소인 두 번째
캘리브레이션 추출을 만들 수 있고, 그것이 `2026-09-06_calibration-size-closedloop.html` 의
두 번째 arm 이다.

안전장치: 앞부분 run 의 클립 순서가 전체 run 의 접두사와 정말 같은지 확인하고(아니면 행 분할이
어긋난다), 복원 항등 `(k*head + (n-k)*tail)/n == all` 을 모든 키에서 검사한 뒤에만 저장한다.

`importance_st4000_b2000` 은 원래 반반 특수형 `2*all - head` 로 만들었고 이 스크립트의 일반형과
연산 순서가 달라 저장값이 비트 단위로는 다르다(상대차 ~1e-16). 다만 `select_mask_ratios` 는
층 안에서 argsort 만 하므로 **선택 집합은 Q/MLP 모두 정확히 일치**하고, `dual_u40_v2` 는 선택만으로
결정되는 surgery 라 체크포인트도 같다. 즉 이 스크립트로 실험을 그대로 재현할 수 있다.

Usage:
  .venv/bin/python experiments/evaluation/make_complement_importance.py \
      --all importance_st4000 --head importance_st4000_c2000 --out importance_st4000_b2000
"""

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", required=True, help="전체 클립을 평균낸 importance run 이름")
    ap.add_argument("--head", required=True, help="그 앞부분을 평균낸 importance run 이름")
    ap.add_argument("--out", required=True, help="만들 나머지 부분 run 이름")
    ap.add_argument("--rtol", type=float, default=1e-6,
                    help="복원 항등 검사의 상대 허용오차 (배열 최대 절대값 기준)")
    args = ap.parse_args()

    d_all, d_head = REPO / "outputs" / args.all, REPO / "outputs" / args.head
    out = REPO / "outputs" / args.out

    c_all = json.loads((d_all / "config.json").read_text())
    c_head = json.loads((d_head / "config.json").read_text())
    n, k = c_all["num_clips"], c_head["num_clips"]
    ids_all, ids_head = c_all["clip_ids"], c_head["clip_ids"]
    if k >= n:
        raise SystemExit(f"--head 가 --all 보다 작아야 합니다 (k={k}, n={n})")
    if len(ids_all) != n or len(set(ids_all)) != n:
        raise SystemExit(f"--all 의 clip_ids 가 {len(ids_all)}개(고유 {len(set(ids_all))})로 "
                         f"num_clips={n} 과 맞지 않습니다")
    if ids_all[:k] != ids_head:
        raise SystemExit("--head 의 클립 순서가 --all 의 접두사가 아닙니다 -- 행 분할이 어긋납니다")

    A = dict(np.load(d_all / "importance.npz"))
    H = dict(np.load(d_head / "importance.npz"))
    if set(A) != set(H):
        raise SystemExit(f"키가 다릅니다: only-all {set(A) - set(H)}, only-head {set(H) - set(A)}")

    m = n - k
    T = {}
    for key, arr in A.items():
        if arr.shape != H[key].shape:
            raise SystemExit(f"{key}: shape 불일치 {arr.shape} vs {H[key].shape}")
        a, h = arr.astype(np.float64), H[key].astype(np.float64)
        t = (n * a - k * h) / m
        # 복원 항등을 그대로 되짚어 본다. 통과하지 못하면 전제(단순 평균, 접두사)가 깨진 것이다.
        err = np.abs((k * h + m * t) / n - a).max()
        scale = max(float(np.abs(a).max()), 1e-12)
        if err > args.rtol * scale:
            raise SystemExit(f"{key}: 복원 항등 실패 err={err:.3e} scale={scale:.3e}")
        T[key] = t.astype(arr.dtype)

    for key, v in T.items():
        if (H[key] >= 0).all() and (v < 0).any():
            print(f"경고: {key} 는 앞부분이 전부 비음수인데 유도값의 "
                  f"{float((v < 0).mean()) * 100:.2f}% 가 음수입니다", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "importance.npz", **T)
    cfg = dict(c_all)
    cfg.update({"num_clips": m, "clip_ids": ids_all[k:],
                "derived_from": f"{args.all} clips {k}-{n - 1}, computed as "
                                f"({n}*mean({args.all}) - {k}*mean({args.head})) / {m}"})
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "metrics.json").write_text(json.dumps({"n_clips": m}, indent=2))

    print(f"복원 항등 통과 ({len(T)}개 키)")
    print(f"클립 교집합 (head ∩ out) = {len(set(ids_head) & set(ids_all[k:]))}  (0이어야 함)")
    print(f"-> {out}  ({m} clips)")


if __name__ == "__main__":
    main()
