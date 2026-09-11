# Ported-code guard. Runs on CPU in a second; no model, no GPU, no calibration data.
"""lp_core 의 salience/축약 수식이 upstream 과 같은 값을 내는지 난수로 대조한다.

이식된 코드에서 정작 검증하기 어려운 것이 "수식이 옮겨지는 동안 상하지 않았는가"다. upstream
`LLMPruner/pruner/hf_llama_pruner.py::TaylorImportance` 의 해당 분기를 **여기에 독립적으로 다시
적어** 같은 입력에 물리고 비교한다. lp_core 를 참조하지 않고 손으로 옮긴 식이므로, 둘이 맞으면
두 곳이 같은 실수를 했을 때만 통과한다.

같이 확인하는 것:
  * `param_first` 는 first_order="mean"/"sum" 어느 쪽이든 **순위가 같다** (전 파이프라인이
    salience 에 대해 1차 동차이므로 전역 배율은 순서를 못 바꾼다).
  * `first_order="sum"` 으로 두면 N 이 커질수록 `param_mix` 가 `param_first` 로 **붕괴**한다 --
    이식하면서 고친 것이 실재하는 문제였음을 수치로 남긴다.

Usage:
  .venv/bin/python experiments/llm_pruner/test_lp_formulas.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import lp_core as lp


def upstream_salience(w, g, acc_grad, taylor):
    """hf_llama_pruner.py:266-271 을 그대로 옮긴 것. acc_grad = (1/N) Σ_j g_j².

    upstream 은 g 자리에 '배치 1회 backward 의 .grad' 를 쓰는데, 그 손실이 배치 평균이므로
    그것은 per-sample 평균 gradient 다 (_upstream_reference/hf_prune.py:145-148).
    """
    salience = w * g
    if taylor == "param_second":
        salience = w * acc_grad * w
    elif taylor == "param_mix":
        salience = salience - 0.5 * w * acc_grad * w
    return salience


def upstream_channel(salience, axis_kind, taylor):
    """hf_llama_pruner.py:274-292. out -> dim 1, in -> dim 0."""
    dim = 1 if axis_kind == "out" else 0
    if taylor == "vectorize":
        return salience.sum(dim).abs()
    return salience.abs().sum(dim)


def main():
    torch.manual_seed(0)
    n = 7                      # 표본 수
    rows, cols = 12, 20
    w = torch.randn(rows, cols, dtype=torch.float64)
    per_sample = torch.randn(n, rows, cols, dtype=torch.float64)   # g_j
    g_sum = per_sample.sum(0)                  # GradAccumulator 가 담는 값
    g2_sum = (per_sample ** 2).sum(0)          # grad_sq 가 담는 값
    acc_grad = g2_sum / n                      # upstream 의 acc_grad
    g_batch = g_sum / n                        # upstream 의 .grad (배치 평균 손실)

    ok = True
    print(f"{'taylor':14s} {'axis':4s} {'max|diff|':>12s}")
    for taylor in lp.TAYLOR_VARIANTS:
        for axis in ("out", "in"):
            mine = lp._salience(w, g_sum, g2_sum, taylor, n, first_order="mean")
            mine = lp._channel_reduce(mine, axis, taylor)
            theirs = upstream_channel(upstream_salience(w, g_batch, acc_grad, taylor),
                                      axis, taylor)
            d = float((mine - theirs).abs().max())
            scale = max(float(theirs.abs().max()), 1e-30)
            bad = d > 1e-12 * scale
            ok &= not bad
            print(f"{taylor:14s} {axis:4s} {d / scale:12.2e}" + ("   <-- 불일치" if bad else ""))

    # group 축약과 head fold
    stacked = torch.randn(3, 24, dtype=torch.float64)
    for red in lp.GROUP_REDUCTIONS:
        mine = lp._reduce_group(stacked, red)
        theirs = {"sum": stacked.sum(0), "mean": stacked.mean(0), "max": stacked.max(0)[0],
                  "prod": torch.prod(stacked, 0), "first": stacked[0],
                  "second": stacked[1]}[red]
        d = float((mine - theirs).abs().max())
        ok &= d == 0
        print(f"group_reduction {red:7s} max|diff| {d:.1e}")

    # param_first 의 순위는 first_order 에 불변인가
    a = lp._channel_reduce(lp._salience(w, g_sum, g2_sum, "param_first", n, "mean"),
                           "out", "param_first")
    b = lp._channel_reduce(lp._salience(w, g_sum, g2_sum, "param_first", n, "sum"),
                           "out", "param_first")
    same_rank = torch.equal(torch.argsort(a), torch.argsort(b))
    ok &= same_rank
    print(f"\nparam_first 순위 불변 (mean vs sum): {same_rank}")

    # first_order="sum" 이면 param_mix 가 param_first 로 붕괴하는가 (고치기 전의 결함)
    print("\nfirst_order='sum' 일 때 param_mix 가 param_first 로 붕괴하는 정도")
    print(f"  {'N':>6s} {'Spearman(mix, first)':>22s}")
    for nn in (1, 10, 100):
        ps = torch.randn(nn, rows, cols, dtype=torch.float64)
        gs, g2s = ps.sum(0), (ps ** 2).sum(0)
        mix = lp._channel_reduce(lp._salience(w, gs, g2s, "param_mix", nn, "sum"),
                                 "out", "param_mix")
        first = lp._channel_reduce(lp._salience(w, gs, g2s, "param_first", nn, "sum"),
                                   "out", "param_first")
        rho = np.corrcoef(np.argsort(np.argsort(mix.numpy())),
                          np.argsort(np.argsort(first.numpy())))[0, 1]
        print(f"  {nn:6d} {rho:22.4f}")

    print("\n" + ("모든 대조 통과" if ok else "실패 -- 위 불일치를 보세요"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
