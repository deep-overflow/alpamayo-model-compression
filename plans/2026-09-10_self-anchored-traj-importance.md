# 자기 앵커 궤적 중요도 — FM 손실의 타깃을 GT에서 모델 자신의 샘플로

## 가설

`dual = max(rank I_traj, rank I_CoC)`에서 **두 half의 앵커가 비대칭**이다.

- `I_CoC`는 **모델 자신의 rollout** NLL (`seq_tf = roll["sequences"]`). 참조 텍스트를 쓰지 않는다.
- `I_traj`는 **GT 앵커**. `x_t = (1−t)ε + t·x1`, `x1 = gt_actions(...)`, 타깃 `u = x1 − ε`.

궤적 분포는 멀티모달인데 GT는 그중 한 모드다. 모델이 다른 모드를 내놓는 클립에서 잔차
`v_θ(x_t,t) − (x1 − ε)`는 **모드 불일치라는 체계적 성분**에 지배되고, Taylor 점수
`∂L/∂g_u = 2(v_θ−u)ᵀ ∂v_θ/∂g_u`는 그 방향으로의 사영이 된다. 즉 **"이 모델이 무엇을 쓰는가"가
아니라 "무엇이 출력을 GT 쪽으로 옮기는가"** 를 재게 된다.

**H1**: `x1`을 모델 자신의 디노이즈 결과 `x̂₁`으로 바꾸면 그 편향이 사라지고, 중요도 추정의
**분산이 줄어든다**(= split-half 안정성이 오른다).

**H2**: 그 결과 kept set이 바뀌고, 같은 예산에서 개루프 성능이 달라진다.

H1은 기전에 대한 주장이고 H2는 결과에 대한 주장이다. **H1이 이 실험의 본체**다 — 오늘 확인한
대로 kept set 변위 크기는 손상을 예측하지 못하므로(`maxstep11` 7.0–9.7% → n.s.,
`dualsafe` 6.0% → +0.1945), H2만으로는 아무것도 설명하지 못한다.

## 왜 지금인가 — 2026-08-23 취소 근거의 전제가 틀렸다

`rollout-vs-tf-traj-importance` 메모는 유사 실험을 종료시켰다. 그 근거의 중심은:

> dense의 x̂₁은 GT에 가까우므로(minADE ~0.84) 자기 궤적 ≈ GT 보간

`baseline_ada_ps_indist` 500클립 × 8샘플로 재측정한 결과 **이 전제는 거짓이다**:

| | 값 | minADE@6 대비 |
|---|---:|---:|
| minADE@6 (프로토콜) | 0.824 m | 1.00× |
| **단일 draw (sample 0)** | **1.760 m** | **2.14×** |
| 8샘플 평균 | 1.789 m | 2.17× |
| 8샘플 중 최악 | 3.342 m | 4.06× |

0.824는 **6개 중 최소값**이지 모델의 샘플이 아니다. FM 손실이 노이즈와 짝짓는 것은 단일
draw이고 그것은 1.76 m다. 멀티모달성도 뚜렷하다:

- 한 클립의 8 draw가 평균 **2.65 m**(중앙값 2.25 m) 범위로 벌어짐
- **56.8%** 의 클립에서 draw들이 2 m 넘게 벌어짐
- 개별 draw의 **59.7%** 가 1 m보다 나쁨, **18.6%** 의 클립은 8개 전부 1 m보다 나쁨

**축도 다르다.** 취소된 실험(`--traj-mode infer`)은 **경로**를 바꿨고(자기 Euler 체인),
타깃은 여전히 GT였다(`MSE(final xy, gt xy)`). 이 계획은 **타깃**을 바꾼다. 직교한다.

메모의 다른 논거 중 **"traj 질량의 75%가 앞 3스텝(t=0.05/0.15/0.25)에 몰려 있고 거기선
x_t가 95/85/75% 노이즈라 TF와 rollout이 겹친다"는 이 축에 적용되지 않는다** — 측정 지점
`x_t`는 거의 안 움직이는 게 맞지만 타깃 `u = x1 − ε`는 `x1` 교체분만큼 통째로 움직인다.
그 논거는 경로 축 전용이었다.

살아남는 논거는 **잡음 바닥** 하나였다: calib_100을 50:50으로 갈라도 kept 겹침이
Q 0.860 / MLP 0.782.

**단, 이 값을 G2의 문턱으로 쓰면 안 된다** (2026-09-10 수정). 그것은 50클립 vs 50클립
일치도이고 G2가 재는 것은 100클립 vs 100클립이라 사과 대 오렌지다. 같은 n=100끼리의 바닥은
오늘 균형 노브 실험에서 잰 서로소 추출 간 변위 **9.0% (Q) / 14.6% (MLP)**, 즉 겹침
0.910 / 0.854다.

더 근본적으로, **변위 크기는 손상을 예측하지 않는다** — 오늘 세 사례가 그렇다
(`maxstep11` 7.0–9.7% → n.s., `dualfm10` 5.6% → +0.0001, `dualsafe` 6.0% → +0.1945).
그러므로 G2는 진행 여부를 가르는 품질 게이트가 아니라 **"기준이 no-op이 아닌가"** 검사로만
쓰고, 진행 여부는 G3가 가른다.

## 설계 — 결합(coupling)은 재고 정한다

$$\hat{x}_1^{(k)} = \text{Denoise}_{10}\big(\varepsilon_0^{(k)}\big), \qquad
L = \big\| v_\theta(x_t, t) - u \big\|^2$$

**갈림길은 손실 입력에 쓰는 노이즈다.** `x̂₁`을 만든 `ε₀`를 그대로 쓸 것인가(**A, tied**),
새로 뽑을 것인가(**B, independent**).

| | `x_t` | 타깃 `u` | 성격 |
|---|---|---|---|
| **A (tied)** | `(1−t)ε₀ + t·x̂₁(ε₀)` | `x̂₁ − ε₀` (10스텝 공통) | 모델의 **실제 결합**. `x_t`는 자기 ODE 경로의 현 위 점, 잔차 = 경로 곡률 |
| **B (independent)** | `(1−t)ε + t·x̂₁(ε₀)`, 스텝마다 새 ε | `x̂₁ − ε` (스텝마다 다름) | 데이터 분포만 모델 분포로 교체한 **정식 FM 목적함수** |

**A를 지지하는 논거**: 모델의 flow는 `ε₀ ↦ x̂₁`을 결정론적으로 사상하므로 `(ε₀, x̂₁)`만이 모델이
실제로 만들어내는 쌍이다. B는 `x̂₁`에 무관한 ε를 짝지으므로 모델의 flow가 결코 만들지 않을 쌍을
쓴다 — 없애려던 불일치를 "GT 대 모델"에서 "모델 대 모델"로 옮기는 셈이다.

**A를 의심하는 논거**: 학습이 경로를 곧게 폈다면 순간 속도 `v_θ(x_t,t)`가 평균 속도 `x̂₁ − ε₀`와
거의 같아져 잔차가 붕괴하고, `∂L/∂g = 2(≈0)ᵀ ∂v/∂g`가 되어 유닛이 기여도가 아니라 **잔차와의
정렬**로 정렬된다(`cache-preservation-is-a-chain`의 "자기 층 마스크 = 항등 refit"과 같은 종류).

**이 의심은 미측정 가정이다.** Alpamayo가 10스텝 Euler로 추론한다는 것은 경로가 완전한 직선은
아니라는 뜻이고, 곡률이 실제로 얼마인지는 이 저장소에 측정된 바 없다. 그러므로 **정하지 않고
잰다** — G1이 두 케이스의 g=1 손실 크기를 같은 클립·같은 `x̂₁`에서 나란히 재고, 숫자가 고른다.
부수적으로 이 측정 자체가 "Alpamayo의 flow가 얼마나 곧은가"라는 새 수치를 준다.

### G1 결과 (2026-09-10, cvlab20, calib_100 100클립) — **A 채택**

| | median | mean | gt 대비 |
|---|---:|---:|---:|
| `gt` (출하본) | **0.1932** | 0.2553 | — |
| **`A` (tied ε₀)** | **0.0578** | 0.0770 | **29.9%** |
| `B` (independent ε) | 0.1450 | 0.1709 | 75.1% |

임계값 0.019 대비 A가 0.0578로 **3.0배 위**. 퇴화 우려는 이 모델에서 실현되지 않는다.

이유도 같이 측정됐다 — **flow 직선성 `‖v − chord‖/‖chord‖ = 0.168 (median) / 0.187 (mean)`**.
Alpamayo의 학습된 field는 현에서 17~19% 벗어난다. 10스텝 Euler로 추론하는 모델이라 강하게
rectify될 이유가 없었고, 그래서 A의 잔차가 살아남는다. 이 저장소에 없던 수치다.

검증 셋 다 통과: `gt` median **0.1932**가 `importance_v2`의 같은 100클립 **0.1933**과 넷째
자리까지 일치(재구현이 출하본 손실을 재현), 샘플 궤적 ADE mean **1.844 m**가 500클립 단일-draw
기준 1.76 m와 같은 자릿수(Euler 재구현이 라이브러리 샘플러와 같은 분포), 매 클립 캐시 길이
복귀 assert 통과.

→ **본 실험은 A(tied)로 간다.** `outputs/selftraj_probe/`.

어느 쪽이 이기든 **`I_CoC`와 앵커가 대칭이 된다**는 이 실험의 동기는 그대로다.

**대가**: 기준이 GT를 전혀 모르게 되므로 모델을 옳게 만드는 유닛과 틀리게 만드는 유닛을
구별하지 못한다. teacher 보존이 목적인 압축에서는 맞는 타깃이지만 공짜가 아니며, 보고서에
명시한다.

### 구현 — drop-in

`analysis_lib.denoise_with_cache`가 **action space `(1, 64, 2)`를 그대로 반환**하므로
(docstring은 `pred_xyz/pred_rot`이라 적혀 있으나 코드는 `sampled_action`을 반환한다 — 함께
고친다), `expert_fm_grads`가 받는 `x1`과 같은 공간이다. 재인코딩도 왕복 오차도 없다.

`run_importance.py`에 `--traj-target {gt, self}` 추가:

```python
if args.traj_target == "self":
    offset = torch.tensor([prefill], device="cuda")
    prefix_mask = torch.ones(1, prefill, device="cuda", dtype=torch.long)
    eps0 = torch.randn(1, 64, 2, generator=Generator("cpu").manual_seed(seed ^ XOR))
    x1 = euler_sample(model, cache, rope_deltas, prefill, x0=eps0)   # (1, 64, 2)
    # A: expert_fm_grads(..., x1, coupling="tied", eps0=eps0)
    # B: expert_fm_grads(..., x1)          <- shipped path, fresh eps per step
```

`FlowMatching.sample`은 `@torch.no_grad`라 그래프를 만들지 않고, 매 스텝
`prompt_cache.crop(prefill)` 하므로 캐시는 원래 길이로 돌아온다(G0에서 확인). `expert_fm_grads`는
자기 `torch.Generator(device="cpu").manual_seed(seed)`로 ε를 뽑으므로 전역 RNG와 절연된다 —
`ε ⊥ ε₀`가 구조적으로 보장된다(G0에서 확인).

**K draws**: `K=10`. A(tied)에서는 한 드로우의 10스텝이 **하나의 현 위 10점**이라 서로
독립이 아니다 — 독립 노이즈 수는 10이 아니라 K다. 출하본 경로는 스텝마다 새 ε를 뽑아 클립당
**10개의 독립 draw**를 평균하므로, 그와 맞추려면 K=10이어야 한다(`expert_infer_grads`가 같은
이유로 `k_draws`를 둔다: "the training path effectively averages ten independent eps draws while
one Euler chain is a single draw"). 캐시 leaf에 누적한 뒤 **VLM backward는 한 번만** 한다.
프로브 실측으로 expert 평가 1회가 ~0.1 s이므로 클립당 200회(Euler 100 + 손실 100)라도 +20 s,
100클립 ~2 h 수준이다.

### 무엇을 만들 것인가

| 산출물 | 무엇 |
|---|---|
| `outputs/importance_selftraj_v1/` | K=4, calib_100, `--traj-target self` |
| `outputs/importance_selftraj_k1/` | K=1 대조 (분산 감소가 K에서 오는지 타깃에서 오는지 분리) |
| `slim_dualself_u40_v2` | `--config dual_u40_v2 --importance importance_selftraj_v1` |

**make_slim 변경 불필요.** `dual_u40_v2` stem에 `--importance`만 갈아끼우면 된다 — `dual_ada`와
`dual_u40_v2 + importance_v1 → slim_dual_uniform` 재현이 쓴 것과 같은 경로다.

## 사전 등록 게이트

| 게이트 | 내용 | 실패하면 |
|---|---|---|
| **G0** 계측 | ① `--traj-target gt`가 `importance_v2`를 **비트 단위 재현**(신규 경로가 꺼졌을 때 no-op) ② `x̂₁`이 `(1,64,2)`이고 캐시 길이가 `prefill`로 복귀 ③ 샘플 시드를 바꾸면 `x̂₁`은 바뀌고 ε 수열은 안 바뀜(`ε ⊥ ε₀` 확인) | 구현 버그. 중단 |
| **G1** 결합 선택 | 같은 클립·같은 `x̂₁`에서 A와 B의 g=1 FM 손실을 GT 앵커와 나란히 잰다. 기준선: `importance_v2` calib_100 median **0.1933** / mean 0.2564. **A가 median 0.019(1/10) 이상이면 A 채택**(모델의 실제 결합), 그 아래로 붕괴하면 **B 채택**, 애매하면 둘 다 빌드해 G2/G3로 가름 | 게이트가 아니라 **선택**이다 — 어느 쪽으로 갈리든 진행한다 |
| **G2** no-op 검사 | `dual_u40_v2`와의 kept 겹침이 재현성 천장(**0.997**, 오늘 측정: `run_importance`는 결정적이지 않다)보다 **뚜렷이 낮을 것**. 즉 "기준이 바뀌긴 했는가"만 본다 | 앵커 교체가 아무것도 안 바꾼 것. 보고하고 종료 |
| **G3** 안정성 (**본 가설**) | calib_100 50:50 split-half kept 겹침이 GT 앵커의 **Q 0.860 / MLP 0.782 이상**일 것. 같은 분할·같은 예산으로 비교 | H1 기각. 멀티모달 분산 논거가 틀린 것 — 그 자체로 보고 가치 있음 |
| **G4** 개루프 | `dualself_u40_v2` vs `dual_u40_v2`, val500, 페어드 중앙값 minADE@6 + 부트스트랩 CI 1차 | — (양방향 모두 결과) |

**G3가 이 실험의 본체다.** G2·G4는 "달라지긴 하는가/비용이 있는가"만 말하고, 제안된 *기전*
(모드 불일치가 분산을 만든다)을 검정하는 것은 G3뿐이다. G3는 GPU 추가 비용이 0이다 —
`importance_perclip.npz`의 클립별 누적을 반으로 갈라 argsort만 하면 된다.

G4는 **G2·G3를 통과한 뒤에만** 실행한다.

## 실행

- **GPU: Blackwell 0–3.** 중요도 측정은 평가가 아니고, 아키텍처가 결론을 바꾸지 않음이 이미
  확인돼 있다(`dual_ada` 재빌드 = no-op, median |0.003|, p≥0.68). 게다가 대조군
  `importance_v2` 자체가 **Blackwell에서 측정**됐으므로 Blackwell이 오히려 **일치하는** 선택이다.
  Ada 4–7은 폐루프가 점유 중이며 건드리지 않는다.
- 개루프(G4)만 Ada 4–7이 필요하고, 그때는 카드가 빌 때까지 대기한다.
- 캘리브레이션은 `calib_100` 그대로. **추출을 바꾸지 않는다** — 추출 분산이 이 실험의 효과보다
  크다는 것이 이미 측정돼 있다(`calib100-is-a-lucky-draw`).

## 검증

1. **G0을 먼저**, 2클립 스모크로. `--traj-target gt` 2클립이 `importance_v2`의 같은 2클립
   per-clip 누적과 비트 단위로 같은지 확인(신규 코드가 기본 경로를 건드리지 않았음을 증명).
2. `x̂₁`의 위생: 샘플된 궤적을 디코딩해 GT와의 ADE를 재고, 위 표의 단일-draw 분포
   (평균 1.76 m)와 같은 자릿수인지 확인. 크게 다르면 `denoise_with_cache` 사용법이 틀린 것.
3. G1 손실 크기를 per-clip으로 기록해 `importance_v2`의 `fm_loss`와 나란히 놓는다.
4. 결과는 `reports/evaluation/`에 HTML 한 편.

## 하지 않는 것

- 캘리브레이션 추출 변경 (교란)
- 스텝 정규화 (VLM 축은 `|Σ_s|`로 닫혀 있음, `expert-taylor-step-aggregation`)
- 결합을 논증으로 확정하는 것 (G1이 숫자로 고른다)
- expert 타워 중요도 변경 (`importance_stepexp_znorm` 그대로 — 이 실험은 VLM 축 전용)
