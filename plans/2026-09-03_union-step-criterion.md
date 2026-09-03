# 스텝별 랭크의 합집합: znorm11이 한 번에 바꾼 두 요인을 분리한다

## 질문

`znorm11`(11개 손실의 층내 z-score 평균)은 `dual`(`max(rank I_traj, rank I_CoC)`) 대비
+0.14~+0.20으로 기각됐다(`reports/evaluation/2026-09-03_criterion-aggregation.html`).
그런데 znorm11은 **두 가지를 동시에** 바꿨다.

1. **궤적 표현**: 합산된 `I_traj` 하나 → 디노이징 스텝별 10개 점수
2. **결합 연산**: `max`(합집합) → `mean`(평균)

기각의 원인을 (2)로 귀속했지만, 그것은 해석이지 측정이 아니다. 2×2를 채운다.

| | 2-way (합산 traj) | 11-way (스텝별) |
|---|---|---|
| **max** | `dualfix` ✔ 측정됨 | **`maxstep11`** ← 신규 |
| **mean** | **`meandual`** ← 신규 | `znorm11` ✔ 측정됨 |

- `maxstep11` = `max(rank FM_0, …, rank FM_9, rank I_CoC)` — 11개 랭크의 합집합
- `meandual` = `mean(z(I_traj), z(I_CoC))` — dual의 두 half를 znorm11의 연산자로

두 신규 arm 모두 `dualfix`의 상수-층 가드를 상속한다(층 35는 FM 10개가 전부 상수라 CoC가
단독 결정 — `dualfix`의 층 35 kept set과 **비트 단위로 동일**함을 확인).

## 사전 정보 (빌드 전에 이미 아는 것)

- `sum_s(q_abs_step)`은 `traj_vlm_q`를 층내 Spearman **+0.989**로 재현한다(MLP +0.984).
  두 축은 실제로 분리 가능하다.
- **스텝 간 랭킹이 이미 매우 비슷하다**: Q head 층내 페어와이즈 Spearman 평균 **+0.920**
  (최소 +0.812). expert에서 스텝 축이 컸던 것과 대조적이다. 따라서 (1)의 여지 자체가 얇다는
  것이 사전 예측이고, 이 실험은 그 예측을 확인하거나 뒤집는다.
- kept set 사전 계산(예산 동일, 19/32 Q · 7390/12288 MLP):

| arm | dualfix 대비 Q 일치 | znorm11 대비 Q 일치 | 파라미터 churn |
|---|---|---|---|
| `maxstep11` | 0.9254 | 0.9211 | 13.7% |
| `meandual` | 0.9415 | 0.8874 | 8.6% |
| `znorm11` | 0.8772 | — | 17.2% |

`maxstep11`은 두 기존 셀의 정확히 중간에 놓인다 — 어느 쪽으로 붙든 정보량이 있다.

## 게이트 (사전 등록)

기준선은 `dualfix`(= 실질적으로 출시 dual), 지표는 paired minADE@6 중앙값, 세 세트
(val500 / test500 / OOD-val 262), 부트스트랩 95% CI.

- **B1 (주 판정)** `|median(maxstep11 − dualfix)| < 0.05` 가 **세 세트 모두**에서 성립
  → 스텝 축은 합집합 연산 아래에서 **무해**하고, znorm11의 손실은 전적으로 연산자 때문이다.
  세 세트 중 하나라도 위반하면 → 11-way라는 **arity 자체**가(합집합이어도) 해롭다.
- **B2 (보조)** `median(meandual − dualfix)`의 크기가 `median(znorm11 − dualfix)`의
  **절반 이상**이면 평균 연산자가 단독으로 대부분의 손해를 설명한다. 그보다 작으면
  손해는 연산자와 arity의 **상호작용**이다.
- **B3** 두 신규 arm 모두 CoC 퇴화율 < 10%. 넘으면 해당 arm의 B1/B2 판정은 무효.

**B1이 PASS이고 `maxstep11`이 `dualfix`보다 유의하게 낫지도 않다면**, 결론은 "VLM에서 스텝 축은
살릴 것이 없다"이다. 사전 정보(스텝 간 Spearman 0.92)가 그쪽을 가리키므로, 이 실험의 주된 가치는
znorm11 기각의 원인을 확정하는 데 있지 새 SOTA를 기대하는 데 있지 않다.

## 실행

```
make_slim.py --config maxstep11_u40_v2 --importance importance_v2_ada \
             --stepvlm importance_stepvlm_v1 --no-state --out outputs/slim_maxstep11_u40_v2
make_slim.py --config meandual_u40_v2  --importance importance_v2_ada \
             --no-state --out outputs/slim_meandual_u40_v2
```

평가는 `launch_arms.sh append 4`로 2 arm × 3 세트 × 4 shard = 24 job, **Ada 4–7**
(기존 arm과 페어링하려면 아키텍처를 고정해야 한다). 분석은
`analyze_criterion_agg.py`에 두 arm을 추가해 같은 표로 확장한다.

## 검증

1. 두 arm 모두 제거 파라미터가 **정확히 2,657,452,032**여야 한다(다르면 예산이 어긋난 것).
2. 층 35 kept set이 `dualfix`와 동일해야 한다(가드가 상속됐다는 뜻). 빌드 전 확인 완료.
3. 시드는 clip 유도이므로 기존 arm과 자동으로 페어링된다 — 공통 클립에서 시드 불일치 0건 확인.
