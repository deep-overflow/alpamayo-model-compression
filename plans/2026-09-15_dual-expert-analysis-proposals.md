# Dual objective / expert MLP 추가 분석 실험 제안

상태: 실험 제안. 신규 모델 평가·GPU 작업은 실행하지 않았다.

주장을 보강할 핵심 질문은 두 가지다. Dual이 서로 다른 능력에 필요한 구조를 실제로
보존하는지, 그리고 expert MLP를 크게 줄였을 때 어떤 계산은 유지되고 어떤 오차가 남는지다.
아래의 예상 패턴은 검증할 가설이며 이미 관찰한 결과가 아니다.

## 우선순위

| 우선순위 | 실험 | 검증할 주장 | 발표용 결과물 | 비용 성격 |
|---|---|---|---|---|
| 1 | Dual 유지집합을 같은 예산에서 교환 | 두 objective의 구조적 상보성이 실제 능력 차이를 만든다 | 교환 집합 × 두 loss heatmap, 교환량에 따른 성능 곡선 |
| 2 | Expert MLP의 상위·무작위·하위 채널 유지 비교 | 중요 채널이 일부에 집중되어 있어 압축 가능하다 | 남긴 채널 수 × 궤적 오차 곡선 |
| 3 | Expert MLP의 denoising 오차 전파와 step별 복원 | 작은 MLP 절단 오차가 최종 궤적까지 어떻게 전달되는가 | Step별 국소 오차·누적 오차, 복원 효과 heatmap |
| 4 | Best-of-K와 전체 샘플·꼬리 오차 재분석 | minADE@6 보존이 출력 분포의 보존도 의미하는가 | K별 성능 차이, clip별 오차 변화 분포 |
| 5 | 동일 calibration draw에서 Dual·Traj·CoC 반복 비교 | Dual의 상대적 이점이 calibration 선택에도 재현된다 | Draw별 paired difference 점·선 그래프 |
| 6 | 정규화를 고정한 max/mean 비교 | Dual의 max 연산 효과를 정규화 효과와 분리한다 | 2×2 ablation 표, 주행–VQA 산점도 |

4번은 저장된 sample별 ADE/FDE로 먼저 실행할 수 있다. 신규 모델 분석의 핵심은 1·2번이며,
3번은 작동 원리를 설명하는 그림에 적합하다. 5번은 논문 주장의 재현성에 중요한 확인이다.
6번은 rank-max 자체를 방법론적 기여로 강조할 경우 우선순위를 올린다.

## 1. Dual이 보호한 구조를 같은 예산에서 교환

### 질문

현재 importance·Spearman·overlap은 두 기준의 선택이 다르다는 사실을 보여준다.
다음 단계는 그 차이가 실제 CoC와 trajectory 계산에 어떤 영향을 주는지 측정하는 것이다.

### 설계

각 layer·axis의 Traj top-k, CoC top-k, 실제 Dual 유지집합을 각각 T, C, D라 한다.

- CoC 쪽에서 보호한 집합: `D ∩ (C − T)`.
- Trajectory 쪽에서 보호한 집합: `D ∩ (T − C)`.
- 공통 유지집합: `D ∩ T ∩ C`.

출발점은 plain Dual이다. 한 집합에서 m개를 제거하고, Dual이 제거했던 집합에서 m개를
복원한다. Q head와 MLP channel은 별도로 다루고 **같은 layer 안에서 같은 수를 교환**하여
layer별 예산과 전체 파라미터 수를 유지한다. 두 처치에서 가능한 공통 m을 사용하고,
복원할 후보도 동일하게 둔다. 같은 수의 무작위 교환을 대조군으로 추가한다.
가능하면 Dual 점수·cutoff 거리 구간도 맞춰 단순히 더 높은 점수의 유닛을 지운 효과를 줄인다.

초기에는 0 / 작은 교환량 / 큰 교환량의 세 수준으로 시작한다. Layer 22–34를 주 분석으로,
앞쪽 layer를 대조로 삼되 평가 성능을 보기 전에 layer 범위와 집합 크기를 결정한다.
공통 집합 수가 부족한 layer의 처리 규칙도 먼저 고정한다.
마지막 layer 35는 trajectory 점수가 구조적으로 상수이므로 모든 arm에서 같은 mask로 고정한다.

같은 입력, 고정 참조 CoC, 같은 FM noise·time grid에서 실제 ΔCoC NLL과 Δtrajectory FM loss를
측정한다. Calibration과 겹치지 않는 driving clips 100개 정도로 탐색하고,
선별된 대조를 더 큰 별도 평가 집합의 minADE/minFDE와 native LingoQA로 확인한다.
Teacher-forced CoC NLL 보존만으로 reasoning 능력 보존을 결론 내리지 않는다.

### 그림과 판정

행은 교환한 집합, 열은 ΔCoC NLL / Δtrajectory loss인 heatmap을 만든다.
서로 단위가 다른 두 loss는 별도 color scale을 쓰고 실제 수치를 병기한다.
교환량–오차 곡선에는 무작위 교환 반복의 분산과 clip 단위 paired CI를 표시한다.

CoC 쪽 집합을 교환했을 때 CoC 손상이 상대적으로 크고, trajectory 쪽에서는 반대라면
기능적 상보성의 근거가 된다. 두 loss가 함께 변하면 공유 기능과 상호작용으로 해석한다.
차이가 없으면 단순 overlap만으로 기능 분리를 주장하지 않는다.

기존 [단독 유닛 제거 분석](../reports/evaluation/2026-08-16_importance-calibration.html)은
25 clips에서 손상 순위의 재현성이 낮았다. 따라서 이 제안은 **실제 압축 상태의 집합 교환**이며,
단독 유닛의 Taylor 점수와 제거 손상 상관을 다시 그리는 실험과 구분한다.
기존 `run_imp_calib.py`, `mask_lib.py`의 고정 loss·mask 측정 경로를 활용할 수 있다.

## 2. Expert MLP의 중요 채널 집중을 직접 검증

### 질문과 설계

“MLP의 많은 채널을 지워도 된다”와 “어떤 채널을 남겨야 하는가”를 분리한다.
VLM을 plain Dual로 고정하고 expert Q heads를 모두 유지한 상태에서 다음을 비교한다.

| 조건 | 남길 채널 선택 |
|---|---|
| 현재 방법 | 최종 step별 z-score 평균 중요도의 상위 채널 |
| 무작위 대조 | Layer별 같은 개수를 무작위 선택, 최소 3개 mask seed |
| 반대 순위 대조 | 같은 중요도의 하위 채널을 유지 |
| 간단한 기준 | Weight magnitude 또는 이미 계산된 expert Wanda |

첫 비교는 layer당 2,064 / 516 channels 유지, 즉 75 / 93.75% 제거 두 점이면 된다.
극단적인 하위 채널 유지가 일찍 붕괴하면, 전체 기준이 무의미해질 정도의 큰 처치를 반복하기보다
상위 집합의 일부만 하위 채널로 교체하는 중간 수준을 추가한다.
Channel 수·layer 배분·CoC·초기 noise·샘플 수·expert Q 구조를 모두 동일하게 둔다.

### 그림과 해석

X축은 유지 채널 수, Y축은 minADE/minFDE와 sample 평균 오차다. 기준별 선을 그리고
무작위 선택의 mask seed 분산을 보여준다. 보조로 실제 inference에서 각 채널이
`down_proj` 출력에 기여하는 크기와 누적 분포를 측정한다. 채널 기여의 norm 합은
상쇄를 반영하지 않으므로 MLP 전체 출력 변화도 함께 표시한다.

상위 유지가 무작위·반대 순위보다 좋으면 중요한 채널의 집중과 선택 필요성이 뒷받침된다.
Magnitude/Wanda도 같다면 “Taylor만의 우월성”보다 “여러 기준으로 식별되는 중요한 소수 채널”이
적절한 주장이다. 기존 [expert MLP ladder](../reports/evaluation/2026-09-01_expert-mlp-ladder.html)는
Wanda/magnitude와 높은 kept overlap을 이미 보고했다. 이를 신규 발견처럼 반복하지 않고
plain Dual 위에서 무작위·반대 순위 대조와 실제 성능으로 확장한다.

## 3. Expert MLP의 denoising 오차 전파

### 설계

Plain Dual의 full expert와 MLP 75 / 87.5 / 93.75% 제거 expert를 비교한다.
모든 조건은 같은 Dual VLM cache, CoC, 초기 noise, 실제 inference time grid를 사용한다.
전체 10-step rollout에서 두 종류의 오차를 나누어 기록한다.

1. **국소 계산 차이:** Full expert의 같은 latent `x_s`를 두 expert에 넣었을 때 vector field 차이.
2. **누적 경로 차이:** 각 expert가 자신의 latent를 갱신한 실제 경로 간 차이.

Euler update가 `x_next = x + Δt · v(x, t)`일 때 경로 차이는 정확히

```text
δ_next = δ_s + Δt · [b_s + r_s]
b_s = v_pruned(x_full, t) − v_full(x_full, t)
r_s = v_pruned(x_pruned, t) − v_pruned(x_full, t)
```

로 분해한다. `b_s`는 같은 상태에서의 절단 효과이고 `r_s`는 달라진 상태를 평가하는 효과다.
실제 코드의 update 부호·step 간격을 그대로 따른다. Gradient 측정용 FM interpolation grid와
inference latent 경로를 섞지 않는다. Norm과 함께 벡터 내적·방향도 확인해 상쇄 가능성을 본다.

이어 동일한 고압축 mask에서 **초반 두 step / 중간 두 step / 마지막 두 step만 MLP를 복원**한다.
복원 step 수를 같게 하여 계산량 증가를 맞추고, 최종 궤적 차이와 GT 오차가 얼마나 줄어드는지 잰다.
이것은 원인 확인용 mask 개입이며, 물리적으로 잘린 checkpoint의 배포 속도 실험과 구분한다.

### 그림과 판정

Step × 제거율의 국소 오차 heatmap, step별 누적 latent/trajectory 차이 곡선,
복원 위치별 최종 오차 감소를 제시한다.
국소 차이도 작으면 낮은 기여의 채널을 제거한 설명을 지지한다.
국소 차이는 크지만 경로 차이가 줄면 이후 step의 상쇄·보정 가능성을 추가 검증한다.
후반에 누적 차이가 커지고 해당 step 복원이 효과가 있으면 그 구간의 민감성을 확인할 수 있다.
어느 패턴이 나올지는 현재 결과만으로 단정하지 않는다.

기존 [run_step_mask.py](../experiments/head_analysis/run_step_mask.py)는 이전 trajectory/magnitude 기준의
Q+MLP mask를 step별로 적용했다. 이번에는 **현재 MLP-only 고압축 mask와 plain Dual parent**,
그리고 국소 계산 차이/누적 상태 차이의 분해가 추가된다.

## 4. Best-of-K가 숨기는 차이 재분석

현재 expert MLP 표는 주로 minADE@6·minFDE@6를 보여준다. 우수한 샘플 하나가 남아 있으면
다른 샘플들의 품질 저하가 잘 드러나지 않을 수 있다. 다음 분석은 저장된
`ade_rollout_k`·`fde_rollout_k` 배열만으로 시작한다.

- 처음 6개 샘플에서 `K = 1, 2, 4, 6`의 기대 minADE/minFDE를 계산한다.
  첫 K개만 취하는 방법 외에, 6개에서 K개를 고르는 모든 조합의 평균도 사용한다.
  모델 간에는 같은 sample index 조합을 쓴다.
- 6개 샘플 전체의 평균 ADE/FDE, clip별 paired 차이의 median·95th percentile을 보고한다.
- 사전에 정한 허용 오차보다 악화된 clip 비율도 확인한다. 허용치를 결과를 보고 선택하지 않는다.
- Bootstrap 단위는 clip이다. 같은 clip의 6개 샘플을 독립된 평가 사례로 세지 않는다.

그림은 K별 `pruned − Dual` 곡선과 clip별 오차 변화의 누적분포다.
K=1·전체 평균·꼬리에서도 변화가 작다면 출력 품질 보존의 범위가 넓어진다.
K=6만 유지된다면 주장을 best-of-6 성능으로 한정하고, closed-loop 저하의 후보 원인으로
샘플 분포 변화가 실제 실행 trajectory에 반영되는지 후속 확인한다.
ADE/FDE 배열만으로 공간적 trajectory 다양성이나 충돌 위험 자체를 측정했다고 설명하지 않는다.

## 5. Calibration draw를 공유한 Dual–단일 목적 비교

기존 [action 층화 calibration 분석](../reports/evaluation/2026-09-12_action-stratified-calib.html)은
여러 추출 규칙·seed의 Dual을 이미 평가했다. 따라서 단순히 다른 calibration에서 Dual을
다시 평가하는 것보다 **각 draw 안에서 Dual, Traj, CoC를 함께 만드는 비교**가 필요하다.

동일한 pool·sample 수 100·구성 규칙으로 독립 draw 3–5개를 사용한다. 각 draw의 두 objective
중요도로 세 mask를 만들고 예산과 마지막 layer 처리 규칙을 맞춘다.
기존 draw별 중요도에 두 신호가 저장되어 있으면 재사용한다.
Mask seed, calibration draw, generation seed를 별도로 기록한다.

각 draw에서 `Dual − Traj`, `Dual − CoC`의 open-loop·LingoQA 차이를 계산하고,
closed-loop까지 주장할 비교는 같은 scene 집합에서도 반복한다.
X축 draw, Y축 paired performance difference인 그래프에 각 draw를 표시한다.
좋은 draw만 선택하지 않고 모든 draw의 결과와 분산을 보여준다.
Clip 수를 늘린 것을 독립 calibration 반복 수를 늘린 것처럼 처리하지 않는다.

기준 선택에 사용한 validation 분포와 최종 평가 분포를 구분한다. 이미 반복적으로 살펴본
Test500에서 실험 설정을 고른 경우 탐색 결과임을 명시하고, 확증에는 미사용 clip 집합을 둔다.

## 6. Max와 정규화 효과를 분리

기존 [union-step 비교](2026-09-03_union-step-criterion.md)에서는 max 쪽은 rank,
mean 쪽은 z-score를 써서 연산자와 정규화가 함께 바뀌었다. 현재 비교를 보강하려면
먼저 CoC·합산 trajectory 두 신호만으로 아래 네 칸을 맞춘다.

| 정규화 | Max | Mean |
|---|---|---|
| 같은 layer 내부 rank | 현재 Dual 규칙 | 두 rank의 동일 가중 평균 |
| 같은 layer 내부 z-score | 두 z-score의 max | 두 z-score의 동일 가중 평균 |

Importance 데이터·예산·layer allocation·tie 처리·상수 점수 처리는 모두 동일하게 둔다.
2-way와 11-way까지 동시에 확대하지 않고, 두 신호 조건에서 연산자 효과부터 분리한다.
VLM 중요도 정규화와 expert MLP의 step별 z-score 집계를 혼동하지 않는다.

Open-loop–LingoQA 평면에서 네 점을 비교하고 closed-loop는 대표 대조를 확인한다.
Max가 두 능력의 극단적 손상을 줄이면 해당 조건의 장점으로 설명한다.
가중 평균도 같다면 max가 유일하게 필요한 규칙이라는 주장은 피한다.

## 실행 순서와 발표 구성

1. 저장된 결과로 4번과 교환 가능한 집합 크기를 확인한다.
2. 1·2번을 작은 독립 probe 집합에서 실시한다. Mask 적용과 동일 입력·dtype 경로를 검증한다.
3. 미리 정한 대표 대조를 더 큰 별도 집합에서 평가하고, 3번으로 expert의 계산 변화를 설명한다.
4. 5번의 draw별 비교로 Dual의 상대적 이점이 재현되는지 확인한다.
5. 본문에는 Dual 집합 교환과 expert 채널 선택/오차 전파 중 가장 직접적인 그림을 한 장씩 추가한다.
   15분 발표에서는 기존 상관 그림 일부를 교체하고, 모든 신규 ablation을 본문에 넣지는 않는다.

모든 비교는 같은 GPU architecture, model revision, token/CoC 조건, generation seed, clip/scene ID를
기록한다. 고정 CoC에서의 loss probe와 각 모델이 직접 생성하는 end-to-end 평가를 구분한다.
기존 audit에서 확인된 refit state 누락을 피하도록 이 제안의 주 parent는 plain Dual로 둔다.
Hard100에서는 사전에 정한 실행 실패 처리와 공통 유효 관측 집합을 함께 보고한다.

개입 실험은 corruption·복원 방식과 측정 지표에 따라 해석이 달라질 수 있어 대조군과 판독 지표를
함께 정해야 한다는 방법론적 근거가 있다.
[Zhang & Nanda, Towards Best Practices of Activation Patching (ICLR 2024)](https://arxiv.org/abs/2309.16042).
위의 Alpamayo 실험 설계는 이 저장소의 결과에 맞춰 제안한 것이며 해당 논문이 검증한 결과가 아니다.
